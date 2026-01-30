#!/usr/bin/env python3
"""
PersonaPlex Desktop - Real-Time Full-Duplex Streaming Server
Matches the original PersonaPlex/Moshi binary protocol exactly.

Binary Protocol:
  0x00 = handshake (server sends when ready)
  0x01 = audio (followed by Opus bytes)
  0x02 = text (followed by UTF-8 text)
  0x03 = control message
  0x04 = metadata (JSON)
  0x05 = error
"""

import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['NO_TORCH_COMPILE'] = '1'
os.environ['NO_CUDA_GRAPH'] = '1'  # Disable CUDA graphs - incompatible with CPU offloading
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import asyncio
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Callable
from dataclasses import dataclass

import numpy as np
import torch
import aiohttp
from aiohttp import web
from aiohttp.web import middleware
import sphn
import sentencepiece


@middleware
async def cors_middleware(request, handler):
    """Add CORS headers to all responses."""
    if request.method == 'OPTIONS':
        response = web.Response()
    else:
        response = await handler(request)

    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

torch._dynamo.config.disable = True
torch._dynamo.reset()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)
DB_PATH = Path(__file__).parent / "conversations.db"
MODELS_DIR = Path(__file__).parent / "models"

MSG_HANDSHAKE = 0x00
MSG_AUDIO = 0x01
MSG_TEXT = 0x02
MSG_CONTROL = 0x03
MSG_METADATA = 0x04
MSG_ERROR = 0x05


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


@dataclass
class Conversation:
    id: int
    created_at: str
    voice_preset: str
    persona_prompt: str
    transcript: list
    audio_path: Optional[str]
    duration_seconds: float


class StorageManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    voice_preset TEXT NOT NULL,
                    persona_prompt TEXT,
                    transcript TEXT,
                    audio_path TEXT,
                    duration_seconds REAL
                )
            """)
            conn.commit()

    def save_conversation(self, voice_preset: str, persona_prompt: str,
                         transcript: list, audio_path: Optional[str] = None,
                         duration_seconds: float = 0.0) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO conversations
                   (voice_preset, persona_prompt, transcript, audio_path, duration_seconds)
                   VALUES (?, ?, ?, ?, ?)""",
                (voice_preset, persona_prompt, json.dumps(transcript),
                 audio_path, duration_seconds)
            )
            conn.commit()
            return cursor.lastrowid

    def list_conversations(self, limit: int = 50, offset: int = 0) -> list:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT id, created_at, voice_preset, persona_prompt, duration_seconds
                   FROM conversations ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (limit, offset)
            ).fetchall()
            return [
                {
                    "id": row[0],
                    "created_at": row[1],
                    "voice_preset": row[2],
                    "persona_preview": row[3][:100] if row[3] else "",
                    "duration_seconds": row[4] if row[4] else 0.0
                }
                for row in rows
            ]


class ServerState:
    """
    Manages the PersonaPlex model and handles streaming conversations.
    Closely follows the original Moshi server implementation.
    """

    def __init__(self, mimi, other_mimi, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm, device, voice_prompt_dir: str):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        from moshi.models import LMGen
        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=False,
        )

        self.lock = asyncio.Lock()
        # Don't initialize streaming here - do it per-connection
        # This avoids issues with accelerate's dispatch_model
        self._streaming_initialized = False

        self.storage = StorageManager()
        logger.info(f"ServerState initialized on {device}")

    def _ensure_streaming(self):
        """Initialize streaming state on first use."""
        if not self._streaming_initialized:
            logger.info("Initializing streaming state...")
            self.mimi.streaming_forever(1)
            self.other_mimi.streaming_forever(1)
            self.lm_gen.streaming_forever(1)
            self._streaming_initialized = True
            logger.info("Streaming state initialized")

    def warmup(self):
        """Warmup the model with dummy frames."""
        logger.info("Warming up model...")

        # Initialize streaming state first
        self._ensure_streaming()
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()

        # Mimi is on CPU, LMGen is on GPU
        mimi_device = next(self.mimi.parameters()).device
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=mimi_device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                # Move codes to GPU for LMGen
                codes_gpu = codes[:, :, c: c + 1].to(self.device)
                tokens = self.lm_gen.step(codes_gpu)
                if tokens is None:
                    continue
                # Move tokens back to CPU for Mimi decode
                tokens_cpu = tokens.cpu()
                _ = self.mimi.decode(tokens_cpu[:, 1:9])
                _ = self.other_mimi.decode(tokens_cpu[:, 1:9])

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        logger.info("Warmup complete")

    async def handle_chat(self, request):
        """Handle a WebSocket chat connection."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        peer = request.remote
        logger.info(f"Incoming connection from {peer}")

        voice_prompt = request.query.get("voice_prompt", "NATF1.pt")
        text_prompt = request.query.get("text_prompt", "")

        voice_prompt_path = None
        if self.voice_prompt_dir:
            voice_prompt_path = os.path.join(self.voice_prompt_dir, voice_prompt)
            if not os.path.exists(voice_prompt_path):
                logger.error(f"Voice prompt not found: {voice_prompt_path}")
                await ws.send_bytes(bytes([MSG_ERROR]) + b"Voice prompt not found")
                await ws.close()
                return ws

        if voice_prompt_path and self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith('.pt'):
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)

        if text_prompt:
            self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(text_prompt))
        else:
            self.lm_gen.text_prompt_tokens = None

        close = False

        async def recv_loop(opus_reader):
            nonlocal close
            async for message in ws:
                if message.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
                elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                    break
                elif message.type != aiohttp.WSMsgType.BINARY:
                    logger.warning(f"Unexpected message type: {message.type}")
                    continue

                data = message.data
                if not isinstance(data, bytes) or len(data) == 0:
                    continue

                kind = data[0]
                payload = data[1:]

                if kind == MSG_AUDIO:
                    opus_reader.append_bytes(payload)
                elif kind == MSG_CONTROL:
                    logger.info(f"Control message: {payload}")
                elif kind == MSG_METADATA:
                    logger.info(f"Metadata: {payload.decode('utf-8', errors='replace')}")
                else:
                    logger.warning(f"Unknown message kind: {kind}")

            close = True
            logger.info("Receive loop ended")

        async def opus_loop(opus_reader, opus_writer):
            nonlocal close
            all_pcm_data = None
            frame_count = 0
            # Mimi is on CPU, LMGen is on GPU
            mimi_device = next(self.mimi.parameters()).device

            while not close:
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()

                if pcm.shape[-1] == 0:
                    continue

                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))

                while all_pcm_data.shape[-1] >= self.frame_size:
                    frame_count += 1
                    if frame_count <= 5:
                        logger.info(f"Processing frame {frame_count}")

                    chunk = all_pcm_data[:self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]

                    # Mimi encode on CPU
                    chunk_tensor = torch.from_numpy(chunk)[None, None]  # Shape: [1, 1, frame_size]
                    codes = self.mimi.encode(chunk_tensor)
                    _ = self.other_mimi.encode(chunk_tensor)

                    for c in range(codes.shape[-1]):
                        # Move codes to GPU for LMGen
                        codes_gpu = codes[:, :, c: c + 1].to(self.device)
                        tokens = self.lm_gen.step(codes_gpu)
                        if tokens is None:
                            continue

                        # Move tokens to CPU for Mimi decode
                        tokens_cpu = tokens.cpu()
                        main_pcm = self.mimi.decode(tokens_cpu[:, 1:9])
                        _ = self.other_mimi.decode(tokens_cpu[:, 1:9])
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())

                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            text_piece = self.text_tokenizer.id_to_piece(text_token)
                            text_piece = text_piece.replace("▁", " ")
                            msg = bytes([MSG_TEXT]) + text_piece.encode('utf-8')
                            await ws.send_bytes(msg)

            logger.info(f"Opus loop ended after {frame_count} frames")

        async def send_loop(opus_writer):
            nonlocal close
            while not close:
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(bytes([MSG_AUDIO]) + msg)

        async with self.lock:
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)

            # Initialize streaming state on first connection
            self._ensure_streaming()

            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()

            async def is_alive():
                if close or ws.closed:
                    return False
                return True

            logger.info("Priming model with system prompts...")
            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            logger.info("System prompts complete")

            if await is_alive():
                await ws.send_bytes(bytes([MSG_HANDSHAKE]))
                logger.info("Sent handshake, starting audio loops")

                tasks = [
                    asyncio.create_task(recv_loop(opus_reader)),
                    asyncio.create_task(opus_loop(opus_reader, opus_writer)),
                    asyncio.create_task(send_loop(opus_writer)),
                ]

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                await ws.close()

        logger.info("Connection closed")
        return ws

    async def handle_api(self, request):
        """Handle REST API requests for voices and history."""
        action = request.match_info.get('action')

        if action == 'voices':
            voices = []
            voices_dir = Path(self.voice_prompt_dir)
            if voices_dir.exists():
                for voice_file in sorted(voices_dir.glob("*.pt")):
                    voice_name = voice_file.stem
                    voice_type = "Natural" if voice_name.startswith("NAT") else "Variety"
                    gender = "Female" if "F" in voice_name else "Male"
                    voices.append({
                        "id": voice_name,
                        "name": f"{voice_type} {gender} {voice_name[-1]}",
                        "type": voice_type,
                        "gender": gender,
                        "file": voice_file.name
                    })
            return web.json_response({"voices": voices})

        elif action == 'history':
            limit = int(request.query.get('limit', 50))
            offset = int(request.query.get('offset', 0))
            conversations = self.storage.list_conversations(limit, offset)
            return web.json_response({"conversations": conversations})

        elif action == 'save':
            # Handle POST request to save conversation
            if request.method != 'POST':
                return web.json_response({"error": "Method not allowed"}, status=405)
            try:
                data = await request.json()
                voice_preset = data.get('voice_preset', 'unknown')
                persona_prompt = data.get('persona_prompt', '')
                transcript = data.get('transcript', [])

                conv_id = self.storage.save_conversation(
                    voice_preset=voice_preset,
                    persona_prompt=persona_prompt,
                    transcript=transcript
                )
                logger.info(f"Saved conversation {conv_id}")
                return web.json_response({"id": conv_id, "success": True})
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")
                return web.json_response({"error": str(e)}, status=500)

        return web.json_response({"error": "Unknown action"}, status=400)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu-offload", action="store_true", default=True)
    parser.add_argument("--quantize-4bit", action="store_true", default=False,
                       help="Use 4-bit quantization (experimental, may not work)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    from moshi.models import loaders

    logger.info("Loading Mimi audio codec (on CPU to save VRAM)...")
    mimi_path = str(MODELS_DIR / "tokenizer-e351c8d8-checkpoint125.safetensors")
    mimi = loaders.get_mimi(mimi_path, 'cpu')
    other_mimi = loaders.get_mimi(mimi_path, 'cpu')
    logger.info("Mimi loaded on CPU")

    logger.info("Loading text tokenizer...")
    text_tokenizer = sentencepiece.SentencePieceProcessor()
    text_tokenizer.load(str(MODELS_DIR / "tokenizer_spm_32k_3.model"))

    logger.info("Loading PersonaPlex model...")
    moshi_path = str(MODELS_DIR / "model.safetensors")
    lm = loaders.get_moshi_lm(
        moshi_path,
        device=device,
        cpu_offload=args.cpu_offload,
        quantize_4bit=args.quantize_4bit
    )
    lm.eval()
    logger.info("PersonaPlex model loaded")

    # Reduce context window to fit in 16GB VRAM with CPU offloading
    logger.info("Setting context window to 1500 tokens...")
    from moshi.modules.transformer import set_attention_context
    set_attention_context(lm, 1500)

    voice_prompt_dir = str(MODELS_DIR / "voices")

    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=device,
        voice_prompt_dir=voice_prompt_dir,
    )

    # Warmup the model (skip for CPU offloading - accelerate hooks cause issues)
    if args.cpu_offload:
        logger.info("Skipping warmup (CPU offload mode)")
    else:
        state.warmup()

    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get("/api/chat", state.handle_chat)
    app.router.add_get("/api/{action}", state.handle_api)
    app.router.add_post("/api/{action}", state.handle_api)

    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"WebSocket endpoint: ws://{args.host}:{args.port}/api/chat")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    with torch.no_grad():
        main()
