#!/usr/bin/env python3
"""
PersonaPlex Desktop - Real-Time Full-Duplex Streaming Server
Compatible with moshi 0.2.12 API.

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

import asyncio
import json
import sqlite3
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
import torch
import aiohttp
from aiohttp import web
from aiohttp.web import middleware
import sphn
import sentencepiece

torch._dynamo.config.disable = True
torch._dynamo.reset()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "conversations.db"
MODELS_DIR = Path(__file__).parent / "models"

MSG_HANDSHAKE = 0x00
MSG_AUDIO = 0x01
MSG_TEXT = 0x02
MSG_CONTROL = 0x03
MSG_METADATA = 0x04
MSG_ERROR = 0x05


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
    Based on the official moshi server implementation for moshi 0.2.12.
    """

    def __init__(self, mimi, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm_gen, device, voice_prompt_dir: str):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.lm_gen = lm_gen
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lock = asyncio.Lock()
        self.storage = StorageManager()

        # Initialize streaming
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        logger.info(f"ServerState initialized on {device}")

    def warmup(self):
        """Warmup the model with dummy frames."""
        logger.info("Warming up model...")

        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])

        torch.cuda.synchronize()
        logger.info("Warmup complete")

    async def handle_chat(self, request):
        """Handle a WebSocket chat connection with concurrent loops."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        peer = request.remote
        logger.info(f"Incoming connection from {peer}")

        voice_prompt = request.query.get("voice_prompt", "default")
        text_prompt = request.query.get("text_prompt", "")
        logger.info(f"Requested voice: {voice_prompt}, persona length: {len(text_prompt)}")

        close = False

        async def recv_loop(opus_reader):
            """Receive WebSocket messages and buffer audio data."""
            nonlocal close
            audio_packets_received = 0
            async for message in ws:
                if message.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
                elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                    break
                elif message.type != aiohttp.WSMsgType.BINARY:
                    continue

                data = message.data
                if not isinstance(data, bytes) or len(data) == 0:
                    continue

                kind = data[0]
                payload = data[1:]

                if kind == MSG_AUDIO:
                    opus_reader.append_bytes(payload)
                    audio_packets_received += 1
                    if audio_packets_received <= 5 or audio_packets_received % 100 == 0:
                        logger.info(f"Received audio packet #{audio_packets_received} ({len(payload)} bytes)")
                elif kind == MSG_CONTROL:
                    logger.info(f"Control message: {payload}")

            close = True
            logger.info(f"Receive loop ended after {audio_packets_received} audio packets")

        async def process_loop(opus_reader, opus_writer):
            """Process audio through the model."""
            nonlocal close
            all_pcm_data = None
            frame_count = 0
            text_tokens_sent = 0

            while not close:
                await asyncio.sleep(0.001)  # Yield to other tasks
                pcm = opus_reader.read_pcm()

                if pcm is None or pcm.shape[-1] == 0:
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

                    chunk_tensor = torch.from_numpy(chunk).to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk_tensor)

                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue

                        # Decode audio
                        main_pcm = self.mimi.decode(tokens[:, 1:])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())

                        # Send text token
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            text_piece = self.text_tokenizer.id_to_piece(text_token)
                            text_piece = text_piece.replace("▁", " ")
                            await ws.send_bytes(bytes([MSG_TEXT]) + text_piece.encode('utf-8'))
                            text_tokens_sent += 1
                            if text_tokens_sent <= 10:
                                logger.info(f"Sent text token #{text_tokens_sent}: '{text_piece}'")

            logger.info(f"Process loop ended after {frame_count} frames, {text_tokens_sent} text tokens")

        async def send_loop(opus_writer):
            """Send encoded audio back to client."""
            nonlocal close
            audio_packets_sent = 0
            while not close:
                await asyncio.sleep(0.001)  # Yield to other tasks
                opus_bytes = opus_writer.read_bytes()
                if opus_bytes is not None and len(opus_bytes) > 0:
                    await ws.send_bytes(bytes([MSG_AUDIO]) + opus_bytes)
                    audio_packets_sent += 1
                    if audio_packets_sent <= 5 or audio_packets_sent % 100 == 0:
                        logger.info(f"Sent audio packet #{audio_packets_sent} ({len(opus_bytes)} bytes)")

            logger.info(f"Send loop ended after {audio_packets_sent} audio packets")

        async with self.lock:
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)

            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()

            # Load voice prompt embeddings if available
            if voice_prompt and voice_prompt != "default":
                voice_path = Path(self.voice_prompt_dir) / voice_prompt
                if voice_path.exists():
                    try:
                        if hasattr(self.lm_gen, 'load_voice_prompt_embeddings'):
                            self.lm_gen.load_voice_prompt_embeddings(str(voice_path))
                            logger.info(f"Loaded voice embeddings: {voice_prompt}")

                            if hasattr(self.lm_gen, 'step_system_prompts'):
                                self.lm_gen.step_system_prompts(self.mimi)
                                self.mimi.reset_streaming()
                                logger.info("Applied voice conditioning")
                    except Exception as e:
                        logger.error(f"Failed to load voice prompt: {e}")

            # Send handshake
            await ws.send_bytes(bytes([MSG_HANDSHAKE]))
            logger.info("Sent handshake, starting concurrent audio loops")

            # Run three concurrent loops
            tasks = [
                asyncio.create_task(recv_loop(opus_reader)),
                asyncio.create_task(process_loop(opus_reader, opus_writer)),
                asyncio.create_task(send_loop(opus_writer)),
            ]

            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

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
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    from moshi.models import loaders, LMGen

    logger.info("Loading Mimi audio codec...")
    mimi_path = str(MODELS_DIR / "tokenizer-e351c8d8-checkpoint125.safetensors")
    mimi = loaders.get_mimi(mimi_path, device=device)
    logger.info("Mimi loaded")

    logger.info("Loading text tokenizer...")
    text_tokenizer = sentencepiece.SentencePieceProcessor()
    text_tokenizer.load(str(MODELS_DIR / "tokenizer_spm_32k_3.model"))

    logger.info("Loading PersonaPlex model...")
    moshi_path = str(MODELS_DIR / "model.safetensors")
    # Use cpu_offload=True to avoid meta tensor issues with PersonaPlex loader
    # (A100 40GB has enough VRAM, accelerate will just put everything on GPU)
    lm = loaders.get_moshi_lm(
        moshi_path,
        device=device,
        dtype=torch.bfloat16,
        cpu_offload=True
    )
    lm.eval()
    logger.info("PersonaPlex model loaded")

    # Create LMGen with PersonaPlex's API
    # text_prompt_tokens=[] to avoid NoneType iteration error in step_system_prompts
    lm_gen = LMGen(lm, device=device, text_prompt_tokens=[])

    voice_prompt_dir = str(MODELS_DIR / "voices")

    state = ServerState(
        mimi=mimi,
        text_tokenizer=text_tokenizer,
        lm_gen=lm_gen,
        device=device,
        voice_prompt_dir=voice_prompt_dir,
    )

    # Warmup the model
    state.warmup()

    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get("/api/chat", state.handle_chat)
    app.router.add_get("/api/{action}", state.handle_api)
    app.router.add_post("/api/{action}", state.handle_api)

    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"WebSocket endpoint: ws://{args.host}:{args.port}/api/chat")
    logger.info("Voice selection: PersonaPlex moshi installed - 18 voices available")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    with torch.no_grad():
        main()
