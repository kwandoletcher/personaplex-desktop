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

    async def decode_and_send(self, tokens: torch.Tensor, ws: web.WebSocketResponse,
                              opus_writer: sphn.OpusStreamWriter):
        """Decode tokens to audio and send via WebSocket."""
        main_pcm = self.mimi.decode(tokens[:, 1:])
        main_pcm = main_pcm.cpu()
        opus_bytes = opus_writer.append_pcm(main_pcm[0, 0].numpy())
        if len(opus_bytes) > 0:
            await ws.send_bytes(bytes([MSG_AUDIO]) + opus_bytes)

        text_token = tokens[0, 0, 0].item()
        if text_token not in (0, 3):
            text_piece = self.text_tokenizer.id_to_piece(text_token)
            text_piece = text_piece.replace("▁", " ")
            await ws.send_bytes(bytes([MSG_TEXT]) + text_piece.encode('utf-8'))

    async def recv_loop(self, ws: web.WebSocketResponse,
                        opus_reader: sphn.OpusStreamReader,
                        opus_writer: sphn.OpusStreamWriter):
        """Main receive and process loop."""
        all_pcm_data = None
        skip_frames = 1
        frame_count = 0

        try:
            async for message in ws:
                if message.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
                elif message.type == aiohttp.WSMsgType.CLOSED:
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
                    pcm = opus_reader.append_bytes(payload)
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

                        be = time.time()
                        chunk = all_pcm_data[:self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size:]

                        chunk_tensor = torch.from_numpy(chunk).to(device=self.device)[None, None]
                        codes = self.mimi.encode(chunk_tensor)

                        if skip_frames:
                            # First frame is in the past from model's perspective
                            self.mimi.reset_streaming()
                            skip_frames -= 1

                        for c in range(codes.shape[-1]):
                            tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                            if tokens is None:
                                continue
                            await self.decode_and_send(tokens, ws, opus_writer)

                        logger.debug(f"Frame handled in {1000 * (time.time() - be):.1f}ms")

                elif kind == MSG_CONTROL:
                    logger.info(f"Control message: {payload}")
                elif kind == MSG_METADATA:
                    logger.info(f"Metadata: {payload.decode('utf-8', errors='replace')}")
                else:
                    logger.warning(f"Unknown message kind: {kind}")

        finally:
            logger.info(f"Connection closed after {frame_count} frames")

    async def handle_chat(self, request):
        """Handle a WebSocket chat connection."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        peer = request.remote
        logger.info(f"Incoming connection from {peer}")

        voice_prompt = request.query.get("voice_prompt", "default")
        text_prompt = request.query.get("text_prompt", "")
        logger.info(f"Requested voice: {voice_prompt}, persona length: {len(text_prompt)}")

        async with self.lock:
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)

            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()

            # Load voice prompt embeddings if available (requires PersonaPlex moshi)
            voice_applied = False
            if voice_prompt and voice_prompt != "default":
                voice_path = Path(self.voice_prompt_dir) / voice_prompt
                if voice_path.exists():
                    try:
                        # PersonaPlex API: load_voice_prompt_embeddings for .pt files
                        if hasattr(self.lm_gen, 'load_voice_prompt_embeddings'):
                            self.lm_gen.load_voice_prompt_embeddings(str(voice_path))
                            logger.info(f"Loaded voice embeddings: {voice_prompt}")

                            # Apply system prompts (voice conditioning)
                            if hasattr(self.lm_gen, 'step_system_prompts'):
                                self.lm_gen.step_system_prompts(self.mimi)
                                self.mimi.reset_streaming()  # Reset after voice prompt encoding
                                logger.info("Applied voice conditioning via step_system_prompts")
                                voice_applied = True
                        else:
                            logger.warning("Voice prompts not supported (need PersonaPlex moshi)")
                    except Exception as e:
                        logger.error(f"Failed to load voice prompt: {e}")
                else:
                    logger.warning(f"Voice file not found: {voice_path}")

            if not voice_applied:
                logger.info("Using default voice (no voice conditioning)")

            # Send handshake
            await ws.send_bytes(bytes([MSG_HANDSHAKE]))
            logger.info("Sent handshake, starting audio processing")

            await self.recv_loop(ws, opus_reader, opus_writer)

        logger.info("Done with connection")
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
    lm_gen = LMGen(lm, device=device)

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
