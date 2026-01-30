#!/usr/bin/env python3
"""
PersonaPlex Desktop - Python Backend Server
Handles model inference, WebSocket communication, and audio streaming
"""

import asyncio
import json
import base64
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import numpy as np
import torch
import websockets
from websockets.server import WebSocketServerProtocol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 24000
CHUNK_DURATION = 0.1  # 100ms chunks
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
DB_PATH = Path(__file__).parent / "conversations.db"
MODELS_DIR = Path(__file__).parent / "models"


@dataclass
class Conversation:
    """Represents a conversation session"""
    id: int
    created_at: str
    voice_preset: str
    persona_prompt: str
    transcript: list
    audio_path: Optional[str]
    duration_seconds: float


class StorageManager:
    """Manages conversation storage in SQLite"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
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
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def save_conversation(self, voice_preset: str, persona_prompt: str, 
                         transcript: list, audio_path: Optional[str] = None,
                         duration_seconds: float = 0.0) -> int:
        """Save a conversation to database"""
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
    
    def get_conversation(self, conv_id: int) -> Optional[Conversation]:
        """Retrieve a conversation by ID"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conv_id,)
            ).fetchone()
            
            if row:
                return Conversation(
                    id=row[0],
                    created_at=row[1],
                    voice_preset=row[2],
                    persona_prompt=row[3],
                    transcript=json.loads(row[4]) if row[4] else [],
                    audio_path=row[5],
                    duration_seconds=row[6] if row[6] else 0.0
                )
            return None
    
    def list_conversations(self, limit: int = 50, offset: int = 0) -> list:
        """List conversations with pagination"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT id, created_at, voice_preset, persona_prompt, 
                          duration_seconds 
                   FROM conversations 
                   ORDER BY created_at DESC 
                   LIMIT ? OFFSET ?""",
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
    
    def delete_conversation(self, conv_id: int) -> bool:
        """Delete a conversation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conv_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def get_setting(self, key: str, default: str = None) -> Optional[str]:
        """Get a setting value"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?",
                (key,)
            ).fetchone()
            return row[0] if row else default
    
    def set_setting(self, key: str, value: str):
        """Set a setting value"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO settings (key, value, updated_at)
                   VALUES (?, ?, CURRENT_TIMESTAMP)""",
                (key, value)
            )
            conn.commit()


class ModelManager:
    """Manages PersonaPlex model loading and inference"""
    
    def __init__(self):
        self.model = None
        self.mimi = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False
        self.current_voice = None
        self.current_persona = None
        
        logger.info(f"ModelManager initialized (device: {self.device})")
    
    def load_model(self, cpu_offload: bool = True):
        """Load the PersonaPlex model"""
        if self.loaded:
            logger.info("Model already loaded")
            return
        
        try:
            from moshi.models import loaders
            
            logger.info("Loading PersonaPlex model...")
            model_path = MODELS_DIR / "model.safetensors"
            
            self.model = loaders.get_moshi_lm(
                filename=str(model_path),
                device=self.device,
                cpu_offload=cpu_offload
            )
            
            logger.info("Loading Mimi audio codec...")
            self.mimi = loaders.get_mimi(
                str(MODELS_DIR / "tokenizer-e351c8d8-checkpoint125.safetensors"),
                device=self.device
            )
            
            logger.info("Loading tokenizer...")
            self.tokenizer = loaders.get_text_tokenizer(
                str(MODELS_DIR / "tokenizer_spm_32k_3.model")
            )
            
            self.loaded = True
            logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def unload_model(self):
        """Unload model to free memory"""
        self.model = None
        self.mimi = None
        self.tokenizer = None
        self.loaded = False
        torch.cuda.empty_cache()
        logger.info("Model unloaded")
    
    def set_voice(self, voice_file: str):
        """Set the voice prompt"""
        voice_path = MODELS_DIR / "voices" / voice_file
        if not voice_path.exists():
            raise ValueError(f"Voice file not found: {voice_file}")
        self.current_voice = str(voice_path)
        logger.info(f"Voice set to: {voice_file}")
    
    def set_persona(self, persona_text: str):
        """Set the persona prompt"""
        self.current_persona = persona_text
        logger.info(f"Persona set: {persona_text[:50]}...")
    
    def generate_response(self, audio_input: np.ndarray) -> np.ndarray:
        """Generate audio response from input audio"""
        # This is a simplified version - full implementation would use
        # the actual PersonaPlex inference code
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        # Placeholder: return silence
        return np.zeros_like(audio_input)


class AudioPipeline:
    """Handles audio capture and playback"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.input_stream = None
        self.output_stream = None
        self.recording = False
        
    def start_recording(self, callback):
        """Start capturing audio from microphone"""
        import sounddevice as sd
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            callback(indata[:, 0])  # Mono audio
        
        self.input_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=CHUNK_SAMPLES,
            callback=audio_callback
        )
        self.input_stream.start()
        self.recording = True
        logger.info("Audio recording started")
    
    def stop_recording(self):
        """Stop capturing audio"""
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
        self.recording = False
        logger.info("Audio recording stopped")
    
    def play_audio(self, audio_data: np.ndarray):
        """Play audio through speakers"""
        import sounddevice as sd
        sd.play(audio_data, self.sample_rate)
    
    def stop_playback(self):
        """Stop audio playback"""
        import sounddevice as sd
        sd.stop()


class PersonaPlexServer:
    """WebSocket server for PersonaPlex communication"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8998):
        self.host = host
        self.port = port
        self.storage = StorageManager()
        self.model_manager = ModelManager()
        self.audio_pipeline = AudioPipeline()
        self.clients: set = set()
        self.active_conversation = None
        self.conversation_start_time = None
        
    async def start(self):
        """Start the WebSocket server"""
        logger.info(f"Starting server on {self.host}:{self.port}")
        
        # Load model on startup
        try:
            self.model_manager.load_model(cpu_offload=True)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Server will run without model loaded")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"✓ Server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever
    
    async def handle_client(self, websocket):
        """Handle a client connection"""
        self.clients.add(websocket)
        client_id = id(websocket)
        logger.info(f"Client {client_id} connected")
        
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        finally:
            self.clients.discard(websocket)
    
    async def process_message(self, websocket: WebSocketServerProtocol, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "start_conversation":
                await self.handle_start_conversation(websocket, data)
            elif msg_type == "audio_chunk":
                await self.handle_audio_chunk(websocket, data)
            elif msg_type == "end_conversation":
                await self.handle_end_conversation(websocket, data)
            elif msg_type == "get_voices":
                await self.handle_get_voices(websocket)
            elif msg_type == "get_history":
                await self.handle_get_history(websocket, data)
            elif msg_type == "save_conversation":
                await self.handle_save_conversation(websocket, data)
            elif msg_type == "interrupt":
                await self.handle_interrupt(websocket)
            else:
                await self.send_error(websocket, f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError:
            await self.send_error(websocket, "Invalid JSON")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.send_error(websocket, str(e))
    
    async def handle_start_conversation(self, websocket: WebSocketServerProtocol, data: dict):
        """Start a new conversation"""
        voice = data.get("voice", "NATF1")
        persona = data.get("persona", "You are a helpful assistant.")
        
        try:
            self.model_manager.set_voice(f"{voice}.pt")
            self.model_manager.set_persona(persona)
            
            self.active_conversation = {
                "voice": voice,
                "persona": persona,
                "transcript": [],
                "audio_chunks": []
            }
            self.conversation_start_time = datetime.now()
            
            await self.send_message(websocket, {
                "type": "conversation_started",
                "voice": voice,
                "persona": persona
            })
            
            logger.info(f"Conversation started with voice={voice}")
            
        except Exception as e:
            await self.send_error(websocket, f"Failed to start conversation: {e}")
    
    async def handle_audio_chunk(self, websocket: WebSocketServerProtocol, data: dict):
        """Process incoming audio chunk"""
        if not self.active_conversation:
            await self.send_error(websocket, "No active conversation")
            return
        
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(data["data"])
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Store for later
            self.active_conversation["audio_chunks"].append(audio_array)
            
            # Generate response (simplified - would use actual model)
            # For now, just echo back a placeholder
            response_audio = np.zeros_like(audio_array)
            response_b64 = base64.b64encode(response_audio.tobytes()).decode()
            
            await self.send_message(websocket, {
                "type": "audio_response",
                "data": response_b64
            })
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            await self.send_error(websocket, f"Audio processing error: {e}")
    
    async def handle_end_conversation(self, websocket: WebSocketServerProtocol, data: dict):
        """End the current conversation"""
        if not self.active_conversation:
            await self.send_error(websocket, "No active conversation")
            return
        
        duration = 0.0
        if self.conversation_start_time:
            duration = (datetime.now() - self.conversation_start_time).total_seconds()
        
        await self.send_message(websocket, {
            "type": "conversation_ended",
            "duration": duration,
            "message": "Conversation ended. Use 'save_conversation' to save it."
        })
        
        self.active_conversation = None
        self.conversation_start_time = None
    
    async def handle_save_conversation(self, websocket: WebSocketServerProtocol, data: dict):
        """Save the conversation to storage"""
        if not self.active_conversation:
            await self.send_error(websocket, "No conversation to save")
            return
        
        try:
            conv_id = self.storage.save_conversation(
                voice_preset=self.active_conversation["voice"],
                persona_prompt=self.active_conversation["persona"],
                transcript=self.active_conversation["transcript"],
                duration_seconds=(datetime.now() - self.conversation_start_time).total_seconds() if self.conversation_start_time else 0.0
            )
            
            await self.send_message(websocket, {
                "type": "conversation_saved",
                "id": conv_id
            })
            
            logger.info(f"Conversation saved with ID: {conv_id}")
            
        except Exception as e:
            await self.send_error(websocket, f"Failed to save conversation: {e}")
    
    async def handle_get_voices(self, websocket: WebSocketServerProtocol):
        """Return list of available voices"""
        voices_dir = MODELS_DIR / "voices"
        voices = []
        
        if voices_dir.exists():
            for voice_file in sorted(voices_dir.glob("*.pt")):
                voice_name = voice_file.stem
                voice_type = "Natural" if voice_name.startswith("NAT") else "Variety"
                gender = "Female" if "F" in voice_name else "Male"
                voices.append({
                    "id": voice_name,
                    "name": voice_name,
                    "type": voice_type,
                    "gender": gender,
                    "file": voice_file.name
                })
        
        await self.send_message(websocket, {
            "type": "voices_list",
            "voices": voices
        })
    
    async def handle_get_history(self, websocket: WebSocketServerProtocol, data: dict):
        """Return conversation history"""
        limit = data.get("limit", 50)
        offset = data.get("offset", 0)
        
        conversations = self.storage.list_conversations(limit, offset)
        
        await self.send_message(websocket, {
            "type": "history_list",
            "conversations": conversations
        })
    
    async def handle_interrupt(self, websocket: WebSocketServerProtocol):
        """Handle conversation interrupt"""
        await self.send_message(websocket, {
            "type": "interrupted",
            "message": "Generation interrupted"
        })
    
    async def send_message(self, websocket: WebSocketServerProtocol, data: dict):
        """Send message to client"""
        await websocket.send(json.dumps(data))
    
    async def send_error(self, websocket: WebSocketServerProtocol, error: str):
        """Send error message to client"""
        await self.send_message(websocket, {
            "type": "error",
            "message": error
        })


def main():
    """Main entry point"""
    server = PersonaPlexServer(host="0.0.0.0", port=8998)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
