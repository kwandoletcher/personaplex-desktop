#!/usr/bin/env python3
"""
PersonaPlex Voice Test - CLI
Tests microphone input and voice output with PersonaPlex
"""

import asyncio
import numpy as np
import sounddevice as sd
import torch
from pathlib import Path
import sys

# Add moshi to path
sys.path.insert(0, str(Path(__file__).parent / "personaplex" / "moshi"))

from moshi.models import loaders

print("=" * 60)
print("PersonaPlex Voice Test")
print("=" * 60)
print()

# Configuration
SAMPLE_RATE = 24000
CHUNK_DURATION = 0.5  # 500ms chunks
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Load model
print("Loading PersonaPlex model...")
print("(This may take a minute...)")
try:
    model = loaders.get_moshi_lm(
        filename="models/model.safetensors",
        device=DEVICE,
        cpu_offload=True
    )
    print("✓ Model loaded successfully!")
    print()
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# Load Mimi codec
print("Loading Mimi audio codec...")
try:
    mimi = loaders.get_mimi(
        "models/tokenizer-e351c8d8-checkpoint125.safetensors",
        device=DEVICE
    )
    print("✓ Mimi loaded!")
    print()
except Exception as e:
    print(f"✗ Failed to load Mimi: {e}")
    sys.exit(1)

# List audio devices
print("Available audio devices:")
print("-" * 60)
devices = sd.query_devices()
input_devices = []
output_devices = []

for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        input_devices.append((i, device['name']))
        print(f"  Input  {i}: {device['name']}")
    if device['max_output_channels'] > 0:
        output_devices.append((i, device['name']))
        print(f"  Output {i}: {device['name']}")
print("-" * 60)
print()

# Find Blue Snowball
blue_snowball = None
for idx, name in input_devices:
    if 'snowball' in name.lower() or 'blue' in name.lower():
        blue_snowball = idx
        print(f"✓ Found Blue Snowball at index {idx}")
        break

if blue_snowball is None:
    print("⚠ Blue Snowball not found. Using default input device.")
    blue_snowball = sd.default.device[0]

print()
print("=" * 60)
print("Test 1: Microphone Input")
print("=" * 60)
print("Recording 3 seconds of audio from your microphone...")
print("Speak now!")

try:
    # Record audio
    recording = sd.rec(
        int(SAMPLE_RATE * 3), 
        samplerate=SAMPLE_RATE, 
        channels=1, 
        dtype=np.float32,
        device=blue_snowball
    )
    sd.wait()
    print("✓ Recording complete!")
    print(f"  Audio shape: {recording.shape}")
    print(f"  Duration: {len(recording) / SAMPLE_RATE:.2f} seconds")
    print(f"  Max amplitude: {np.max(np.abs(recording)):.4f}")
    print()
except Exception as e:
    print(f"✗ Recording failed: {e}")
    sys.exit(1)

print("=" * 60)
print("Test 2: Audio Processing")
print("=" * 60)
print("Processing audio through Mimi codec...")

try:
    # Convert to tensor
    audio_tensor = torch.from_numpy(recording).squeeze().to(DEVICE)
    
    # Encode with Mimi
    print("  Encoding audio...")
    codes = mimi.encode(audio_tensor.unsqueeze(0).unsqueeze(0))
    print(f"✓ Encoded to {codes.shape} codes")
    
    # Decode back (to test)
    print("  Decoding audio...")
    decoded = mimi.decode(codes)
    print(f"✓ Decoded back to audio shape: {decoded.shape}")
    print()
except Exception as e:
    print(f"✗ Processing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("Test 3: Audio Playback")
print("=" * 60)
print("Playing back the processed audio...")

try:
    audio_out = decoded.squeeze().cpu().numpy()
    sd.play(audio_out, SAMPLE_RATE)
    sd.wait()
    print("✓ Playback complete!")
    print()
except Exception as e:
    print(f"✗ Playback failed: {e}")
    sys.exit(1)

print("=" * 60)
print("Test 4: Model Inference (Text Only)")
print("=" * 60)
print("Testing model with dummy input...")

try:
    # Create dummy input
    dummy_input = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
    
    # Test forward pass
    print("  Running inference...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ Model output shape: {output.shape}")
    print()
except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print()
print("Summary:")
print("  ✓ Model loads successfully")
print("  ✓ Microphone captures audio")
print("  ✓ Audio codec (Mimi) works")
print("  ✓ Audio playback works")
print("  ✓ Model inference works")
print()
print("Ready to proceed with Option B (Desktop App)!")
print()
print("Press Enter to exit...")
input()
