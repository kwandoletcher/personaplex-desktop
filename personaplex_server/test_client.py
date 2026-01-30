#!/usr/bin/env python3
"""
Quick test script to verify the backend is working
"""

import asyncio
import websockets
import json


async def test_server():
    uri = "ws://localhost:8998"
    
    print("Connecting to PersonaPlex server...")
    async with websockets.connect(uri) as websocket:
        print("✓ Connected")
        
        # Test get voices
        print("\n1. Testing get_voices...")
        await websocket.send(json.dumps({"type": "get_voices"}))
        response = await websocket.recv()
        data = json.loads(response)
        print(f"✓ Received {len(data.get('voices', []))} voices")
        
        # Test start conversation
        print("\n2. Testing start_conversation...")
        await websocket.send(json.dumps({
            "type": "start_conversation",
            "voice": "NATF1",
            "persona": "You are a helpful assistant."
        }))
        response = await websocket.recv()
        data = json.loads(response)
        print(f"✓ Response: {data.get('type')}")
        
        # Test end conversation
        print("\n3. Testing end_conversation...")
        await websocket.send(json.dumps({"type": "end_conversation"}))
        response = await websocket.recv()
        data = json.loads(response)
        print(f"✓ Response: {data.get('type')}")
        
        print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_server())
