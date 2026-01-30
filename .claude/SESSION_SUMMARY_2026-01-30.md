# PersonaPlex Desktop - Session Summary (2026-01-30)

## Status: Working but with remaining bugs

The system is functional - audio streams in real-time and text appears in the transcript. User reports "it works decent but there are still some bugs."

---

## What Was Fixed Today

### 1. Audio Performance (MAJOR FIX)
**Problem:** Audio was "extremely slow and skippy" - unusable latency and choppy playback.

**Root Cause:** The original server architecture used a single blocking loop that couldn't receive audio while processing. This caused audio to queue up and play back delayed/choppy.

**Solution:** Rewrote `personaplex_server/main.py` with THREE concurrent async loops:
- `recv_loop`: Continuously receives WebSocket messages, buffers audio in `opus_reader`
- `process_loop`: Processes audio through the model, generates responses
- `send_loop`: Continuously sends encoded audio back to client from `opus_writer`

This allows receiving, processing, and sending to happen simultaneously.

**Key Code Pattern:**
```python
tasks = [
    asyncio.create_task(recv_loop(opus_reader)),
    asyncio.create_task(process_loop(opus_reader, opus_writer)),
    asyncio.create_task(send_loop(opus_writer)),
]
done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
```

### 2. Transcript Disappearing Text (IMPROVED)
**Problem:** Text would appear while AI was speaking, then vanish completely instead of staying in the transcript.

**Root Cause:** React state synchronization issues - using `useState` for text accumulation caused stale closures.

**Solution (in `src/App.jsx`):**
- Use `useRef` (`currentTextRef`) for synchronous text accumulation
- Added `textCommitTimeoutRef` - commits text after 2s pause (increased from 1.5s)
- Added `speakingTimeoutRef` - properly resets `isSpeaking` flag
- Added `lastTextTimeRef` - tracks when text was last received
- Fixed cleanup to commit remaining text before unmounting
- Added console logging: `[Text] Received/Committing` for debugging

### 3. Model Loading Issues
**Problem:** Various errors during model initialization on Lambda Labs A100.

**Fixes Applied:**
- Added `cpu_offload=True` to `loaders.get_moshi_lm()` to fix meta tensor errors
- Added `text_prompt_tokens=[]` to `LMGen()` constructor to fix NoneType iteration
- Added `device=device` parameter to `LMGen()` constructor

### 4. sphn API Compatibility
**Problem:** sphn 0.1.x has different API than 0.2.x docs suggested.

**Fix:** Use correct method names:
- `opus_reader.append_bytes(payload)` then `opus_reader.read_pcm()`
- `opus_writer.append_pcm(pcm)` then `opus_writer.read_bytes()`

---

## Current Architecture

```
Desktop App (Tauri/React)
    |
    | WebSocket (ws://150.136.94.234:8080/api/chat)
    |
    v
PersonaPlex Server (Lambda Labs A100 40GB)
    |
    +-- recv_loop: Receives Opus audio, buffers in opus_reader
    +-- process_loop: Mimi encode -> LMGen step -> Mimi decode
    +-- send_loop: Reads from opus_writer, sends Opus audio
```

**Binary Protocol:**
- 0x00 = handshake
- 0x01 = audio (Opus bytes)
- 0x02 = text (UTF-8)
- 0x03 = control
- 0x04 = metadata
- 0x05 = error

---

## Known Remaining Bugs

User said "there are still some bugs" but didn't specify. Likely issues:

1. **Voice selection may not work** - Need to verify PersonaPlex's `load_voice_prompt_embeddings()` is being called correctly
2. **Transcript timing** - Text commit timeout (2s) may be too aggressive or not aggressive enough
3. **isSpeaking indicator** - May not reset properly in all cases
4. **Interruption handling** - Full-duplex interruption may not work smoothly

---

## Files Modified

### `personaplex_server/main.py`
- Complete rewrite of `handle_chat()` with concurrent loops
- Added logging for audio packets and text tokens
- Fixed model loading parameters

### `src/App.jsx`
- Added refs: `currentTextRef`, `textCommitTimeoutRef`, `lastTextTimeRef`, `speakingTimeoutRef`
- Improved text handling with timeout-based commit
- Added console logging for debugging
- Fixed cleanup handlers

---

## Server Deployment

**Lambda Labs A100 (40GB):**
- IP: 150.136.94.234
- Port: 8080
- VRAM Usage: ~19GB (model + KV cache)
- Environment: `~/moshi-env`
- Server path: `~/personaplex-models/personaplex-desktop/personaplex_server`

**Start server:**
```bash
ssh ubuntu@150.136.94.234
cd ~/personaplex-models/personaplex-desktop/personaplex_server
source ~/moshi-env/bin/activate
python main.py --host 0.0.0.0 --port 8080
```

**Check logs:**
```bash
tail -f /tmp/server.log
```

---

## Dependencies

**Server (Python):**
- moshi (from PersonaPlex repo, NOT PyPI)
- aiohttp, sphn, safetensors, sentencepiece, accelerate

**Client (Node):**
- opus-recorder
- lucide-react
- Tauri

---

## Next Steps

1. Test voice selection with different presets (NATF0-3, NATM0-3, VARF0-4, VARM0-4)
2. Fine-tune text commit timeout based on user feedback
3. Add user speech display (Wispr Flow integration)
4. Test full-duplex interruption
5. Consider UI improvements based on feedback
