# PersonaPlex Desktop

Real-time voice AI with 18 customizable voices, powered by NVIDIA PersonaPlex 7B.

## Features

- **18 Voice Presets**: Natural (4 male, 4 female) + Variety (5 male, 5 female)
- **Custom Persona Prompts**: Create unique AI personalities
- **Full-Duplex Conversation**: Interruption support for natural dialogue
- **Conversation History**: Save and review past conversations
- **Cloud GPU Inference**: Lambda Labs A100 for real-time performance
- **Microphone Settings**: Sensitivity control, noise suppression, echo cancellation

## Quick Start

### 1. Start Cloud Server

```bash
ssh ubuntu@150.136.94.234
cd ~/personaplex-models/personaplex-desktop/personaplex_server
source ~/moshi-env/bin/activate
python main.py --host 0.0.0.0 --port 8080
```

### 2. Start Desktop App

```bash
cd C:\Users\User\Documents\moonshot\personaplex-desktop
npm run tauri dev
```

## Usage

1. **Select Voice**: Choose from 18 voice presets using the dropdown
2. **Set Persona**: Pick a preset or write a custom persona prompt
3. **Adjust Mic**: Fine-tune sensitivity and toggle noise suppression/echo cancellation
4. **Start**: Click "Start Conversation" and speak naturally
5. **Interrupt**: Talk over the AI at any time - it will stop and respond

### Persona Examples

```
"You are a wise and friendly teacher. Answer questions clearly."

"You work for customer service. Be polite and professional."

"You enjoy casual conversation. Be friendly and engaging."
```

## Architecture

```
Desktop App (Tauri) --> WebSocket --> PersonaPlex Server (Lambda A100)
                                              |
                                              v
                                      40GB VRAM (A100)
                                      17GB Model + KV Cache
```

## Server Management

**Lambda Labs bills hourly** - terminate instances when not in use.

Model files persist on Lambda filesystem (no re-download needed).

### Server Setup (First Time)

**IMPORTANT:** Must install PersonaPlex's modified moshi library (NOT PyPI moshi).

```bash
# SSH to server
ssh ubuntu@150.136.94.234

# Clone PersonaPlex repo (includes modified moshi library)
git clone https://github.com/NVIDIA/personaplex.git ~/personaplex

# Create/activate environment
python -m venv ~/moshi-env
source ~/moshi-env/bin/activate

# Install PersonaPlex moshi (enables voice embeddings)
cd ~/personaplex
pip install moshi/.

# Install additional dependencies
pip install aiohttp sphn safetensors sentencepiece accelerate

# Copy server files
mkdir -p ~/personaplex-models/personaplex-desktop/personaplex_server
# (Copy main.py and models/ from this repo)

# Download model files to models/ directory
# - tokenizer-e351c8d8-checkpoint125.safetensors (Mimi codec)
# - model.safetensors (PersonaPlex 7B)
# - tokenizer_spm_32k_3.model (text tokenizer)
# - voices/*.pt (18 voice embeddings)
```

### Server Dependencies

```bash
pip install aiohttp sphn safetensors sentencepiece accelerate
```

**Note:** The standard `pip install moshi` from PyPI does NOT include voice conditioning.
You must install from PersonaPlex's repo for voice selection to work.

## Development

### Project Structure

```
personaplex-desktop/
├── src/
│   ├── App.jsx              # Main application (compact left panel)
│   └── main.jsx             # Entry point
├── src-tauri/
│   └── src/main.rs          # Tauri backend
├── personaplex_server/
│   ├── main.py              # WebSocket server
│   └── models/voices/       # Voice embeddings
└── public/assets/           # Opus encoder/decoder workers
```

### Key Components

- **VoiceDropdown**: Compact voice selection
- **PersonaInput**: Preset chips + custom textarea
- **MicSettings**: Sensitivity slider + NS/EC toggles
- **TranscriptList**: Scrolling conversation display
- **AudioLevelMeter**: Real-time mic level visualization

## Roadmap

### v1.1 - Current Sprint
- [x] Cloud GPU deployment (Lambda Labs A100)
- [x] UI redesign (compact left panel)
- [x] Transcript bug fix (timeout-based commit for text without punctuation)
- [x] Microphone settings UI
- [x] Voice selection UI (18 presets)
- [ ] Voice embeddings active (requires PersonaPlex moshi on server)
- [ ] Wispr Flow integration for user speech

### v1.2 - Enhanced Audio
- [ ] Noise gate option
- [ ] Voice activity detection tuning
- [ ] Audio device selection

### v2.0 - Virtual Audio Device
- [ ] VB-Cable integration
- [ ] Route AI voice to Discord/Zoom/etc
- [ ] Voice changer mode
- [ ] Real-time voice modification

### v3.0 - B2B Applications
- [ ] Headless API mode
- [ ] CRM webhooks (Salesforce, HubSpot)
- [ ] Call analytics & transcripts
- [ ] Multi-agent conversations
- [ ] Auto-dialer integration

## Troubleshooting

### "Server not connected"
1. Verify Lambda Labs instance is running
2. SSH in and check if `python main.py` is running
3. Check firewall allows port 8080

### Audio issues
1. Check microphone permissions in Windows
2. Adjust sensitivity slider if input is too quiet/loud
3. Toggle noise suppression off if voice is being filtered

### WebSocket connection drops
- Lambda may timeout after inactivity
- Restart the server: `python main.py --host 0.0.0.0 --port 8080`

### Voice selection not working (all voices sound the same)
This happens when using PyPI moshi instead of PersonaPlex moshi:
1. Check server logs for "Voice prompts not supported (need PersonaPlex moshi)"
2. Install PersonaPlex moshi: `cd ~/personaplex && pip install moshi/.`
3. Restart the server

## Performance

- **Model Size**: 7B parameters (~17GB VRAM)
- **GPU**: NVIDIA A100 (40GB)
- **Latency**: ~200-400ms response time
- **Sample Rate**: 24kHz audio I/O

## License

- PersonaPlex model: [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
- Moshi architecture: MIT License
- Desktop application: MIT License

## Acknowledgments

- NVIDIA for creating PersonaPlex
- Kyutai for the Moshi architecture
- Lambda Labs for cloud GPU infrastructure
