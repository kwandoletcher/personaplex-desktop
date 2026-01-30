# PersonaPlex Desktop

A native desktop application for NVIDIA's PersonaPlex voice AI, running locally on your RTX 5070 Ti.

## Features

- ✅ **18 Built-in Voices**: 8 Natural voices (4 male, 4 female) + 10 Variety voices (5 male, 5 female)
- ✅ **Custom Personas**: Create unique personalities with text prompts
- ✅ **Real-time Conversations**: Full-duplex voice dialogue with interruption support
- ✅ **Conversation History**: Save and review past conversations (optional)
- ✅ **Offline Operation**: Complete privacy - runs entirely on your local machine
- ✅ **Custom Voice Conditioning**: Upload your own voice samples (experimental)

## System Requirements

- **GPU**: NVIDIA GeForce RTX 5070 Ti (16GB VRAM)
- **RAM**: 64GB DDR5
- **Storage**: 20GB+ free space for models
- **OS**: Windows 11 Pro with WSL2
- **Network**: Internet connection for initial model download only

## Installation & Setup

### Step 1: WSL2 Setup (One-time)

The application requires WSL2 Ubuntu 24.04. If not already installed:

```powershell
# In PowerShell as Administrator
wsl --install Ubuntu-24.04
```

### Step 2: Start the Python Backend

The Python server runs inside WSL2 and handles the AI model:

```powershell
# Open PowerShell and run:
wsl

# Navigate to the project
cd /mnt/c/Users/User/Documents/moonshot/personaplex-desktop/personaplex_server

# Activate virtual environment
source venv/bin/activate

# Set your HuggingFace token
export HF_TOKEN='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'

# Start the server
python main.py
```

You should see:
```
✓ Server running on ws://0.0.0.0:8998
```

**Note**: First startup will download ~15GB of model files. This takes 10-20 minutes depending on your internet speed.

### Step 3: Build & Run the Desktop App

The desktop app is built with Tauri (Rust + React):

```powershell
# In a NEW PowerShell window (keep the server running in WSL)
cd C:\Users\User\Documents\moonshot\personaplex-desktop

# Install Node.js dependencies
npm install

# Build and run the app
npm run tauri-dev
```

The app will open in a new window.

## Usage Guide

### Starting a Conversation

1. **Select Voice**: Choose from 18 available voice presets
   - Natural voices sound more conversational
   - Variety voices offer more diversity

2. **Define Persona**: Either:
   - Click a preset (Helpful Assistant, Casual, Customer Service, Creative Writer)
   - Write your own custom persona description

3. **Start Talking**: Click the microphone button and speak naturally
   - The AI will respond in real-time
   - You can interrupt at any time

4. **Save Conversation**: When finished, you'll be asked if you want to save it to history

### Tips for Best Results

**Persona Prompts**: Be specific about the persona's:
- Role (teacher, customer service, friend, etc.)
- Personality traits (friendly, professional, humorous)
- Background knowledge (expert in certain topics)
- Speaking style (formal, casual, technical)

**Example prompts**:
```
"You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."

"You work for CitySan Services which is a waste management company. Your name is Ayelen. Be helpful and professional."

"You enjoy having a good conversation. Have a casual discussion about eating at home versus dining out."
```

### Voice Conditioning (Custom Voices)

To use your own voice:
1. Record a 5-10 second sample of your voice (24kHz, mono, WAV format)
2. Place it in `personaplex_server/models/voices/custom_yourname.pt`
3. Restart the server
4. Your custom voice will appear in the voice selection

**Note**: Voice conditioning requires technical setup (extracting embeddings). This feature is experimental.

## Troubleshooting

### "Server not connected" error

1. Make sure WSL2 is running: `wsl --list --verbose`
2. Ensure the Python server is running in WSL2
3. Check firewall settings - port 8998 must be open

### CUDA out of memory

If you see OOM errors:
1. Close other GPU-intensive applications
2. The model automatically uses CPU offloading to fit in 16GB VRAM
3. Consider closing browser tabs with heavy GPU usage

### WSL2 audio not working

If audio capture doesn't work:
1. Install PulseAudio in WSL2: `sudo apt install pulseaudio`
2. Configure Windows to share audio with WSL2
3. Check microphone permissions in Windows Privacy settings

### Model loading fails

If the model won't load:
1. Verify HF_TOKEN is set correctly
2. Check you have 20GB+ free disk space
3. Try deleting `personaplex_server/models/` and re-downloading

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Desktop App    │────▶│  WebSocket API   │────▶│  PersonaPlex    │
│  (Tauri/React)  │     │  (Port 8998)     │     │  Model (WSL2)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  RTX 5070 Ti    │
                                                │  14GB VRAM used │
                                                └─────────────────┘
```

## Development

### Project Structure

```
personaplex-desktop/
├── personaplex_server/      # Python backend (WSL2)
│   ├── main.py              # WebSocket server
│   ├── models/              # AI model files
│   └── voices/              # Voice embeddings
├── src/                     # React frontend
│   ├── App.jsx              # Main application
│   └── main.jsx             # Entry point
├── src-tauri/               # Rust/Tauri backend
│   └── src/main.rs          # Desktop shell
└── conversations/           # Saved conversation history
```

### Adding New Features

**To add a new voice preset**:
1. Add voice file to `personaplex_server/models/voices/`
2. Update `VOICE_PRESETS` in `src/App.jsx`

**To modify the AI behavior**:
1. Edit `personaplex_server/main.py`
2. The server handles all model inference

**To change the UI**:
1. Edit React components in `src/`
2. Run `npm run tauri-dev` to see changes

## Performance

- **Model Size**: 7B parameters (~15GB on disk)
- **VRAM Usage**: ~14GB with CPU offloading
- **Latency**: ~200-400ms response time
- **Sample Rate**: 24kHz audio I/O

## License

This project uses:
- PersonaPlex model: [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
- Moshi architecture: MIT License
- Desktop application: MIT License

## Support

For issues with:
- **Model behavior**: See [NVIDIA PersonaPlex GitHub](https://github.com/NVIDIA/personaplex)
- **This application**: Open an issue in this repository

## Acknowledgments

- NVIDIA for creating PersonaPlex
- Kyutai for the Moshi architecture
- PyTorch team for CUDA 13.0 support
