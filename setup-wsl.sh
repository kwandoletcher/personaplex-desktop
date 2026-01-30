#!/bin/bash
# Setup script for PersonaPlex Desktop
# Run this in WSL2 Ubuntu

set -e

echo "======================================"
echo "PersonaPlex Desktop Setup"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in WSL
if ! grep -q Microsoft /proc/version && ! grep -q WSL /proc/version; then
    echo -e "${RED}Error: This script must be run in WSL2${NC}"
    exit 1
fi

# Navigate to project directory
cd /mnt/c/Users/User/Documents/moonshot/personaplex-desktop/personaplex_server

echo -e "${GREEN}Step 1/5: Setting up Python environment...${NC}"
if [ ! -d "venv" ]; then
    python3.12 -m venv venv
fi
source venv/bin/activate

echo -e "${GREEN}Step 2/5: Installing PyTorch with CUDA 13.0 (for RTX 5070 Ti)...${NC}"
pip install --upgrade pip
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 torchaudio==2.10.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130 \
    --force-reinstall --no-cache-dir

echo -e "${GREEN}Step 3/5: Installing other dependencies...${NC}"
pip install websockets accelerate

echo -e "${GREEN}Step 4/5: Installing PersonaPlex (moshi)...${NC}"
if [ ! -d "personaplex" ]; then
    git clone https://github.com/NVIDIA/personaplex.git
fi
cd personaplex/moshi
# Modify torch requirement to allow 2.10
sed -i 's/torch >= 2.2.0, < 2.5/torch >= 2.2.0, < 2.11/' pyproject.toml
pip install -e . --no-deps
cd ../..

echo -e "${GREEN}Step 5/5: Verifying installation...${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "To start the server, run:"
echo "  cd /mnt/c/Users/User/Documents/moonshot/personaplex-desktop/personaplex_server"
echo "  source venv/bin/activate"
echo "  export HF_TOKEN='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'"
echo "  python main.py"
echo ""
echo "Then in a new PowerShell window, run:"
echo "  cd C:\Users\User\Documents\moonshot\personaplex-desktop"
echo "  npm run tauri-dev"
