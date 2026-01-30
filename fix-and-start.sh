#!/bin/bash
# Quick fix script for PersonaPlex setup
# Run this in WSL2

echo "======================================"
echo "Quick Fix for PersonaPlex Server"
echo "======================================"
echo ""

# Kill any existing servers
echo "Step 1: Stopping any existing servers..."
pkill -9 -f "python main.py" 2>/dev/null
pkill -9 -f "moshi" 2>/dev/null
sleep 2
echo "✓ Servers stopped"

# Navigate to project
cd /mnt/c/Users/User/Documents/moonshot/personaplex-desktop/personaplex_server
source venv/bin/activate

# Fix broken packages
echo ""
echo "Step 2: Fixing package installation..."
rm -rf venv/lib/python3.12/site-packages/~* 2>/dev/null

# Install minimal working versions
echo "Installing torch 2.4.1..."
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 -q

echo "Installing accelerate..."
pip install accelerate -q

echo "✓ Packages installed"

# Test
echo ""
echo "Step 3: Testing installation..."
python -c "import torch; import accelerate; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA: {torch.version.cuda}'); print(f'✓ Accelerate: OK')"

echo ""
echo "======================================"
echo "Setup complete! Starting server..."
echo "======================================"
echo ""

# Start server
export HF_TOKEN='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'
python main.py
