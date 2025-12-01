#!/bin/bash

##############################################################################
# ASR Training System - Setup Script for Cloud Platforms
# Compatible with: RunPod, Google Colab, Lambda Labs, Vast.ai
##############################################################################

set -e  # Exit on error

echo "=========================================="
echo "ASR Training System - Setup"
echo "=========================================="

# Detect platform
if [ -d "/content" ]; then
    PLATFORM="colab"
    echo "Platform: Google Colab"
elif [ -d "/workspace" ]; then
    PLATFORM="runpod"
    echo "Platform: RunPod"
else
    PLATFORM="unknown"
    echo "Platform: Generic Linux"
fi

# Check Python version
echo ""
echo "Checking Python version..."
python --version

# Check CUDA availability
echo ""
echo "Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Training will be CPU-only (very slow)."
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (with CUDA support)
echo ""
echo "Installing PyTorch..."
if [ "$PLATFORM" = "colab" ]; then
    # Colab already has PyTorch, just ensure CUDA version
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # For other platforms, install with CUDA 11.8 support
    pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Install requirements
echo ""
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Verify installations
echo ""
echo "=========================================="
echo "Verifying Installations..."
echo "=========================================="

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"
python -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Set your HuggingFace token:"
echo "   export HF_TOKEN='your_token_here'"
echo ""
echo "2. Or login interactively:"
echo "   python -c 'from huggingface_hub import login; login()'"
echo ""
echo "3. Run training:"
echo "   cd src"
echo "   python optimized_training.py"
echo ""
echo "See RUN.md for more options!"
echo "=========================================="
