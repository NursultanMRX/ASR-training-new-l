# 🚀 GOOGLE COLAB - COPY & PASTE THESE CELLS
# ============================================
# Just copy each section below into a new Colab cell and run!

# ============================================
# CELL 1: Install Everything
# ============================================
print("📦 Installing packages (2-3 minutes)...")
!pip install -q transformers datasets accelerate torchaudio torchcodec librosa jiwer evaluate psutil soundfile
print("✅ Installation complete!")

# ============================================
# CELL 2: Clone Repository
# ============================================
!git clone https://github.com/NursultanMRX/ASR-training-new-l.git
%cd ASR-training-new-l
print("✅ Code downloaded!")

# ============================================
# CELL 3: Check GPU
# ============================================
import sys
import torch
import psutil

print("System Information:")
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("\n✅ Perfect! GPU is ready!")
else:
    print("\n⚠️ No GPU! Go to Runtime → Change runtime type → Select GPU")

# ============================================
# CELL 4: Login to HuggingFace
# ============================================
from huggingface_hub import login
print("🔑 Paste your HuggingFace token:")
print("Get it from: https://huggingface.co/settings/tokens\n")
login()
print("✅ Logged in!")

# ============================================
# CELL 5: Activate Keep-Alive (Optional)
# ============================================
import os
import sys

# Add src to path
src_path = os.path.abspath('src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Try to activate keep-alive
try:
    from colab_keeper import activate_colab_keepalive
    keeper = activate_colab_keepalive()
    print("✅ Keep-alive active!")
except:
    print("⚠️ Keep-alive not available (OK - just keep browser open)")

# ============================================
# CELL 6: START TRAINING! 🚀
# ============================================
import os
import sys

# Setup paths
src_path = os.path.abspath('src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import training function
from optimized_training import train_asr_model

# YOUR SETTINGS (CHANGE THESE!)
YOUR_HF_USERNAME = "nickoo004"  # ← Change to YOUR username!
DATASET = "nickoo004/karakalpak-speech-60h-production-v2"
MODEL = "facebook/wav2vec2-xls-r-1b"
OUTPUT_NAME = "wav2vec2-xls-r-1b-karakalpak-colab"

print("🚀 Starting Training...")
print(f"Dataset: {DATASET}")
print(f"Model: {MODEL}")
print(f"Output: {OUTPUT_NAME}\n")

# Run training (this will take hours!)
train_asr_model(
    dataset_repo=DATASET,
    base_model=MODEL,
    output_name=OUTPUT_NAME,
    hf_username=YOUR_HF_USERNAME,
    num_epochs=20,
    target_batch_size=32,
    learning_rate=3e-4,
    use_deepspeed=False,  # Set True for 2x memory savings
    push_to_hub=True,
    skip_health_check=False  # Set True to skip checks
)

print(f"\n🎉 Complete! Model at: https://huggingface.co/{YOUR_HF_USERNAME}/{OUTPUT_NAME}")

# ============================================
# OPTIONAL: Monitor with TensorBoard
# ============================================
%load_ext tensorboard
%tensorboard --logdir outputs/
