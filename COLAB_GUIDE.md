# 🚀 SUPER EASY GOOGLE COLAB GUIDE

## Quick Start (Copy-Paste Method)

**Before starting:** Runtime → Change runtime type → **GPU** → Save

---

## 📋 Copy Each Cell Below

### Cell 1: Install
```python
print("📦 Installing packages...")
!pip install -q transformers datasets accelerate torchaudio torchcodec librosa jiwer evaluate psutil soundfile
print("✅ Done!")
```

### Cell 2: Clone Code
```python
!git clone https://github.com/NursultanMRX/ASR-training-new-l.git
%cd ASR-training-new-l
print("✅ Code ready!")
```

### Cell 3: Check GPU
```python
import sys, torch, psutil

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("✅ GPU Ready!")
else:
    print("⚠️ No GPU! Go to Runtime → Change runtime → GPU")
```

### Cell 4: Login
```python
from huggingface_hub import login
print("Paste your token from: https://huggingface.co/settings/tokens")
login()
print("✅ Logged in!")
```

### Cell 5: Keep-Alive (Optional)
```python
import os, sys
src_path = os.path.abspath('src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from colab_keeper import activate_colab_keepalive
    keeper = activate_colab_keepalive()
    print("✅ Keep-alive active!")
except:
    print("⚠️ Keep-alive not available (OK)")
```

### Cell 6: START TRAINING! 🚀
```python
import os, sys

# Setup path
src_path = os.path.abspath('src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from optimized_training import train_asr_model

# CHANGE THIS TO YOUR USERNAME!
YOUR_USERNAME = "nickoo004"  # ← Put YOUR HuggingFace username here!

print("🚀 Starting training...")

train_asr_model(
    dataset_repo="nickoo004/karakalpak-speech-60h-production-v2",
    base_model="facebook/wav2vec2-xls-r-1b",
    output_name="wav2vec2-karakalpak-colab",
    hf_username=YOUR_USERNAME,
    num_epochs=20,
    push_to_hub=True
)

print(f"🎉 Done! Model at: https://huggingface.co/{YOUR_USERNAME}/wav2vec2-karakalpak-colab")
```

---

## ⏱️ Expected Time

| GPU | Time |
|-----|------|
| T4 (Free) | ~24-36 hours |
| L4 (Pro) | ~12-18 hours |
| A100 (Pro+) | ~6-8 hours |

---

## ❓ If Something Goes Wrong

### Error: ModuleNotFoundError
**Fix:** Make sure you ran Cell 2 (clone code) first!

### Error: No GPU
**Fix:** Runtime → Change runtime type → GPU → Save

### Error: Out of Memory
**Fix:** In Cell 6, add `use_deepspeed=True` parameter

### Training Interrupted?
**Fix:** Just re-run Cell 6 - it will auto-resume!

---

## 💡 Pro Tips

1. **Keep browser open** - Don't close the tab while training
2. **Check console** - Press F12 to see keep-alive pings
3. **Monitor progress** - Add a cell with:
   ```python
   %load_ext tensorboard
   %tensorboard --logdir outputs/
   ```
4. **Save often** - Model auto-saves every N steps

---

**That's it! Training on Colab is now super easy! 🎉**

For more help: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
