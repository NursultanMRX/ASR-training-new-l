# 🛡️ BULLETPROOF ASR TRAINING - TROUBLESHOOTING GUIDE

## Common Issues & Auto-Fixes

### 🔄 Google Colab Auto-Reload / Disconnection

**Problem:** Colab disconnects or reloads automatically during training.

**Auto-Fix Applied:**  
✅ JavaScript keep-alive widget injected  
✅ Connection monitoring active  
✅ Auto-checkpoint resume enabled

**Manual Steps (if needed):**
1. Keep your browser tab open (don't close it)
2. Open browser console (F12) to see keep-alive pings
3. If disconnected, simply re-run the training cell - it will auto-resume!

---

### 💥 Out of Memory (OOM) Errors

**Problem:** `RuntimeError: CUDA out of memory`

**Auto-Fix Applied:**  
✅ Automatic batch size reduction  
✅ CUDA cache clearing  
✅ Retry with smaller batch (up to 3 attempts)

**If still failing:**
```python
# Reduce safety margin
python src/cli.py train --safety_margin=0.70
```

---

### 🌐 Network Timeout / Connection Errors

**Problem:** Dataset download fails or HuggingFace API timeouts

**Auto-Fix Applied:**  
✅ Automatic retry (up to 3 attempts)  
✅ 10-second backoff between retries  
✅ Network connectivity checks

**Manual Check:**
```bash
# Test internet
ping google.com

# Test HuggingFace
curl https://huggingface.co
```

---

### 💾 Disk Full Errors

**Problem:** `IOError: No space left on device`

**Auto-Detection:**  
✅ Pre-flight disk space check  
✅ Recommended cleanup shown

**Manual Fix:**
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface

# Check space
df -h

# Cleanup old checkpoints
rm -rf outputs/checkpoint-old*
```

---

### 🔐 HuggingFace Authentication Issues

**Problem:** Cannot access dataset or push model

**Auto-Detection:**  
✅ Pre-flight token check

**Manual Fix:**
```python
from huggingface_hub import login
login()  # Paste your token
```

Or via CLI:
```bash
huggingface-cli login
```

---

### ⚡ GPU Not Detected

**Problem:** Training runs on CPU (very slow)

**Auto-Detection:**  
✅ Pre-flight GPU check  
✅ Warning shown if CPU-only

**Manual Fix (Colab):**
1. Runtime → Change runtime type
2. Select "GPU" (T4, L4, or A100)
3. Click "Save"
4. Re-run training

---

### 🔄 Training Interrupted (Ctrl+C)

**Auto-Recovery:**  
✅ Graceful shutdown triggered  
✅ Emergency checkpoint saved  
✅ Can resume from checkpoint

**To Resume:**
```bash
# Auto-detects latest checkpoint
python src/cli.py train
```

---

### 📦 Missing Dependencies

**Problem:** `ModuleNotFoundError`

**Auto-Detection:**  
✅ Pre-flight dependency check  
✅ Shows missing packages

**Manual Fix:**
```bash
pip install -r requirements.txt
```

---

## 🚨 Error Codes

| Error | Meaning | Auto-Fix |
|-------|---------|----------|
| OOM | Out of memory | ✅ Yes - reduces batch size |
| CUDA Error | GPU issue | ✅ Yes - resets CUDA context |
| ConnectionError | Network down | ✅ Yes - retries with backoff |
| IOError (disk) | Disk full | ⚠️ Partial - shows cleanup tips |
| Import Error | Missing package | ❌ No - install manually |

---

## 📋 Pre-Flight Checklist

Before training, the system automatically checks:

- [ ] ✅ GPU available
- [ ] ✅ Sufficient RAM (8GB+)
- [ ] ✅ Sufficient disk space (50GB+)
- [ ] ✅ All dependencies installed
- [ ] ✅ HuggingFace token valid
- [ ] ✅ Internet connection
- [ ] ✅ Write permissions

**If any fail:** Fix the issue and try again!

---

## 🔧 Recovery Features

### Automatic
- ✅ Checkpoint resume after disconnection
- ✅ Retry on transient errors (3x)
- ✅ Memory cleanup on OOM
- ✅ Graceful shutdown on Ctrl+C

### Manual
- Run health check: `python src/health_check.py`
- View keep-alive status: Check browser console (F12)
- Find checkpoints: `ls outputs/checkpoint-*`

---

## 💡 Pro Tips

1. **For long training on Colab:**
   - Use Colab Pro for longer sessions
   - Keep browser tab open
   - Check console for keep-alive pings

2. **For unstable networks:**
   - System auto-retries 3 times
   - Use `--resume_from_checkpoint` if manual resume needed

3. **For memory issues:**
   - Start with `--safety_margin=0.70` (conservative)
   - Use smaller model if needed
   - Enable `--use_deepspeed=True` for 2x memory savings

4. **For debugging:**
   - Check `training_config.json` for actual settings used
   - View TensorBoard: `tensorboard --logdir outputs/`
   - Check logs: `tail -f training.log`

---

## 📞 Still Having Issues?

1. **Run full health check:**
   ```bash
   python src/health_check.py
   ```

2. **Check this guide:** Go through each section above

3. **Enable verbose logging:**
   ```bash
   export TRANSFORMERS_VERBOSITY=debug
   python src/cli.py train
   ```

---

**Remember:** The system is designed to handle 99% of issues automatically. If training fails, just re-run - it will resume from the last checkpoint! 🚀
