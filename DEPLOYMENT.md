# 🚀 Deployment Guide - Cloud Platforms

Complete guide for deploying on RunPod, Google Colab, Lambda Labs, Vast.ai, and other cloud platforms.

---

## 📋 Prerequisites

1. **GitHub Account** - To host your code
2. **HuggingFace Account** - For model storage
3. **Cloud GPU Account** - RunPod, Colab, Lambda Labs, or Vast.ai

---

## 🔧 Setup on GitHub

### 1. Create GitHub Repository

```bash
# Initialize git (if not already done)
cd /path/to/asr
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: High-architecture ASR training system"

# Create repo on GitHub, then:
git remote add origin https://github.com/NursultanMRX/asr-training-system.git
git branch -M main
git push -u origin main
```

### 2. Update Repository URLs

Replace `NursultanMRX` in these files:
- `colab_notebook.ipynb` (line with `git clone`)
- `README.md` (badges and links)
- Update `HF_USERNAME` in `src/optimized_training.py`

---

## ☁️ Google Colab Deployment

### Method 1: Using Provided Notebook (Easiest)

1. **Upload `colab_notebook.ipynb` to your GitHub repo**

2. **Open in Colab:**
   - Go to: `https://colab.research.google.com/github/NursultanMRX/asr-training-system/blob/main/colab_notebook.ipynb`
   - Or click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NursultanMRX/asr-training-system/blob/main/colab_notebook.ipynb)

3. **Select GPU:**
   - Runtime → Change runtime type → GPU
   - Recommended: L4 for XLS-R-1B

4. **Run All Cells**

### Method 2: Manual Setup

```python
# Cell 1: Clone repo
!git clone https://github.com/NursultanMRX/asr-training-system.git
%cd asr-training-system

# Cell 2: Install
!bash setup.sh

# Cell 3: Login to HF
from huggingface_hub import login
login()

# Cell 4: Run training
!python src/optimized_training.py
```

### Colab-Specific Tips

**Prevent Disconnect:**
```javascript
// Run in browser console
function ClickConnect(){
    console.log("Keeping alive");
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
```

**GPU Options:**
- **T4 (Free):** 16GB - Good for XLS-R-300M
- **L4 (Pro):** 24GB - Perfect for XLS-R-1B ✅
- **A100 (Pro+):** 40GB - Best performance

**Session Limits:**
- Free: 12 hours max, may disconnect
- Pro: 24 hours max
- Pro+: Up to 24 hours with high priority

---

## 🖥️ RunPod Deployment

### 1. Create Pod

1. Go to [RunPod.io](https://runpod.io)
2. Select GPU: L4, RTX 4090, or A100
3. Choose template: "PyTorch" or "RunPod Pytorch"
4. Start pod

### 2. Connect via SSH or Web Terminal

```bash
# SSH connection (get details from RunPod dashboard)
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

### 3. Setup and Run

```bash
# Clone repository
git clone https://github.com/NursultanMRX/asr-training-system.git
cd asr-training-system

# Install dependencies
bash setup.sh

# Set HuggingFace token
export HF_TOKEN='your_hf_token_here'

# Run training in background
nohup python src/optimized_training.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### RunPod-Specific Tips

**Persistent Storage:**
```bash
# Use /workspace (persistent across sessions)
cd /workspace
git clone https://github.com/NursultanMRX/asr-training-system.git
```

**Cost Optimization:**
- **Spot Instances:** 50-70% cheaper (may be interrupted)
- **Reserved Instances:** Guaranteed availability
- **GPU Recommendations:**
  - L4 ($0.40-0.60/hr): Best value for XLS-R-1B
  - RTX 4090 ($0.60-0.90/hr): Fastest
  - A100 ($1.50-2.50/hr): Large batch sizes

**Auto-shutdown:**
```bash
# Stop pod after training completes
python src/optimized_training.py && shutdown -h now
```

---

## ⚡ Lambda Labs Deployment

### 1. Launch Instance

1. Go to [Lambda Labs Cloud](https://lambdalabs.com/service/gpu-cloud)
2. Select GPU: A100 (40GB or 80GB)
3. Choose region with availability
4. Launch instance

### 2. Connect and Setup

```bash
# SSH (get from dashboard)
ssh ubuntu@<instance-ip>

# Clone repo
git clone https://github.com/NursultanMRX/asr-training-system.git
cd asr-training-system

# Setup
bash setup.sh

# Run training
python src/optimized_training.py
```

### Lambda-Specific Tips

- **Pre-installed:** PyTorch usually pre-installed
- **Storage:** 512GB NVMe SSD included
- **Network:** 1 Gbps up/down
- **Cost:** ~$1.10-2.50/hr for A100

---

## 🌐 Vast.ai Deployment

### 1. Find and Rent GPU

1. Go to [Vast.ai](https://vast.ai)
2. Search for: L4, RTX 4090, or A100
3. Filter: "Verified" hosts
4. Rent instance

### 2. Setup

```bash
# SSH (get from Vast.ai console)
ssh root@<ip> -p <port>

# Update and install git
apt-get update && apt-get install -y git

# Clone repo
git clone https://github.com/NursultanMRX/asr-training-system.git
cd asr-training-system

# Setup
bash setup.sh

# Run
python src/optimized_training.py
```

### Vast.ai-Specific Tips

- **Cheapest Option:** Often 30-50% cheaper than others
- **Reliability:** Varies by host (check reviews)
- **Interruptible:** Some machines may restart
- **Recommendation:** Use verified hosts with \u003e4.5 stars

---

## 📊 Platform Comparison

| Platform | GPU Options | Cost (L4) | Ease of Use | Best For |
|----------|-------------|-----------|-------------|----------|
| **Google Colab** | T4, L4, A100 | Free-$50/mo | ⭐⭐⭐⭐⭐ | Beginners, experiments |
| **RunPod** | All GPUs | $0.40-0.60/hr | ⭐⭐⭐⭐ | Flexible, good value |
| **Lambda Labs** | A100 mainly | $1.10/hr | ⭐⭐⭐⭐ | Simple, reliable |
| **Vast.ai** | All GPUs | $0.30-0.50/hr | ⭐⭐⭐ | Budget-conscious |

---

## 🔐 Security Best Practices

### 1. Never Commit Tokens

```bash
# Create .env file (NOT committed)
echo "HF_TOKEN=your_token_here" > .env

# Load in Python
from dotenv import load_dotenv
load_dotenv()
import os
token = os.getenv('HF_TOKEN')
```

### 2. Use Environment Variables

```bash
# Set token
export HF_TOKEN='your_token_here'

# Use in Python
import os
from huggingface_hub import login
login(token=os.environ.get('HF_TOKEN'))
```

### 3. GitHub Secrets (for CI/CD)

- Repository Settings → Secrets and variables → Actions
- Add: `HF_TOKEN`
- Use in workflows: `${{ secrets.HF_TOKEN }}`

---

## 📦 Complete Deployment Workflow

### 1. Prepare Code
```bash
# Update configs
nano src/optimized_training.py  # Set DATASET_REPO_ID, MODEL_NAME, etc.

# Test locally (optional)
python src/optimized_training.py

# Commit changes
git add .
git commit -m "Update training config"
git push
```

### 2. Deploy to Cloud
```bash
# On cloud instance
git clone https://github.com/NursultanMRX/asr-training-system.git
cd asr-training-system
bash setup.sh
python src/optimized_training.py
```

### 3. Monitor Training
```bash
# Check logs
tail -f training.log

# Check GPU usage
nvidia-smi

# TensorBoard
tensorboard --logdir outputs/ --host 0.0.0.0 --port 6006
```

### 4. Download Results
```bash
# Zip outputs
tar -czf training_results.tar.gz outputs/ configs/

# Download via scp
scp user@host:/path/to/training_results.tar.gz ./
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

**Solution 1:** Reduce safety margin
```python
# In src/optimized_training.py
safety_margin=0.70  # Instead of 0.85
```

**Solution 2:** Use smaller model
```python
BASE_MODEL = "facebook/wav2vec2-xls-r-300m"
```

### Connection Timeout (Colab)

**Solution:** Keep browser open and run keep-alive script (see Colab section)

### Slow Download Speed

**Solution:** Use cloud storage
```bash
# Upload dataset to cloud first
wget https://your-cloud-storage/dataset.tar.gz
```

### Disk Space Full

**Solution:** Clean up
```bash
# Remove HuggingFace cache
rm -rf ~/.cache/huggingface

# Remove old checkpoints
rm -rf outputs/checkpoint-*
```

---

## ⏱️ Expected Training Times

### 60-hour Karakalpak Dataset

| GPU | Model | Batch | Time | Cost |
|-----|-------|-------|------|------|
| **T4** | XLS-R-300M | 16 | ~36h | Free (Colab) |
| **L4** | XLS-R-1B | 4 | ~15h | $6-9 (RunPod) |
| **RTX 4090** | XLS-R-1B | 8 | ~10h | $6-9 (RunPod) |
| **A100 40GB** | XLS-R-1B | 16 | ~6h | $9-15 (Lambda) |

---

## 🎯 Recommended Configurations

### Budget (<$10)
- **Platform:** Vast.ai or RunPod (spot)
- **GPU:** L4 or RTX 4090
- **Duration:** 12-15 hours
- **Cost:** $5-8

### Speed (<6 hours)
- **Platform:** Lambda Labs or RunPod
- **GPU:** A100 40GB
- **Duration:** 6-8 hours
- **Cost:** $9-15

### Free (Colab)
- **Platform:** Google Colab
- **GPU:** T4 or L4 (Pro)
- **Duration:** 24-36 hours
- **Cost:** Free (T4) or $10/mo (L4 via Pro)

---

## 📝 Post-Training Checklist

- [ ] Training completed successfully
- [ ] Model pushed to HuggingFace Hub
- [ ] Downloaded checkpoints (backup)
- [ ] Saved `training_config.json`
- [ ] Stopped/terminated cloud instance
- [ ] Documented results (WER, training time)

---

## 🔗 Quick Links

- **RunPod:** https://runpod.io
- **Google Colab:** https://colab.research.google.com
- **Lambda Labs:** https://lambdalabs.com
- **Vast.ai:** https://vast.ai
- **HuggingFace Hub:** https://huggingface.co

---

## 📞 Support

Issues? Check:
1. [`RUN.md`](RUN.md) - Main execution guide
2. [`docs/QUICK_START.md`](docs/QUICK_START.md) - Quick start
3. GitHub Issues - Report problems

---

**Happy Training on the Cloud! ☁️🚀**
