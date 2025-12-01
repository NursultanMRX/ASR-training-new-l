# 🚀 GitHub & Cloud Deployment Checklist

## ✅ Pre-Deployment Checklist

### 1. Code Preparation

- [ ] **Update repository URLs**
  - [ ] Replace `NursultanMRX` in `README.md`
  - [ ] Replace `NursultanMRX` in `colab_notebook.ipynb`
  - [ ] Replace `NursultanMRX` in `DEPLOYMENT.md`

- [ ] **Update training configuration**
  - [ ] Set `DATASET_REPO_ID` in `src/optimized_training.py`
  - [ ] Set `HF_USERNAME` in `src/optimized_training.py`
  - [ ] Set `MODEL_NAME` in `src/optimized_training.py`

- [ ] **Verify all files exist**
  ```bash
  python verify_setup.py
  ```

### 2. Dependencies Check

- [ ] **Test requirements.txt**
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **Test setup script**
  ```bash
  bash setup.sh
  ```

- [ ] **Verify versions**
  - [ ] Python ≥ 3.8
  - [ ] PyTorch ≥ 2.0
  - [ ] Transformers ≥ 4.35
  - [ ] CUDA available (for GPU training)

### 3. GitHub Setup

- [ ] **Initialize Git (if not done)**
  ```bash
  git init
  git add .
  git commit -m "Initial commit: High-architecture ASR training system"
  ```

- [ ] **Create GitHub repository**
  - Go to https://github.com/new
  - Name: `asr-training-system` (or your choice)
  - Public or Private
  - Don't initialize with README (we have one)

- [ ] **Push to GitHub**
  ```bash
  git remote add origin https://github.com/NursultanMRX/asr-training-system.git
  git branch -M main
  git push -u origin main
  ```

- [ ] **Verify GitHub Actions**
  - Check `.github/workflows/verify.yml` runs successfully
  - Fix any errors

### 4. Security

- [ ] **Add .gitignore** (✅ Already created)
- [ ] **Never commit tokens/secrets**
  - HuggingFace tokens
  - API keys
  - Passwords

- [ ] **Use environment variables**
  ```bash
  export HF_TOKEN='your_token_here'
  ```

### 5. Documentation

- [ ] **Update README badges** with your username
- [ ] **Add project description** if needed
- [ ] **Update LICENSE** if needed (currently MIT)

---

## 🌐 Cloud Platform Deployment

### Google Colab

- [ ] **Test Colab notebook**
  - Open `colab_notebook.ipynb` in Colab
  - Run all cells
  - Verify auto-configuration works

- [ ] **Update Colab badge** in README with correct URL

### RunPod

- [ ] **Create RunPod account** (if needed)
- [ ] **Test deployment**
  ```bash
  git clone https://github.com/NursultanMRX/asr-training-system.git
  cd asr-training-system
  bash setup.sh
  python src/optimized_training.py
  ```

- [ ] **Verify GPU detection**
- [ ] **Test training for few steps**

### Lambda Labs / Vast.ai

- [ ] **Follow DEPLOYMENT.md** instructions
- [ ] **Test on small dataset first**

---

## ✅ Post-Deployment Verification

### After Pushing to GitHub

- [ ] **Repository is public/accessible**
- [ ] **All files visible** (check branch is `main`)
- [ ] **README renders correctly** with badges
- [ ] **Colab badge works** (click and opens notebook)
- [ ] **No sensitive data committed** (check git history)

### After Cloud Deployment

- [ ] **Dependencies install without errors**
- [ ] **GPU detected** (if cloud has GPU)
- [ ] **HuggingFace login works**
- [ ] **Dataset loads successfully**
- [ ] **Auto-configuration runs**
- [ ] **Training starts**
- [ ] **Checkpoints save**

## 🎯 Quick Deployment Commands

### First-Time GitHub Push
```bash
# From project root
git init
git add .
git commit -m "feat: High-architecture ASR training system"
git remote add origin https://github.com/NursultanMRX/asr-training-system.git
git branch -M main
git push -u origin main
```

### Update After Changes
```bash
git add .
git commit -m "Update: [describe changes]"
git push
```

### Deploy on RunPod
```bash
# SSH to RunPod instance
git clone https://github.com/NursultanMRX/asr-training-system.git
cd asr-training-system
bash setup.sh
export HF_TOKEN='your_token'
python src/optimized_training.py
```

### Deploy on Colab
1. Click badge in README
2. Runtime → Change runtime type → GPU
3. Run all cells
4. Enter HF token when prompted

---

## 🔧 Common Issues & Fixes

### Issue: GitHub push rejected

**Fix:**
```bash
git pull origin main --rebase
git push origin main
```

### Issue: Large files error

**Fix:**
```bash
# Remove large files from git
git rm --cached outputs/*
git commit -m "Remove large files"
git push
```

### Issue: Requirements install fails

**Fix:**
```bash
# Update pip first
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Issue: CUDA not available

**Check:**
```bash
nvidia-smi  # Should show GPU
python -c "import torch; print(torch.cuda.is_available())"
```

**Fix:** Select GPU in Colab/RunPod settings

---

## 📊 Verification Tests

### Test 1: Local Verification
```bash
python verify_setup.py
```
**Expected:** All ✅ checks pass

### Test 2: Import Test
```python
from src.asr_config_manager import create_optimal_config
print("✅ Import successful")
```

### Test 3: GPU Test
```python
import torch
assert torch.cuda.is_available(), "No GPU detected!"
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
```

### Test 4: Quick Training Test
```bash
# Run for just 1 step to verify everything works
python src/optimized_training.py  # Ctrl+C after first step
```

---

## 🎉 Success Criteria

Your deployment is successful when:

- [ ] ✅ Code on GitHub (public/accessible)
- [ ] ✅ README displays correctly
- [ ] ✅ Colab notebook opens and runs
- [ ] ✅ Cloud platform deployment works
- [ ] ✅ Dependencies install without errors
- [ ] ✅ GPU detected (if using GPU platform)
- [ ] ✅ Training starts successfully
- [ ] ✅ Auto-configuration generates optimal settings
- [ ] ✅ No OOM errors
- [ ] ✅ Checkpoints save correctly

---

## 📝 Final Notes

1. **Always test locally first** before cloud deployment
2. **Start with small dataset** to verify everything works
3. **Monitor costs** on paid platforms (RunPod, Lambda, Vast.ai)
4. **Keep HF token secure** - never commit it
5. **Document your results** - WER, training time, cost
6. **Share your trained model** on HuggingFace Hub

---

## 🔗 Resources

- **GitHub Repo:** Will be at `https://github.com/NursultanMRX/asr-training-system`
- **Colab Notebook:** Update URL in README after pushing
- **Deployment Guide:** [DEPLOYMENT.md](DEPLOYMENT.md)
- **Main Documentation:** [RUN.md](RUN.md)

---

**Ready to deploy? Start with the checklist above! 🚀**
