# Quick Integration Guide: Add High-Architecture Config to Your Existing Notebook

## 🚀 Quick Start (2 Steps!)

### Step 1: Add the Config Manager Cell

Add this cell **BEFORE** your model loading section:

```python
# ============================================================================
# CELL: HIGH-ARCHITECTURE CONFIGURATION MANAGER
# ============================================================================
# Copy the entire asr_config_manager.py content here, OR:
# Upload asr_config_manager.py to Colab and import it

from asr_config_manager import create_optimal_config, ASRConfigManager

print("✅ High-Architecture Config Manager loaded!")
```

### Step 2: Replace Your Training Configuration

Replace your existing training configuration with this:

```python
# ============================================================================
# AUTO-GENERATE OPTIMAL CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("🎯 GENERATING OPTIMIZED CONFIGURATION")
print("="*80)

# This automatically profiles everything and generates optimal config!
training_config, config_manager = create_optimal_config(
    dataset=raw_datasets['train'],
    model=model,
    model_name=MODEL_NAME,
    num_epochs=20,
    target_batch_size=32,  # Your desired effective batch size
    learning_rate=3e-4,
    safety_margin=0.85  # Use 85% of available memory (adjust 0.7-0.95)
)

# Save for future reference
config_manager.save_config(training_config, "training_config.json")

print("\n✅ Configuration optimized and saved!")
```

### Step 3: Use Auto-Config in TrainingArguments

Update your TrainingArguments to use the auto-generated config:

```python
# ============================================================================
# TRAINING ARGUMENTS (WITH AUTO-CONFIG)
# ============================================================================

training_args = TrainingArguments(
    output_dir=MODEL_NAME,
    logging_dir=f"{MODEL_NAME}/logs",
    
    # ✨ AUTO-OPTIMIZED SETTINGS ✨
    per_device_train_batch_size=training_config.per_device_train_batch_size,
    per_device_eval_batch_size=training_config.per_device_eval_batch_size,
    gradient_accumulation_steps=training_config.gradient_accumulation_steps,
    gradient_checkpointing=training_config.gradient_checkpointing,
    fp16=training_config.fp16,
    dataloader_num_workers=training_config.dataloader_num_workers,
    
    # TRAINING PARAMETERS
    num_train_epochs=training_config.num_train_epochs,
    learning_rate=training_config.learning_rate,
    warmup_steps=training_config.warmup_steps,
    optim="adafactor",  # Memory-efficient optimizer
    
    # EVALUATION (AUTO-OPTIMIZED)
    eval_strategy="steps",
    eval_steps=training_config.eval_steps,
    save_steps=training_config.save_steps,
    logging_steps=training_config.logging_steps,
    
    # OTHER SETTINGS
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
    push_to_hub=True,
    hub_model_id=f"{HF_USERNAME}/{MODEL_NAME}",
    hub_private_repo=True,
    report_to=["tensorboard"],
    seed=42,
)

print("✅ TrainingArguments created with auto-optimized settings!")
print(f"   Batch size: {training_config.per_device_train_batch_size}")
print(f"   Gradient accumulation: {training_config.gradient_accumulation_steps}")
print(f"   Effective batch size: {training_config.effective_batch_size}")
```

---

## 📝 Complete Cell-by-Cell Integration

Here's exactly what to change in your notebook:

### Original Notebook Structure:
1. Install dependencies
2. Login to HF
3. Environment setup
4. Load dataset
5. Create vocabulary
6. Create processor
7. Load model
8. **Configure training** ← CHANGE THIS
9. Process dataset
10. Create trainer
11. Train

### What to Change:

#### Cell 8 (Training Configuration) - BEFORE:
```python
# Old manual configuration
batch_size = 32  # ❌ Manual, might cause OOM
gradient_accumulation = 1  # ❌ Not adaptive
```

#### Cell 8 (Training Configuration) - AFTER:
```python
# ✅ NEW: Auto-configuration
training_config, config_manager = create_optimal_config(
    dataset=raw_datasets['train'],
    model=model,
    model_name=MODEL_NAME,
    num_epochs=20,
    target_batch_size=32
)

# Save configuration
config_manager.save_config(training_config, "training_config.json")
```

#### Update your prepare_dataset function:
```python
def prepare_dataset_safe(batch):
    # Use auto-determined max duration
    max_duration = training_config.max_audio_duration_seconds
    
    # ... rest of your code
```

---

## 🎯 Example: Full Modified Training Cell

```python
# ============================================================================
# TRAINING CONFIGURATION - HIGH ARCHITECTURE VERSION
# ============================================================================

# Auto-generate optimal configuration
training_config, config_manager = create_optimal_config(
    dataset=raw_datasets['train'],
    model=model,
    model_name=MODEL_NAME,
    num_epochs=20,
    target_batch_size=32,
    learning_rate=3e-4
)

# Use in TrainingArguments
training_args = TrainingArguments(
    output_dir=MODEL_NAME,
    
    # Auto-optimized batch configuration
    per_device_train_batch_size=training_config.per_device_train_batch_size,
    per_device_eval_batch_size=training_config.per_device_eval_batch_size,
    gradient_accumulation_steps=training_config.gradient_accumulation_steps,
    
    # Auto-optimized memory settings
    gradient_checkpointing=training_config.gradient_checkpointing,
    fp16=training_config.fp16,
    dataloader_num_workers=training_config.dataloader_num_workers,
    
    # Training parameters
    num_train_epochs=training_config.num_train_epochs,
    learning_rate=training_config.learning_rate,
    warmup_steps=training_config.warmup_steps,
    optim="adafactor",
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=training_config.eval_steps,
    save_steps=training_config.save_steps,
    logging_steps=training_config.logging_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    
    # Other
    remove_unused_columns=False,
    push_to_hub=True,
    hub_model_id=f"{HF_USERNAME}/{MODEL_NAME}",
    hub_private_repo=True,
    report_to=["tensorboard"],
    seed=42,
)

print("✅ Configuration complete!")
print(f"\nOptimized Settings:")
print(f"  • Batch Size: {training_config.per_device_train_batch_size}")
print(f"  • Gradient Accumulation: {training_config.gradient_accumulation_steps}")
print(f"  • Effective Batch Size: {training_config.effective_batch_size}")
print(f"  • FP16: {training_config.fp16}")
print(f"  • Gradient Checkpointing: {training_config.gradient_checkpointing}")
```

---

## 🔧 Advanced: Custom Overrides

If you want to override specific settings:

```python
# Generate base config
config, manager = create_optimal_config(...)

# Customize specific settings
config.per_device_train_batch_size = 8  # Force specific batch size
config.num_train_epochs = 30  # More epochs
config.learning_rate = 1e-4  # Different LR

# Use customized config
training_args = TrainingArguments(
    per_device_train_batch_size=config.per_device_train_batch_size,
    ...
)
```

---

## 🎨 Optional: Add Memory Monitoring

Add this callback to your trainer for real-time memory monitoring:

```python
from transformers import TrainerCallback

class MemoryMonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 100 == 0:
            mem_info = get_memory_info()
            print(f"\nStep {state.global_step}:")
            print(f"  GPU Memory: {mem_info['gpu_used_gb']:.2f}/{mem_info['gpu_total_gb']:.2f} GB")

# Add to trainer
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[MemoryMonitorCallback()],
    ...
)
```

---

## ✅ Verification Checklist

After integration, verify:

- [ ] Config manager imported successfully
- [ ] Auto-config runs without errors
- [ ] Batch sizes are reasonable (not 1, not >64)
- [ ] Training starts without OOM
- [ ] Memory monitoring shows usage <90%
- [ ] Configuration saved to `training_config.json`

---

## 🆘 Troubleshooting

### Problem: Import error for `asr_config_manager`

**Solution:**
```python
# In Colab, upload the file first
from google.colab import files
uploaded = files.upload()  # Upload asr_config_manager.py

# Then import
from asr_config_manager import create_optimal_config
```

OR paste the entire content of `asr_config_manager.py` into a notebook cell.

### Problem: Still getting OOM

**Solution:**
```python
# Reduce safety margin
config, manager = create_optimal_config(
    dataset=raw_datasets['train'],
    model=model,
    safety_margin=0.70  # Use only 70% of memory (more conservative)
)
```

### Problem: Training too slow

**Solution:**
```python
# Increase safety margin to use more memory
config, manager = create_optimal_config(
    dataset=raw_datasets['train'],
    model=model,
    safety_margin=0.95  # Use 95% of memory (more aggressive)
)
```

---

## 📊 Example Output

When you run the auto-config, you'll see:

```
================================================================================
🎯 GENERATING OPTIMIZED CONFIGURATION
================================================================================
🔍 Profiling hardware...
🔍 Analyzing dataset...
🔍 Analyzing model...
⚙️  Generating optimal configuration...

================================================================================
ASR TRAINING CONFIGURATION SUMMARY
================================================================================

Hardware Profile:
├─ GPU: NVIDIA L4
│  ├─ Total: 23.80 GB
│  └─ Available: 20.23 GB
...

✅ Configuration complete!

Optimized Settings:
  • Batch Size: 4
  • Gradient Accumulation: 8
  • Effective Batch Size: 32
  • FP16: True
  • Gradient Checkpointing: True
```

---

That's it! Your notebook is now using **high-architecture adaptive configuration**! 🚀
