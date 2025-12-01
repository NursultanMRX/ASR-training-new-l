# High-Architecture ASR Training Configuration System

## 🎯 Overview

This system provides **automatic, intelligent configuration** for training large ASR models by dynamically adapting all settings based on:

- **Hardware Capabilities** (GPU/CPU RAM, CUDA version)
- **Dataset Characteristics** (volume, duration, size)
- **Model Architecture** (parameters, layers, memory footprint)

**No more manual tuning or OOM crashes!**

---

## 🚀 Features

### ✅ Automatic Resource Optimization
- **Dynamic Batch Sizing**: Calculates optimal batch size based on available GPU memory
- **Smart Gradient Accumulation**: Automatically determines accumulation steps to achieve target batch size
- **Memory-Safe Processing**: Prevents OOM errors with intelligent safety margins

### ✅ Adaptive Configuration
- **Hardware Profiling**: Analyzes GPU/CPU specs and available memory
- **Dataset Analysis**: Samples dataset to understand audio duration patterns
- **Model Inspection**: Calculates memory requirements for model + gradients + optimizer

### ✅ Production-Ready Safety
- **Memory Monitoring**: Real-time tracking of RAM usage during training
- **Automatic Recovery**: Retries with reduced batch size on OOM errors
- **Checkpoint Management**: Saves configuration for reproducibility

---

## 📦 Installation

The system uses standard HuggingFace libraries:

```bash
pip install transformers datasets accelerate
pip install torchaudio soundfile librosa
pip install jiwer evaluate
pip install psutil
```

---

## 🔧 Usage

### Option 1: Quick Start (Recommended)

Use the convenience function in your training script:

```python
from asr_config_manager import create_optimal_config

# After loading your dataset and model:
config, manager = create_optimal_config(
    dataset=raw_datasets['train'],
    model=model,
    model_name='wav2vec2-xls-r-1b-karakalpak',
    num_epochs=20,
    target_batch_size=32,
    learning_rate=3e-4,
    safety_margin=0.85  # Use 85% of available memory
)

# The config is automatically optimized!
print(config)
```

**Output Example:**
```
============================================================
ASR TRAINING CONFIGURATION SUMMARY
============================================================

Hardware Profile:
├─ GPU: NVIDIA L4
│  ├─ Total: 23.80 GB
│  └─ Available: 20.23 GB
├─ CPU RAM:
│  ├─ Total: 56.86 GB
│  └─ Available: 48.33 GB
└─ CUDA: 12.6

Dataset Profile:
├─ Samples: 26,670
├─ Duration:
│  ├─ Average: 8.12s
│  ├─ Min: 1.23s
│  └─ Max: 29.87s
├─ Total Hours: 60.12h
└─ Estimated Size: 6.84 GB

Model Profile:
├─ Name: wav2vec2-xls-r-1b
├─ Parameters:
│  ├─ Total: 1,267,345,984
│  └─ Trainable: 123,456,789
├─ Architecture:
│  ├─ Hidden Size: 1280
│  └─ Layers: 48
└─ Estimated Size: 12.45 GB

Training Configuration:
├─ Batch Configuration:
│  ├─ Train Batch Size: 4
│  ├─ Eval Batch Size: 4
│  ├─ Gradient Accumulation: 8
│  └─ Effective Batch Size: 32
├─ Memory Optimizations:
│  ├─ Gradient Checkpointing: True
│  ├─ Mixed Precision (FP16): True
│  ├─ DataLoader Workers: 0
│  └─ Max Audio Duration: 29.87s
├─ Data Processing:
│  ├─ Audio Chunk Length: 20.0s
│  ├─ Use Streaming: False
│  └─ Cache Dataset: False
└─ Safety:
   ├─ Memory Reserve: 2.49 GB
   └─ Max Memory Usage: 90%
============================================================
```

### Option 2: Step-by-Step Manual Control

```python
from asr_config_manager import ASRConfigManager

# 1. Initialize manager
manager = ASRConfigManager(safety_margin=0.85)

# 2. Profile hardware (automatic)
hardware = manager.hardware
print(hardware)

# 3. Analyze dataset
dataset_profile = manager.profile_dataset(raw_datasets['train'])
print(dataset_profile)

# 4. Analyze model
model_profile = manager.profile_model(model, 'wav2vec2-xls-r-1b')
print(model_profile)

# 5. Generate configuration
config = manager.generate_config(
    model_profile=model_profile,
    dataset_profile=dataset_profile,
    num_epochs=20,
    target_batch_size=32
)

# 6. Save configuration
manager.save_config(config, 'training_config.json')
```

### Option 3: Use Optimized Training Script

We've included a complete, production-ready training script:

```bash
python optimized_training_notebook.py
```

This script:
- ✅ Automatically profiles your hardware
- ✅ Analyzes your dataset
- ✅ Configures optimal settings
- ✅ Trains with memory monitoring
- ✅ Auto-recovers from OOM errors
- ✅ Saves checkpoints and pushes to HuggingFace Hub

---

## 📊 How It Works

### 1. Hardware Profiling
```python
HardwareProfile(
    gpu_total_gb=23.80,
    gpu_available_gb=20.23,  # After applying safety margin
    cpu_total_gb=56.86,
    cpu_available_gb=48.33,
    gpu_name="NVIDIA L4",
    cuda_version="12.6"
)
```

### 2. Dataset Analysis
Samples random subset of dataset to calculate:
- Average/min/max audio duration
- Estimated dataset size
- Total training hours

### 3. Model Inspection
Analyzes model to determine:
- Total parameters
- Trainable parameters
- Memory footprint (params + gradients + optimizer states)

### 4. Optimal Configuration Calculation

**Batch Size Calculation:**
```python
# Memory per sample
audio_memory = avg_duration * 16000 * 4 bytes  # float32
activation_memory = model_size * 0.1

total_per_sample = audio_memory + activation_memory

# Available memory for batches
available_for_batch = gpu_available - model_size

# Maximum batch size
max_batch_size = available_for_batch / total_per_sample

# Apply safety constraints
batch_size = min(max_batch_size, 64)  # Cap at 64
gradient_accumulation = target_batch_size / batch_size
```

**Automatic Optimizations:**
- **FP16**: Enabled if GPU has ≥8GB
- **Gradient Checkpointing**: Enabled for models >100M params
- **Streaming**: Enabled for datasets >50GB
- **Workers**: 0 if CPU RAM <32GB, else 2

---

## ⚙️ Configuration Parameters

### ASRConfigManager Initialization

```python
ASRConfigManager(safety_margin=0.85)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `safety_margin` | float | 0.85 | Fraction of memory to use (0.0-1.0) |

### create_optimal_config()

```python
create_optimal_config(
    dataset,
    model,
    model_name="wav2vec2-xls-r-1b",
    num_epochs=20,
    target_batch_size=32,
    learning_rate=3e-4,
    safety_margin=0.85
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | Dataset | **Required** | HuggingFace dataset (train split) |
| `model` | PreTrainedModel | **Required** | The ASR model |
| `model_name` | str | "wav2vec2-xls-r-1b" | Model identifier |
| `num_epochs` | int | 20 | Number of training epochs |
| `target_batch_size` | int | 32 | Desired effective batch size |
| `learning_rate` | float | 3e-4 | Learning rate |
| `safety_margin` | float | 0.85 | Memory safety margin |

---

## 🎯 Integration with TrainingArguments

Use the generated config in HuggingFace Trainer:

```python
from transformers import TrainingArguments, Trainer

# Generate optimized config
config, manager = create_optimal_config(
    dataset=raw_datasets['train'],
    model=model,
    model_name='my-asr-model'
)

# Use in TrainingArguments
training_args = TrainingArguments(
    output_dir="my-asr-model",
    
    # Use auto-optimized settings
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    gradient_checkpointing=config.gradient_checkpointing,
    fp16=config.fp16,
    dataloader_num_workers=config.dataloader_num_workers,
    
    # Training parameters
    num_train_epochs=config.num_train_epochs,
    learning_rate=config.learning_rate,
    warmup_steps=config.warmup_steps,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=config.eval_steps,
    save_steps=config.save_steps,
    logging_steps=config.logging_steps,
    
    # Other settings
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    push_to_hub=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train!
trainer.train()
```

---

## 🛡️ Safety Features

### Memory Monitoring Callback

The system includes an adaptive memory monitoring callback:

```python
class AdaptiveMemoryCallback(TrainerCallback):
    """Monitor memory and adapt during training"""
    
    def on_step_begin(self, args, state, control, **kwargs):
        # Check memory before each step
        if gpu_memory_percent > 90:
            cleanup_memory()
```

### Automatic Recovery

```python
max_retries = 3
retry_count = 0

while retry_count < max_retries:
    try:
        trainer.train()
        break
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Reduce batch size and retry
            trainer.args.per_device_train_batch_size //= 2
            retry_count += 1
```

---

## 📈 Expected Performance

### Memory Usage Examples

| GPU | Model | Dataset Size | Auto Batch Size | Gradient Accum | Effective Batch |
|-----|-------|--------------|-----------------|----------------|-----------------|
| L4 (24GB) | XLS-R-1B | 60h | 4 | 8 | 32 |
| A100 (40GB) | XLS-R-1B | 60h | 8 | 4 | 32 |
| 3090 (24GB) | XLS-R-300M | 60h | 32 | 1 | 32 |
| T4 (16GB) | XLS-R-300M | 60h | 16 | 2 | 32 |
| V100 (32GB) | XLS-R-1B | 60h | 8 | 4 | 32 |

### Optimization Benefits

- **No manual tuning**: System automatically finds optimal settings
- **Prevents OOM**: Safety margins prevent crashes
- **Maximizes throughput**: Uses as much memory as safely possible
- **Adapts to changes**: Recalculates if you change model/dataset

---

## 🔍 Troubleshooting

### Issue: Still getting OOM errors

**Solution:**
1. Reduce safety margin: `ASRConfigManager(safety_margin=0.75)`
2. Reduce max audio duration in dataset preparation
3. Use model with fewer parameters (e.g., XLS-R-300M instead of XLS-R-1B)

### Issue: Training is too slow

**Solution:**
1. Increase safety margin: `ASRConfigManager(safety_margin=0.90)`
2. Enable more workers if you have CPU RAM: Set manually in config
3. Use larger batch size: Increase `target_batch_size`

### Issue: Want to override auto-config

**Solution:**
```python
config, manager = create_optimal_config(...)

# Override specific settings
config.per_device_train_batch_size = 8
config.gradient_accumulation_steps = 4

# Use modified config
training_args = TrainingArguments(
    per_device_train_batch_size=config.per_device_train_batch_size,
    ...
)
```

---

## 📝 Configuration File Format

The system saves configuration as JSON:

```json
{
  "per_device_train_batch_size": 4,
  "per_device_eval_batch_size": 4,
  "gradient_accumulation_steps": 8,
  "effective_batch_size": 32,
  "gradient_checkpointing": true,
  "fp16": true,
  "dataloader_num_workers": 0,
  "max_audio_duration_seconds": 30.0,
  "audio_chunk_length_seconds": 20.0,
  "use_streaming": false,
  "cache_dataset": false,
  "num_train_epochs": 20,
  "learning_rate": 0.0003,
  "warmup_steps": 500,
  "eval_steps": 200,
  "save_steps": 200,
  "logging_steps": 50,
  "memory_reserve_gb": 2.49,
  "max_memory_usage_percent": 90.0
}
```

---

## 🤝 Contributing

To add support for new model architectures:

1. Update `profile_model()` to extract architecture-specific details
2. Adjust memory estimation formulas in `calculate_optimal_batch_size()`
3. Test with your specific model and dataset

---

## 📚 References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [XLS-R Paper](https://arxiv.org/abs/2111.09296)
- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)

---

## 📄 License

This configuration system is provided as-is for use with the Karakalpak ASR training project.

---

## 🎓 Example: Full Workflow

```python
# 1. Install dependencies (one time)
!pip install transformers datasets torchaudio psutil

# 2. Import and setup
from asr_config_manager import create_optimal_config
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Trainer, TrainingArguments

# 3. Load your data and model
dataset = load_dataset("your-dataset")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-1b")

# 4. Generate optimal configuration (MAGIC HAPPENS HERE!)
config, manager = create_optimal_config(
    dataset=dataset['train'],
    model=model,
    model_name='my-custom-asr',
    num_epochs=20,
    target_batch_size=32
)

# 5. Use in training
training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    fp16=config.fp16,
    num_train_epochs=config.num_train_epochs,
    # ... other args
)

trainer = Trainer(model=model, args=training_args, ...)
trainer.train()

# 6. That's it! No OOM errors, no manual tuning!
```

---

**Happy Training! 🚀**
