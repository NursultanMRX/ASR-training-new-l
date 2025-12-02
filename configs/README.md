# 📋 JSON Configuration Guide

## Overview

The ASR Training System now supports **JSON-based configuration files** for managing all training parameters. This provides a cleaner, more organized way to configure your training runs.

## Quick Start

### 1. Using JSON Config

```bash
# Train with JSON config file
python src/cli.py train --config=configs/example_config.json

# Or programmatically
python -c "from src.optimized_training import train_asr_model; train_asr_model(config_file='configs/example_config.json')"
```

### 2. Using CLI Parameters (Legacy)

```bash
# Train with individual parameters
python src/cli.py train \
  --dataset_repo="nickoo004/karakalpak-speech-60h" \
  --model_name="facebook/wav2vec2-xls-r-1b" \
  --output_dir="./output"
```

### 3. Hybrid Approach (Config + Overrides)

```bash
# Use config but override specific parameters
python src/cli.py train \
  --config=configs/example_config.json \
  --epochs=50 \
  --learning_rate=1e-4
```

---

## Configuration File Structure

### Template: `config_template.json`

See [`config_template.json`](config_template.json) for the complete template with all available parameters.

### Example: `example_config.json`

A real-world example configuration for TTS training:

```json
{
  "model_name_or_path": "./mms-tts-kaa-with-discriminator",
  "dataset_name": "./my_local_dataset",
  "output_dir": "./mms-tts-kaa-finetuned-speaker1",
  "push_to_hub": true,
  "hub_model_id": "nickoo004/mms-tts-kaa-finetuned-speaker1",
  "hf_token": "",
  
  "audio_column_name": "file_name",
  "text_column_name": "text",
  "speaker_id_column_name": "speaker_name",
  "filter_on_speaker_id": "Speaker_1",
  
  "max_duration_in_seconds": 20.0,
  "min_duration_in_seconds": 1.0,
  
  "num_train_epochs": 150,
  "per_device_train_batch_size": 4,
  "learning_rate": 2e-5
}
```

---

## Configuration Parameters

### 🎯 Model & Dataset

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_name_or_path` | string | Path or HF model ID | - |
| `dataset_name` | string | Path or HF dataset ID | - |
| `output_dir` | string | Output directory | `"./output"` |
| `overwrite_output_dir` | bool | Overwrite existing output | `true` |

### 🔐 HuggingFace Hub

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `push_to_hub` | bool | Upload to HF Hub | `false` |
| `hub_model_id` | string | HF Hub model ID | `null` |
| `hf_token` | string | HF API token | `""` |

### 📊 Dataset Column Mapping

**NEW!** Configure custom column names:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `audio_column_name` | string | Audio column name | `"audio"` |
| `text_column_name` | string | Text column name | `"text"` |
| `cleaned_text_column_name` | string | Cleaned text column | `"cleaned_text"` |
| `speaker_id_column_name` | string | Speaker ID column | `null` |
| `filter_on_speaker_id` | string | Filter for specific speaker | `null` |

### 🎵 Audio Filtering

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `max_duration_in_seconds` | float | Max audio duration | `20.0` |
| `min_duration_in_seconds` | float | Min audio duration | `1.0` |

### 🎓 Training Hyperparameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `num_train_epochs` | int | Number of epochs | `20` |
| `per_device_train_batch_size` | int | Batch size per device | `4` |
| `learning_rate` | float | Learning rate | `3e-4` |
| `warmup_ratio` | float | Warmup ratio | `0.1` |
| `gradient_accumulation_steps` | int | Gradient accumulation | `1` |
| `gradient_checkpointing` | bool | Enable checkpointing | `true` |

### ⚖️ Loss Weights (for TTS/VITS models)

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `weight_disc` | float | Discriminator weight | `3.0` |
| `weight_fmaps` | float | Feature maps weight | `1.0` |
| `weight_gen` | float | Generator weight | `1.0` |
| `weight_kl` | float | KL divergence weight | `1.5` |
| `weight_duration` | float | Duration weight | `1.0` |
| `weight_mel` | float | Mel-spectrogram weight | `35.0` |

### ⚡ Optimization

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `fp16` | bool | Use mixed precision | `true` |
| `optim` | string | Optimizer type | `"adamw_torch"` |
| `max_grad_norm` | float | Gradient clipping | `1.0` |

---

## Usage Examples

### Example 1: Single Speaker TTS Training

```json
{
  "model_name_or_path": "./pretrained-tts-model",
  "dataset_name": "./my_speech_data",
  "output_dir": "./tts-speaker1-output",
  
  "audio_column_name": "file_name",
  "text_column_name": "text",
  "speaker_id_column_name": "speaker_name",
  "filter_on_speaker_id": "Speaker_1",
  
  "num_train_epochs": 150,
  "per_device_train_batch_size": 4,
  "learning_rate": 2e-5
}
```

Run with:
```bash
python src/cli.py train --config=my_config.json
```

### Example 2: Multi-Language ASR

```json
{
  "model_name_or_path": "facebook/wav2vec2-xls-r-1b",
  "dataset_name": "mozilla-foundation/common_voice_11_0",
  "output_dir": "./asr-multilingual",
  
  "text_column_name": "sentence",
  "max_duration_in_seconds": 30.0,
  
  "num_train_epochs": 30,
  "fp16": true
}
```

### Example 3: Private Dataset with HF Token

```json
{
  "model_name_or_path": "facebook/wav2vec2-large-xlsr-53",
  "dataset_name": "myusername/private-dataset",
  "hf_token": "hf_xxxxxxxxxxxxxxxxxxxxx",
  
  "push_to_hub": true,
  "hub_model_id": "myusername/my-finetuned-model"
}
```

---

## Best Practices

### 1. **Use Comments for Organization**

While JSON doesn't support comments, you can use keys starting with `//`:

```json
{
  "// MODEL SETTINGS": "",
  "model_name_or_path": "./my-model",
  
  "// TRAINING SETTINGS": "",
  "num_train_epochs": 50
}
```

These will be automatically filtered out by the config loader.

### 2. **Version Control Your Configs**

Store your configs in version control:
```bash
configs/
├── config_template.json     # Template
├── experiment_v1.json        # Experiment 1
├── experiment_v2.json        # Experiment 2
└── production.json           # Production settings
```

### 3. **Keep Sensitive Data Secure**

Never commit tokens to version control:
```json
{
  "hf_token": ""  // Set via environment variable instead
}
```

Use environment variables:
```bash
export HF_TOKEN="your_token_here"
python src/cli.py train --config=config.json --hf_token=$HF_TOKEN
```

### 4. **Start from Template**

Always start with `config_template.json` and modify:
```bash
cp configs/config_template.json configs/my_experiment.json
# Edit my_experiment.json
python src/cli.py train --config=configs/my_experiment.json
```

---

## Troubleshooting

### Config file not found
```
Error: FileNotFoundError: Config file not found: configs/myconfig.json
```
**Fix:** Check the file path is correct and use absolute or relative path.

### Invalid JSON
```
Error: json.JSONDecodeError: Expecting property name...
```
**Fix:** Validate your JSON at [jsonlint.com](https://jsonlint.com)

### Unknown parameter warning
```
⚠️ Unknown config parameter: typo_param (ignoring)
```
**Fix:** Check parameter name spelling against the template.

---

## Migration Guide

### From CLI to JSON

**Before (CLI):**
```bash
python src/cli.py train \
  --dataset_repo="my/dataset" \
  --model_name="facebook/wav2vec2" \
  --epochs=50 \
  --batch_size=8
```

**After (JSON):**
```json
{
  "dataset_name": "my/dataset",
  "model_name_or_path": "facebook/wav2vec2",
  "num_train_epochs": 50,
  "per_device_train_batch_size": 8
}
```

```bash
python src/cli.py train --config=my_config.json
```

---

## Advanced: Programmatic Usage

### Python Script

```python
from src.optimized_training import train_asr_model

# With JSON config
train_asr_model(config_file="configs/my_config.json")

# With parameters (legacy)
train_asr_model(
    dataset_repo="my/dataset",
    base_model="facebook/wav2vec2-xls-r-1b",
    output_name="my-output",
    num_epochs=50
)

# Hybrid (config + overrides)
train_asr_model(
    config_file="configs/base.json",
    num_epochs=100  # Override config value
)
```

---

## See Also

- [`config_template.json`](config_template.json) - Complete parameter reference
- [`example_config.json`](example_config.json) - Real-world example
- [`../DATASET_FORMAT.md`](../DATASET_FORMAT.md) - Dataset format guide
- [`../src/config_loader.py`](../src/config_loader.py) - Configuration loader code

---

**Happy Training! 🚀**
