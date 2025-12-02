# Quick Start: Using JSON Configuration

## TL;DR - Get Training Now!

```bash
# 1. Edit the example config with your HF token
nano configs/example_config.json

# 2. Run training
python src/cli.py train --config=configs/example_config.json
```

That's it! ✅

---

## Your Dataset Format

Make sure your dataset looks like this:

```python
# Example dataset row
{
    "file_name": "audio_001.wav",      # Your audio files
    "text": "Hello, how are you?",      # Original text
    "cleaned_text": "hello how are you", # Cleaned version (optional)
    "speaker_name": "Speaker_1"         # Speaker ID
}
```

---

## Configuration Options

### Option 1: Use JSON Config (Recommended)

**configs/my_training.json:**
```json
{
  "model_name_or_path": "./mms-tts-kaa-with-discriminator",
  "dataset_name": "./my_local_dataset",
  "output_dir": "./output",
  
  "audio_column_name": "file_name",
  "text_column_name": "text",
  "speaker_id_column_name": "speaker_name",
  "filter_on_speaker_id": "Speaker_1",
  
  "num_train_epochs": 150,
  "per_device_train_batch_size": 4,
  "learning_rate": 2e-5,
  
  "weight_disc": 3.0,
  "weight_mel": 35.0,
  
  "fp16": true,
  "hf_token": "YOUR_TOKEN_HERE"
}
```

Run with:
```bash
python src/cli.py train --config=configs/my_training.json
```

### Option 2: CLI Parameters

```bash
python src/cli.py train \
  --dataset_repo="nickoo004/my-dataset" \
  --model_name="facebook/wav2vec2-xls-r-1b" \
  --output_dir="./output" \
  --epochs=50
```

### Option 3: Hybrid (Config + Overrides)

```bash
python src/cli.py train \
  --config=configs/base.json \
  --epochs=200 \
  --learning_rate=1e-5
```

---

## Key Parameters Explained

| Parameter | What It Does | Example |
|-----------|--------------|---------|
| `filter_on_speaker_id` | Train on specific speaker only | `"Speaker_1"` |
| `max_duration_in_seconds` | Filter out long audio | `20.0` |
| `min_duration_in_seconds` | Filter out short audio | `1.0` |
| `weight_disc` | Discriminator loss weight (TTS) | `3.0` |
| `weight_mel` | Mel-spectrogram loss weight | `35.0` |
| `hf_token` | HuggingFace API token | `"hf_xxx"` |

---

## Common Use Cases

### 1. Single Speaker TTS Training
```json
{
  "filter_on_speaker_id": "Speaker_1",
  "num_train_epochs": 150,
  "per_device_train_batch_size": 4
}
```

### 2. Multi-Speaker (No Filtering)
```json
{
  "filter_on_speaker_id": null,
  "num_train_epochs": 100
}
```

### 3. Private Dataset
```json
{
  "dataset_name": "myusername/private-data",
  "hf_token": "hf_xxxxxxxxxxxxx"
}
```

---

## Troubleshooting

**Q: Config file not found?**
```bash
# Use absolute or relative path
python src/cli.py train --config=./configs/my_config.json
```

**Q: Invalid JSON?**
- Check for missing commas
- Validate at [jsonlint.com](https://jsonlint.com)

**Q: Column not found error?**
- Check your dataset has the columns specified in `audio_column_name`, `text_column_name`, etc.
- Or use the auto-detection (just don't specify column names)

---

## Documentation

- Full config reference: [`configs/README.md`](configs/README.md)
- Dataset format guide: [`DATASET_FORMAT.md`](DATASET_FORMAT.md)
- All parameters: [`configs/config_template.json`](configs/config_template.json)

---

**Happy Training! 🚀**
