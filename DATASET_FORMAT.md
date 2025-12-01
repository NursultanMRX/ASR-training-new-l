# 📊 Dataset Format Guide for ASR Training

## Overview

This guide explains the exact dataset format required to use the ASR training system. Follow this format and your training will work perfectly!

---

## ✅ Required Dataset Structure

### Minimum Required Columns

Your dataset **MUST** have these columns:

| Column Name | Type | Description | Example |
|-------------|------|-------------|---------|
| `audio` | Audio | Audio file (WAV, MP3, FLAC) | `{"path": "audio_1.wav", "array": [...], "sampling_rate": 16000}` |
| `sentence` | String | Text transcription | `"Hello world"` |

**That's it!** Just 2 columns minimum.

---

## 📁 Dataset Format Options

### Option 1: HuggingFace Dataset (Recommended ✅)

**Structure:**
```
your-username/your-dataset
├── train/
│   ├── audio_1.wav
│   ├── audio_2.wav
│   └── ...
├── test/ (optional)
│   ├── audio_1.wav
│   └── ...
└── metadata.csv or data.parquet
```

**Metadata File (CSV example):**
```csv
audio,sentence
train/audio_1.wav,"This is the first sentence"
train/audio_2.wav,"This is the second sentence"
train/audio_3.wav,"This is the third sentence"
```

**Metadata File (Parquet/JSON):**
```json
[
  {"audio": "train/audio_1.wav", "sentence": "This is the first sentence"},
  {"audio": "train/audio_2.wav", "sentence": "This is the second sentence"},
  {"audio": "train/audio_3.wav", "sentence": "This is the third sentence"}
]
```

---

### Option 2: Local Directory

**Structure:**
```
my_dataset/
├── train/
│   ├── audio/
│   │   ├── clip_1.wav
│   │   ├── clip_2.wav
│   │   └── ...
│   └── transcripts.csv
└── test/ (optional)
    ├── audio/
    └── transcripts.csv
```

**transcripts.csv:**
```csv
file_name,text
clip_1.wav,Hello how are you
clip_2.wav,I am fine thank you
clip_3.wav,What is your name
```

---

## 🎵 Audio Requirements

### Supported Formats
- ✅ **WAV** (recommended)
- ✅ **MP3**
- ✅ **FLAC**
- ✅ **OGG**
- ✅ **M4A**

### Audio Specifications

| Property | Requirement | Recommended |
|----------|-------------|-------------|
| **Sampling Rate** | Any (auto-resampled to 16kHz) | 16000 Hz |
| **Channels** | Mono or Stereo (auto-converted to mono) | Mono |
| **Bit Depth** | 16-bit or 24-bit | 16-bit |
| **Duration** | 0.5s - 30s per clip | 3-15 seconds |
| **Format** | WAV, MP3, FLAC, etc. | WAV (uncompressed) |

### Audio Quality Tips
- ✅ Clear speech (no background noise if possible)
- ✅ Single speaker per clip
- ✅ Natural speaking pace
- ❌ Avoid: Music, multiple speakers, heavy noise

---

## 📝 Text Requirements

### Text Format Rules

1. **Encoding:** UTF-8 (supports all languages/scripts)
2. **Case:** Any (lowercase, uppercase, mixed)
3. **Punctuation:** Optional (can include or exclude)
4. **Numbers:** Spell out or use digits (be consistent)
5. **Special Characters:** Allowed (will be auto-processed)

### Supported Languages/Scripts
- ✅ Latin (English, Spanish, French, etc.)
- ✅ Cyrillic (Russian, Karakalpak, etc.)
- ✅ Arabic
- ✅ Chinese (Simplified/Traditional)
- ✅ Japanese (Hiragana, Katakana, Kanji)
- ✅ Any Unicode text!

### Text Examples

**Good:**
```
"Hello, how are you today?"
"Привет, как дела?"
"Сәлем, қалың қалай?"
"مرحبا كيف حالك"
```

**Also Good (without punctuation):**
```
"hello how are you today"
"привет как дела"
"сәлем қалың қалай"
```

---

## 📊 Optional Columns (Enhanced Features)

You can include additional columns for better organization:

| Column | Type | Purpose | Example |
|--------|------|---------|---------|
| `duration` | Float | Audio length in seconds | `3.5` |
| `speaker_id` | String | Speaker identifier | `"speaker_001"` |
| `gender` | String | Speaker gender | `"M"` or `"F"` |
| `age` | Integer | Speaker age | `25` |
| `accent` | String | Accent/dialect | `"northern"` |
| `domain` | String | Topic/domain | `"conversational"` |

**These are optional** - the system only needs `audio` and `sentence`!

---

## 🔨 Creating Your Own Dataset

### Method 1: Upload to HuggingFace (Recommended)

**Step 1:** Prepare your files
```
my_data/
├── audio/
│   ├── clip_001.wav
│   ├── clip_002.wav
│   └── ...
└── metadata.csv
```

**Step 2:** Create metadata.csv
```python
import pandas as pd

data = {
    'audio': ['audio/clip_001.wav', 'audio/clip_002.wav', ...],
    'sentence': ['First sentence', 'Second sentence', ...]
}
df = pd.DataFrame(data)
df.to_csv('metadata.csv', index=False)
```

**Step 3:** Upload to HuggingFace
```python
from datasets import Dataset, Audio
import pandas as pd

# Load your data
df = pd.read_csv('metadata.csv')

# Create dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio", Audio())

# Push to HuggingFace
dataset.push_to_hub("your-username/your-dataset-name")
```

**Step 4:** Use in training!
```python
train_asr_model(
    dataset_repo="your-username/your-dataset-name",
    ...
)
```

---

### Method 2: Use Local Files

**Step 1:** Organize files
```
dataset/
├── wavs/
│   └── *.wav
└── metadata.csv
```

**Step 2:** Load in training script
```python
from datasets import load_dataset

dataset = load_dataset("csv", data_files="metadata.csv")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Then use with training
train_asr_model(
    dataset_repo=dataset,  # Pass directly
    ...
)
```

---

## 📋 Example Datasets

### 1. Common Voice Format
```json
{
  "audio": {"path": "common_voice_en_123.mp3", "array": [...], "sampling_rate": 48000},
  "sentence": "The quick brown fox jumps over the lazy dog",
  "accent": "us",
  "age": "twenties",
  "gender": "male"
}
```

### 2. LibriSpeech Format
```json
{
  "audio": {"path": "1089-134686-0001.flac", "array": [...], "sampling_rate": 16000},
  "sentence": "HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS"
}
```

### 3. Simple Custom Format
```json
{
  "audio": {"path": "my_audio.wav", "array": [...], "sampling_rate": 16000},
  "sentence": "hello world this is a test"
}
```

---

## ✅ Quick Validation Checklist

Before training, verify:

- [ ] Dataset has `audio` column (type: Audio)
- [ ] Dataset has `sentence` column (type: String)
- [ ] Audio files are accessible (can be loaded)
- [ ] Audio files are 0.5s - 30s duration
- [ ] Text is UTF-8 encoded
- [ ] Text matches audio content
- [ ] Dataset has at least 100 samples (more is better!)
- [ ] Train/test split exists (or will auto-split)

---

## 🔍 Testing Your Dataset

**Quick test script:**

```python
from datasets import load_dataset, Audio

# Load your dataset
dataset = load_dataset("your-username/your-dataset")

# Check structure
print(f"Columns: {dataset['train'].column_names}")
print(f"Number of samples: {len(dataset['train'])}")

# Check first sample
sample = dataset['train'][0]
print(f"\nFirst sample:")
print(f"  Audio: {sample['audio']['path']}")
print(f"  Text: {sample['sentence']}")
print(f"  Duration: {len(sample['audio']['array']) / sample['audio']['sampling_rate']:.2f}s")

# Test loading audio
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
sample = dataset['train'][0]
print(f"  Resampled to: {sample['audio']['sampling_rate']} Hz")
print(f"  Array shape: {len(sample['audio']['array'])}")

print("\n✅ Dataset looks good!")
```

---

## 🐛 Common Dataset Issues & Fixes

### Issue 1: "Column 'audio' not found"
**Fix:** Rename your audio column
```python
dataset = dataset.rename_column("audio_file", "audio")
dataset = dataset.rename_column("path", "audio")
```

### Issue 2: "Column 'sentence' not found"
**Fix:** Rename your text column
```python
dataset = dataset.rename_column("text", "sentence")
dataset = dataset.rename_column("transcription", "sentence")
dataset = dataset.rename_column("transcript", "sentence")
```

### Issue 3: Audio files not loading
**Fix:** Check file paths are relative to dataset root
```python
# If paths are absolute, make them relative
dataset = dataset.map(lambda x: {"audio": x["audio"].replace("/full/path/", "")})
```

### Issue 4: Wrong sampling rate
**Fix:** Cast to Audio with target rate
```python
from datasets import Audio
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

### Issue 5: Empty text fields
**Fix:** Filter out empty samples
```python
dataset = dataset.filter(lambda x: len(x["sentence"].strip()) > 0)
```

---

## 📏 Dataset Size Recommendations

| Dataset Size | Training Time (L4 GPU) | Expected WER | Use Case |
|--------------|------------------------|--------------|----------|
| **100-500 samples** | ~30 min | High (>50%) | Quick test |
| **1,000-5,000** | ~2-4 hours | Medium (20-40%) | Small vocabulary |
| **10,000-50,000** | ~8-24 hours | Good (10-20%) | Production (small) |
| **50,000-100,000** | ~1-3 days | Very Good (<10%) | Production (medium) |
| **100,000+** | ~3-7 days | Excellent (<5%) | Production (large) |

**More data = Better results!**

---

## 💡 Pro Tips

1. **Balance your data:**
   - Mix of male/female speakers
   - Various speaking speeds
   - Different accents/dialects

2. **Clean your audio:**
   - Remove silence at start/end
   - Normalize volume levels
   - Filter background noise

3. **Clean your text:**
   - Fix typos
   - Consistent formatting
   - Remove special annotations (unless needed)

4. **Split properly:**
   - 80-90% train
   - 10-20% test
   - Ensure test speakers not in train

5. **Augment if needed:**
   - Speed perturbation
   - Noise injection
   - SpecAugment (done automatically by Wav2Vec2)

---

## 🔗 Ready-to-Use Example Datasets

These datasets work perfectly with the system:

1. **Common Voice:** `mozilla-foundation/common_voice_11_0`
2. **LibriSpeech:** `openslr/librispeech_asr`
3. **TIMIT:** `timit_asr`
4. **VoxPopuli:** `facebook/voxpopuli`
5. **FLEURS:** `google/fleurs`

**Just change the `dataset_repo` parameter!**

---

## 📞 Need Help?

- **Can't find your dataset format?** Check the [HuggingFace datasets docs](https://huggingface.co/docs/datasets)
- **Dataset too large?** Use `streaming=True` in load_dataset
- **Custom format?** Create a dataset loading script

---

**Your dataset is ready when it has `audio` + `sentence` columns!** 🎉

For more help: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
