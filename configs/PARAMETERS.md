# Configuration Template Reference

This file documents all available parameters in `config_template.json`.

## Organization

Parameters are organized into these categories:

### MODEL & DATASET
- `model_name_or_path`: Path or HF model ID
- `dataset_name`: Path or HF dataset ID  
- `output_dir`: Output directory
- `overwrite_output_dir`: Overwrite existing output

### HUGGINGFACE HUB
- `push_to_hub`: Upload to HF Hub
- `hub_model_id`: HF Hub model ID
- `hf_token`: HF API token

### DATASET COLUMN MAPPING
- `audio_column_name`: Audio column name (default: "audio")
- `text_column_name`: Text column name (default: "text")
- `cleaned_text_column_name`: Cleaned text column
- `speaker_id_column_name`: Speaker ID column
- `filter_on_speaker_id`: Filter for specific speaker

### AUDIO FILTERING
- `max_duration_in_seconds`: Max audio duration (default: 20.0)
- `min_duration_in_seconds`: Min audio duration (default: 1.0)

### TRAINING HYPERPARAMETERS
- `num_train_epochs`: Number of epochs (default: 20)
- `per_device_train_batch_size`: Batch size per device (default: 4)
- `per_device_eval_batch_size`: Eval batch size (default: 4)
- `learning_rate`: Learning rate (default: 0.0003)
- `warmup_ratio`: Warmup ratio (default: 0.1)
- `warmup_steps`: Warmup steps (alternative to ratio)
- `gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `gradient_checkpointing`: Enable checkpointing (default: true)
- `group_by_length`: Group samples by length

### EVALUATION & CHECKPOINTING
- `do_train`: Enable training (default: true)
- `do_eval`: Enable evaluation (default: true)
- `evaluation_strategy`: "steps" or "epoch"
- `eval_steps`: Evaluate every N steps (default: 500)
- `save_steps`: Save every N steps (default: 500)
- `save_total_limit`: Max checkpoints to keep (default: 3)
- `logging_steps`: Log every N steps (default: 100)
- `load_best_model_at_end`: Load best checkpoint at end
- `metric_for_best_model`: Metric to track (default: "wer")

### LOSS WEIGHTS (for TTS/VITS models)
- `weight_disc`: Discriminator weight (default: 3.0)
- `weight_fmaps`: Feature maps weight (default: 1.0)
- `weight_gen`: Generator weight (default: 1.0)
- `weight_kl`: KL divergence weight (default: 1.5)
- `weight_duration`: Duration weight (default: 1.0)
- `weight_mel`: Mel-spectrogram weight (default: 35.0)

### OPTIMIZATION
- `fp16`: Use FP16 mixed precision (default: true)
- `fp16_full_eval`: FP16 during eval
- `bf16`: Use BF16 instead of FP16
- `optim`: Optimizer type (default: "adamw_torch")
- `weight_decay`: Weight decay (default: 0.01)
- `adam_beta1`: Adam beta1 (default: 0.9)
- `adam_beta2`: Adam beta2 (default: 0.999)
- `adam_epsilon`: Adam epsilon (default: 0.00000001)
- `max_grad_norm`: Gradient clipping (default: 1.0)

### DATA LOADING
- `dataloader_num_workers`: Number of workers (default: 4)
- `dataloader_pin_memory`: Pin memory (default: true)
- `streaming`: Use streaming mode

### SYSTEM
- `seed`: Random seed (default: 42)
- `local_rank`: Local rank for distributed training
- `ddp_find_unused_parameters`: DDP unused params
- `ignore_data_skip`: Ignore data skip

### DEEPSPEED (optional)
- `use_deepspeed`: Enable DeepSpeed (default: false)
- `deepspeed_config`: Path to DeepSpeed config

### ADVANCED
- `resume_from_checkpoint`: Resume from checkpoint path
- `skip_health_check`: Skip pre-flight checks (default: false)
- `safety_margin`: Memory safety margin (default: 0.85)

## Usage

See `README.md` for complete usage examples.
