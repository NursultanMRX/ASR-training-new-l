"""
Configuration Loader for ASR/TTS Training
==========================================
Handles loading, validation, and merging of JSON configuration files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class TrainingConfig:
    """Complete training configuration with all parameters."""
    
    # Model & Dataset
    model_name_or_path: str = "facebook/wav2vec2-xls-r-1b"
    dataset_name: str = None
    output_dir: str = "./output"
    overwrite_output_dir: bool = True
    
    # HuggingFace Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hf_token: Optional[str] = None
    
    # Dataset Column Mapping
    audio_column_name: str = "audio"
    text_column_name: str = "text"
    speaker_id_column_name: Optional[str] = None
    filter_on_speaker_id: Optional[str] = None
    
    # Audio Filtering
    max_duration_in_seconds: float = 20.0
    min_duration_in_seconds: float = 1.0
    
    # Training Hyperparameters
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.1
    warmup_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    group_by_length: bool = False
    
    # Evaluation & Checkpointing
    do_train: bool = True
    do_eval: bool = True
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 100
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    
    # Optimization
    fp16: bool = True
    fp16_full_eval: bool = False
    bf16: bool = False
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Data Loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    streaming: bool = False
    
    # System
    seed: int = 42
    local_rank: int = -1
    ddp_find_unused_parameters: bool = False
    ignore_data_skip: bool = False
    
    # DeepSpeed
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    
    # Advanced
    resume_from_checkpoint: Optional[str] = None
    skip_health_check: bool = False
    safety_margin: float = 0.85
    
    # Backward compatibility
    target_batch_size: int = 32


def load_config_from_json(config_path: str) -> TrainingConfig:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        TrainingConfig object with loaded parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"📄 Loading configuration from: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # Filter out comment keys (starting with //)
    config_dict = {k: v for k, v in config_dict.items() if not k.startswith("//")}
    
    # Create config object
    config = TrainingConfig()
    
    # Update with values from JSON
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"⚠️ Unknown config parameter: {key} (ignoring)")
    
    return config


def merge_configs(json_config: TrainingConfig, cli_args: Dict[str, Any]) -> TrainingConfig:
    """
    Merge JSON config with CLI arguments. CLI args take precedence.
    
    Args:
        json_config: Configuration loaded from JSON
        cli_args: Dictionary of CLI arguments
        
    Returns:
        Merged TrainingConfig
    """
    merged = json_config
    
    for key, value in cli_args.items():
        if value is not None and hasattr(merged, key):
            setattr(merged, key, value)
    
    return merged


def save_config(config: TrainingConfig, output_path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: TrainingConfig to save
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = asdict(config)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Configuration saved to: {output_path}")


def print_config(config: TrainingConfig):
    """Print configuration in a readable format."""
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION".center(80))
    print("="*80)
    
    sections = {
        "Model & Dataset": ["model_name_or_path", "dataset_name", "output_dir"],
        "HuggingFace Hub": ["push_to_hub", "hub_model_id", "hf_token"],
        "Dataset Columns": ["audio_column_name", "text_column_name", 
                           "speaker_id_column_name", "filter_on_speaker_id"],
        "Audio Filtering": ["max_duration_in_seconds", "min_duration_in_seconds"],
        "Training": ["num_train_epochs", "per_device_train_batch_size", "learning_rate", "warmup_ratio"],
        "Optimization": ["fp16", "gradient_checkpointing", "gradient_accumulation_steps", "optim"]
    }
    
    for section_name, keys in sections.items():
        print(f"\n{section_name}:")
        for key in keys:
            value = getattr(config, key, None)
            if value is not None and value != "":
                # Mask token if present
                if key == "hf_token" and value:
                    value = "*" * 8 + value[-4:] if len(value) > 4 else "****"
                print(f"  {key}: {value}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Test loading example config
    example_path = Path(__file__).parent.parent / "configs" / "example_config.json"
    
    if example_path.exists():
        config = load_config_from_json(str(example_path))
        print_config(config)
    else:
        print(f"Example config not found at: {example_path}")
