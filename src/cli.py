#!/usr/bin/env python3
"""
🚀 ASR Training System - Professional CLI
=========================================
Command Line Interface for high-architecture ASR training.

Usage:
    python src/cli.py train --model="facebook/wav2vec2-large" --dataset="mozilla/common_voice"
    python src/cli.py train --help
"""

import fire
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from optimized_training import train_asr_model
from asr_config_manager import ASRConfigManager

class ASRCLI:
    """
    High-Architecture ASR Training System CLI
    """
    
    def train(
        self,
        config: str = None,
        dataset_repo: str = None,
        model_name: str = None,
        output_dir: str = None,
        hf_username: str = None,
        hf_token: str = None,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        safety_margin: float = 0.85,
        use_deepspeed: bool = False,
        push_to_hub: bool = None,
        resume_from_checkpoint: str = None
    ):
        """
        Start ASR training with auto-configuration.
        
        Args:
            config: Path to JSON configuration file (takes precedence over other params)
            dataset_repo: HuggingFace dataset repository ID
            model_name: Base model to fine-tune
            output_dir: Directory to save results
            hf_username: Your HuggingFace username (optional, tries to infer)
            hf_token: HuggingFace API token for private datasets/models
            epochs: Number of training epochs
            batch_size: Target effective batch size
            learning_rate: Learning rate
            safety_margin: Memory usage limit (0.1-1.0)
            use_deepspeed: Enable DeepSpeed optimization (ZeRO-2)
            push_to_hub: Upload model to HuggingFace Hub
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        print(f"\n🚀 Starting ASR Training System")
        
        if config:
            print(f"   Configuration: {config}")
        else:
            print(f"   Model: {model_name or 'Not specified'}")
            print(f"   Dataset: {dataset_repo or 'Not specified'}")
        
        print(f"   DeepSpeed: {'Enabled ✅' if use_deepspeed else 'Disabled ❌'}")
        
        # Call the main training function
        train_asr_model(
            config_file=config,
            dataset_repo=dataset_repo,
            base_model=model_name,
            output_name=output_dir,
            hf_username=hf_username,
            hf_token=hf_token,
            num_epochs=epochs,
            target_batch_size=batch_size,
            learning_rate=learning_rate,
            safety_margin=safety_margin,
            use_deepspeed=use_deepspeed,
            push_to_hub=push_to_hub,
            resume_from_checkpoint=resume_from_checkpoint
        )

    def profile(self, dataset_repo: str, model_name: str = "facebook/wav2vec2-xls-r-1b"):
        """
        Run profiling only (Hardware + Dataset + Model) without training.
        Useful for checking what config would be generated.
        """
        print("\n🔍 Running System Profiler...")
        
        # Initialize manager (we need to mock dataset/model loading for full profile, 
        # but here we'll just show hardware)
        manager = ASRConfigManager()
        hw_profile = manager.profile_hardware(safety_margin=0.85)
        
        print("\nHardware Profile:")
        print(f"  GPU: {hw_profile.gpu_name}")
        print(f"  VRAM: {hw_profile.gpu_total_gb:.2f} GB (Available: {hw_profile.gpu_available_gb:.2f} GB)")
        print(f"  RAM: {hw_profile.cpu_total_gb:.2f} GB")
        
        print("\nTo see full config, run training!")

if __name__ == '__main__':
    fire.Fire(ASRCLI)
