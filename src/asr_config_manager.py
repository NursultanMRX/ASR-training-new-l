import torch
import psutil
import math
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any
from transformers import PreTrainedModel, TrainingArguments
from datasets import Dataset

@dataclass
class HardwareProfile:
    gpu_name: str
    gpu_total_gb: float
    gpu_available_gb: float
    cpu_total_gb: float
    cpu_available_gb: float
    cuda_version: str

@dataclass
class DatasetProfile:
    num_samples: int
    avg_duration_sec: float
    max_duration_sec: float
    min_duration_sec: float
    total_hours: float
    estimated_size_gb: float

@dataclass
class ModelProfile:
    total_params: int
    trainable_params: int
    model_size_gb: float
    hidden_size: int
    num_layers: int

@dataclass
class TrainingConfig:
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    learning_rate: float
    num_train_epochs: int
    warmup_steps: int
    eval_steps: int
    save_steps: int
    logging_steps: int
    fp16: bool
    gradient_checkpointing: bool
    dataloader_num_workers: int
    max_audio_duration_seconds: float
    use_streaming: bool
    cache_dataset: bool
    deepspeed_config: Optional[Dict[str, Any]] = None

class ASRConfigManager:
    """
    Intelligent Configuration Manager for ASR Training.
    Automatically profiles hardware, dataset, and model to generate optimal settings.
    """
    
    def __init__(self):
        self.hardware = None
        self.dataset_profile = None
        self.model_profile = None

    def profile_hardware(self, safety_margin: float = 0.85) -> HardwareProfile:
        """Detects GPU and CPU capabilities."""
        # CPU
        vm = psutil.virtual_memory()
        cpu_total = vm.total / 1e9
        cpu_avail = vm.available / 1e9 * safety_margin

        # GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_total = gpu_props.total_memory / 1e9
            # Get actual free memory
            free_mem, _ = torch.cuda.mem_get_info()
            gpu_avail = (free_mem / 1e9) * safety_margin
            cuda_ver = torch.version.cuda
        else:
            gpu_name = "CPU Only"
            gpu_total = 0
            gpu_avail = 0
            cuda_ver = "N/A"

        self.hardware = HardwareProfile(
            gpu_name=gpu_name,
            gpu_total_gb=gpu_total,
            gpu_available_gb=gpu_avail,
            cpu_total_gb=cpu_total,
            cpu_available_gb=cpu_avail,
            cuda_version=cuda_ver
        )
        return self.hardware

    def profile_dataset(self, dataset: Dataset, sample_size: int = 200) -> DatasetProfile:
        """Analyzes dataset statistics by sampling."""
        if len(dataset) < sample_size:
            sample_size = len(dataset)
            
        # Sample random indices
        import random
        indices = random.sample(range(len(dataset)), sample_size)
        
        durations = []
        for idx in indices:
            item = dataset[idx]
            # Assuming 'audio' column has 'array' and 'sampling_rate'
            if 'audio' in item:
                dur = len(item['audio']['array']) / item['audio']['sampling_rate']
                durations.append(dur)
        
        if not durations:
            durations = [5.0] # Fallback

        avg_dur = sum(durations) / len(durations)
        max_dur = max(durations)
        min_dur = min(durations)
        
        total_samples = len(dataset)
        total_hours = (total_samples * avg_dur) / 3600
        
        # Estimate size: duration * sample_rate * 4 bytes (float32)
        est_size_gb = (total_hours * 3600 * 16000 * 4) / 1e9

        self.dataset_profile = DatasetProfile(
            num_samples=total_samples,
            avg_duration_sec=avg_dur,
            max_duration_sec=max_dur,
            min_duration_sec=min_dur,
            total_hours=total_hours,
            estimated_size_gb=est_size_gb
        )
        return self.dataset_profile

    def profile_model(self, model: PreTrainedModel) -> ModelProfile:
        """Inspects model architecture and parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory footprint (params + gradients + optimizer states)
        # 4 bytes per param + 4 bytes per grad + 8 bytes per optimizer state (AdamW)
        # = ~16 bytes per trainable param + 4 bytes per frozen param
        mem_gb = ((trainable_params * 16) + ((total_params - trainable_params) * 4)) / 1e9
        
        self.model_profile = ModelProfile(
            total_params=total_params,
            trainable_params=trainable_params,
            model_size_gb=mem_gb,
            hidden_size=model.config.hidden_size if hasattr(model.config, 'hidden_size') else 768,
            num_layers=model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 12
        )
        return self.model_profile

    def _generate_deepspeed_config(self, batch_size: int, accum_steps: int) -> Dict[str, Any]:
        """Generates DeepSpeed ZeRO-2 Configuration."""
        return {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "betas": "auto",
                    "eps": "auto",
                    "weight_decay": "auto",
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": "auto",
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto",
                }
            },
            "fp16": {
                "enabled": True,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            }
        }

    def generate_config(self, 
                       target_batch_size: int = 32, 
                       num_epochs: int = 20,
                       learning_rate: float = 3e-4,
                       use_deepspeed: bool = False) -> TrainingConfig:
        """
        Generates the optimal training configuration.
        """
        if not all([self.hardware, self.dataset_profile, self.model_profile]):
            raise ValueError("Must profile hardware, dataset, and model first!")

        # 1. Calculate Memory per Sample
        # Audio: duration * 16k * 4 bytes
        audio_mem = self.dataset_profile.avg_duration_sec * 16000 * 4 / 1e9
        # Activation memory (heuristic: 10% of model size per sample for forward/backward)
        activation_mem = self.model_profile.model_size_gb * 0.1
        
        total_mem_per_sample = audio_mem + activation_mem
        
        # 2. Calculate Available Memory
        # Reserve memory for model weights
        available_mem = self.hardware.gpu_available_gb - self.model_profile.model_size_gb
        
        if available_mem <= 0:
            print("⚠️ Warning: Model might be too large for GPU even with batch size 1!")
            available_mem = 2.0 # Try anyway with swap/optimizations
            
        # 3. Calculate Batch Size
        max_batch = int(available_mem / total_mem_per_sample)
        if max_batch < 1: max_batch = 1
        
        # Cap batch size to avoid instability
        batch_size = min(max_batch, 64)
        
        # DeepSpeed allows larger batches due to ZeRO
        if use_deepspeed:
            batch_size = int(batch_size * 1.5) # Conservative estimate
            
        # 4. Calculate Gradient Accumulation
        # effective = batch * accum * num_gpus (assuming 1 GPU for now)
        grad_accum = max(1, target_batch_size // batch_size)
        
        effective_batch = batch_size * grad_accum
        
        # 5. Determine Optimizations
        use_fp16 = self.hardware.gpu_total_gb >= 8.0
        use_checkpointing = self.model_profile.total_params > 100_000_000
        
        # Data loading
        num_workers = 2 if self.hardware.cpu_available_gb > 16 else 0
        use_streaming = self.dataset_profile.estimated_size_gb > 50
        cache_dataset = self.dataset_profile.estimated_size_gb < 10 and not use_streaming
        
        # Steps
        steps_per_epoch = self.dataset_profile.num_samples // effective_batch
        eval_steps = max(1, steps_per_epoch // 4) # Eval 4 times per epoch
        
        # DeepSpeed Config
        ds_config = self._generate_deepspeed_config(batch_size, grad_accum) if use_deepspeed else None

        return TrainingConfig(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            effective_batch_size=effective_batch,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            warmup_steps=int(steps_per_epoch * num_epochs * 0.1), # 10% warmup
            eval_steps=eval_steps,
            save_steps=eval_steps,
            logging_steps=max(1, eval_steps // 4),
            fp16=use_fp16,
            gradient_checkpointing=use_checkpointing,
            dataloader_num_workers=num_workers,
            max_audio_duration_seconds=min(30.0, self.dataset_profile.max_duration_sec),
            use_streaming=use_streaming,
            cache_dataset=cache_dataset,
            deepspeed_config=ds_config
        )

    def save_config(self, config: TrainingConfig, path: str):
        """Saves config to JSON."""
        with open(path, 'w') as f:
            json.dump(asdict(config), f, indent=2)

    def load_config(self, path: str) -> TrainingConfig:
        """Loads config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return TrainingConfig(**data)

def create_optimal_config(dataset, model, model_name: str, 
                         target_batch_size: int = 32,
                         num_epochs: int = 20,
                         learning_rate: float = 3e-4,
                         safety_margin: float = 0.85,
                         use_deepspeed: bool = False):
    """Convenience wrapper for the entire process."""
    
    manager = ASRConfigManager()
    
    print("🔍 Profiling hardware...")
    manager.profile_hardware(safety_margin)
    
    print("🔍 Analyzing dataset...")
    manager.profile_dataset(dataset)
    
    print("🔍 Analyzing model...")
    manager.profile_model(model)
    
    print("⚙️  Generating optimal configuration...")
    config = manager.generate_config(
        target_batch_size=target_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        use_deepspeed=use_deepspeed
    )
    
    return config, manager
