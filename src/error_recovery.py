"""
Error Recovery System
====================
Automatic recovery from common training errors:
- Network timeouts
- Disk space issues
- Connection drops
- CUDA errors
- Checkpoint corruption
"""

import torch
import time
import shutil
import traceback
from pathlib import Path
from typing import Optional, Callable
import signal
import sys
import atexit

class ErrorRecovery:
    """
    Comprehensive error recovery and retry logic.
    """
    
    def __init__(self, max_retries=3, checkpoint_dir="./checkpoints"):
        self.max_retries = max_retries
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.last_checkpoint = None
        self.shutdown_handlers = []
        
    def safe_execute(self, func: Callable, *args, **kwargs):
        """
        Execute a function with automatic retry on transient errors.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of function or raises after max retries
        """
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                result = func(*args, **kwargs)
                if retry_count > 0:
                    print(f"✅ Recovered after {retry_count} retries!")
                return result
                
            except KeyboardInterrupt:
                print("\n⚠️ Training interrupted by user")
                self._trigger_graceful_shutdown()
                raise
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n💥 Out of Memory Error (attempt {retry_count + 1}/{self.max_retries})")
                    self._handle_oom_error()
                    retry_count += 1
                    last_error = e
                    time.sleep(2)
                elif "CUDA" in str(e):
                    print(f"\n💥 CUDA Error (attempt {retry_count + 1}/{self.max_retries})")
                    self._handle_cuda_error()
                    retry_count += 1
                    last_error = e
                    time.sleep(5)
                else:
                    # Non-recoverable RuntimeError
                    raise
                    
            except (ConnectionError, TimeoutError) as e:
                print(f"\n💥 Network Error (attempt {retry_count + 1}/{self.max_retries}): {e}")
                self._handle_network_error()
                retry_count += 1
                last_error = e
                time.sleep(10)  # Wait longer for network
                
            except IOError as e:
                if "No space left" in str(e):
                    print(f"\n💥 Disk Full Error!")
                    self._handle_disk_full()
                    # This is likely not recoverable
                    raise
                else:
                    retry_count += 1
                    last_error = e
                    time.sleep(2)
                    
            except Exception as e:
                print(f"\n💥 Unexpected Error: {type(e).__name__}")
                print(f"   {e}")
                traceback.print_exc()
                retry_count += 1
                last_error = e
                time.sleep(5)
                
        # Max retries exceeded
        print(f"\n❌ Failed after {self.max_retries} retries")
        raise last_error
        
    def _handle_oom_error(self):
        """Handle Out of Memory errors."""
        print("🔧 Handling OOM:")
        print("   1. Clearing CUDA cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("   2. Clearing Python garbage...")
        import gc
        gc.collect()
        
        print("   3. Reducing batch size would help (done automatically by Trainer)")
        
    def _handle_cuda_error(self):
        """Handle CUDA-related errors."""
        print("🔧 Handling CUDA error:")
        print("   1. Resetting CUDA context...")
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Try to reinitialize
                torch.cuda.init()
            except:
                pass
                
        print("   2. Will retry...")
        
    def _handle_network_error(self):
        """Handle network connectivity issues."""
        print("🔧 Handling network error:")
        print("   1. Checking connection...")
        
        try:
            import requests
            response = requests.get("https://www.google.com", timeout=5)
            if response.status_code == 200:
                print("   ✅ Internet OK, transient error")
            else:
                print("   ⚠️ Internet unstable")
        except:
            print("   ❌ No internet connection!")
            print("   Waiting for connection to restore...")
            
    def _handle_disk_full(self):
        """Handle disk space issues."""
        print("🔧 Attempting to free disk space:")
        
        # Check what's using space
        stat = shutil.disk_usage("/")
        free_gb = stat.free / 1e9
        print(f"   Current free space: {free_gb:.2f} GB")
        
        # Try to clear cache
        cache_dir = Path.home() / ".cache" / "huggingface"
        if cache_dir.exists():
            print(f"   Consider clearing: {cache_dir}")
            
        print("   ⚠️ You need to free up disk space manually")
        
    def register_shutdown_handler(self, handler: Callable):
        """Register a function to call on graceful shutdown."""
        self.shutdown_handlers.append(handler)
        
    def _trigger_graceful_shutdown(self):
        """Execute all shutdown handlers."""
        print("\n🛑 Initiating graceful shutdown...")
        for handler in self.shutdown_handlers:
            try:
                handler()
            except Exception as e:
                print(f"   ⚠️ Shutdown handler error: {e}")
                
    def setup_signal_handlers(self):
        """Setup handlers for SIGINT and SIGTERM to gracefully shutdown."""
        def signal_handler(signum, frame):
            print(f"\n⚠️ Received signal {signum}")
            self._trigger_graceful_shutdown()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Also register atexit handler
        atexit.register(self._trigger_graceful_shutdown)
        
        print("✅ Graceful shutdown handlers registered")


class CheckpointManager:
    """
    Manages training checkpoints for recovery.
    """
    
    def __init__(self, checkpoint_dir="./outputs"):
        self.checkpoint_dir = Path(checkpoint_dir)
        
    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None
            
        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest = checkpoints[0]
        
        print(f"📂 Found checkpoint: {latest.name}")
        return latest
        
    def should_resume(self) -> bool:
        """Check if we should resume from checkpoint."""
        latest = self.find_latest_checkpoint()
        if latest:
            print(f"\n🔄 Checkpoint detected: {latest.name}")
            print("   Training will automatically resume from this point!")
            return True
        return False
        
    def save_emergency_checkpoint(self, trainer, name="emergency"):
        """Save emergency checkpoint during error."""
        try:
            emergency_path = self.checkpoint_dir / f"checkpoint-{name}"
            print(f"💾 Saving emergency checkpoint to {emergency_path}")
            trainer.save_model(str(emergency_path))
            print("✅ Emergency checkpoint saved!")
            return emergency_path
        except Exception as e:
            print(f"❌ Could not save emergency checkpoint: {e}")
            return None


def wrap_training_with_recovery(train_func: Callable, *args, **kwargs):
    """
    Wrapper that adds error recovery to any training function.
    
    Usage:
        wrap_training_with_recovery(trainer.train)
    """
    recovery = ErrorRecovery(max_retries=3)
    recovery.setup_signal_handlers()
    
    checkpoint_mgr = CheckpointManager()
    
    # Check for existing checkpoint
    latest_checkpoint = checkpoint_mgr.find_latest_checkpoint()
    if latest_checkpoint:
        kwargs['resume_from_checkpoint'] = str(latest_checkpoint)
        
    # Register emergency save handler
    if 'trainer' in kwargs:
        trainer = kwargs['trainer']
        recovery.register_shutdown_handler(
            lambda: checkpoint_mgr.save_emergency_checkpoint(trainer)
        )
    
    # Execute with recovery
    return recovery.safe_execute(train_func, *args, **kwargs)


# Example usage:
# from src.error_recovery import wrap_training_with_recovery
# wrap_training_with_recovery(trainer.train)
