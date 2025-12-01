#!/usr/bin/env python3
"""
Quick Verification Script
Checks if everything is ready for deployment and training
"""

import sys
import subprocess

def check_command(cmd, package_name=None):
    """Check if a command exists"""
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_python_package(package):
    """Check if a Python package is installed"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    print("="*60)
    print(" ASR Training System - Pre-Deployment Verification")
    print("="*60)
    print()
    
    checks = {
        "Python 3.8+": sys.version_info >= (3, 8),
        "Git installed": check_command("git"),
        "PyTorch": check_python_package("torch"),
        "Transformers": check_python_package("transformers"),
        "Datasets": check_python_package("datasets"),
        "TorchAudio": check_python_package("torchaudio"),
       "Evaluate": check_python_package("evaluate"),
        "PSUtil": check_python_package("psutil"),
    }
    
    # Check CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            checks[f"CUDA (GPU: {gpu_name}, {gpu_memory:.1f}GB)"] = True
        else:
            checks["CUDA (GPU)"] = False
    except:
        checks["CUDA (GPU)"] = False
    
    # Print results
    all_passed = True
    for check, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {check}")
        if not status:
            all_passed = False
    
    print()
    print("="*60)
    
    if all_passed:
        print("✅ All checks passed! Ready to deploy and train.")
        print()
        print("Next steps:")
        print("1. Push to GitHub: git push origin main")
        print("2. Deploy to cloud platform (see DEPLOYMENT.md)")
        print("3. Run training: python src/optimized_training.py")
    else:
        print("❌ Some checks failed. Please install missing dependencies.")
        print()
        print("To install all requirements:")
        print("  pip install -r requirements.txt")
        print()
        if not checks.get("CUDA (GPU)", False):
            print("⚠️  No GPU detected. Training will be very slow (CPU-only).")
            print("   Consider using a cloud platform with GPU (see DEPLOYMENT.md)")
    
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
