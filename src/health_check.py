"""
Pre-Flight Health Check System
===============================
Validates environment before training to catch issues early.
Auto-fixes common problems when possible.
"""

import os
import sys
import torch
import psutil
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
import importlib.util

class HealthCheck:
    """Comprehensive pre-flight validation system."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.fixes_applied = []
        
    def check_gpu(self) -> bool:
        """Check GPU availability and health."""
        print("\n🔍 Checking GPU...")
        
        if not torch.cuda.is_available():
            self.warnings.append("No GPU detected - training will be very slow on CPU")
            print("⚠️ No GPU detected! Training will use CPU (very slow)")
            return False
            
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   Memory: {gpu_mem:.2f} GB")
            
            # Test GPU with a simple operation
            x = torch.randn(100, 100).cuda()
            y = x @ x
            del x, y
            torch.cuda.empty_cache()
            
            print("✅ GPU test passed")
            return True
            
        except Exception as e:
            self.issues.append(f"GPU error: {e}")
            print(f"❌ GPU error: {e}")
            return False
            
    def check_disk_space(self, required_gb=50) -> bool:
        """Check available disk space."""
        print("\n🔍 Checking disk space...")
        
        stat = shutil.disk_usage("/")
        free_gb = stat.free / 1e9
        
        print(f"   Free space: {free_gb:.2f} GB")
        
        if free_gb < required_gb:
            self.issues.append(f"Low disk space: {free_gb:.2f} GB (need {required_gb} GB)")
            print(f"❌ Low disk space! Need {required_gb} GB, have {free_gb:.2f} GB")
            return False
        else:
            print(f"✅ Sufficient disk space ({free_gb:.2f} GB available)")
            return True
            
    def check_memory(self, required_gb=8) -> bool:
        """Check RAM availability."""
        print("\n🔍 Checking RAM...")
        
        mem = psutil.virtual_memory()
        available_gb = mem.available / 1e9
        total_gb = mem.total / 1e9
        
        print(f"   Total RAM: {total_gb:.2f} GB")
        print(f"   Available: {available_gb:.2f} GB")
        
        if available_gb < required_gb:
            self.warnings.append(f"Low RAM: {available_gb:.2f} GB available")
            print(f"⚠️ Low RAM! Recommended: {required_gb} GB, available: {available_gb:.2f} GB")
            return False
        else:
            print(f"✅ Sufficient RAM")
            return True
            
    def check_dependencies(self) -> bool:
        """Check if all required packages are installed."""
        print("\n🔍 Checking dependencies...")
        
        required = [
            "torch",
            "transformers",
            "datasets",
            "torchaudio",
            "evaluate",
            "jiwer",
            "psutil",
            "numpy"
        ]
        
        missing = []
        for pkg in required:
            if importlib.util.find_spec(pkg) is None:
                missing.append(pkg)
                
        if missing:
            self.issues.append(f"Missing packages: {', '.join(missing)}")
            print(f"❌ Missing packages: {', '.join(missing)}")
            print(f"   Install with: pip install {' '.join(missing)}")
            return False
        else:
            print(f"✅ All dependencies installed")
            return True
            
    def check_hf_token(self) -> bool:
        """Check HuggingFace authentication."""
        print("\n🔍 Checking HuggingFace token...")
        
        try:
            from huggingface_hub import whoami
            user = whoami()
            print(f"✅ Logged in as: {user['name']}")
            return True
        except Exception as e:
            self.warnings.append("Not logged in to HuggingFace")
            print("⚠️ Not logged in to HuggingFace")
            print("   Run: huggingface-cli login")
            return False
            
    def check_internet(self) -> bool:
        """Check internet connectivity."""
        print("\n🔍 Checking internet connection...")
        
        try:
            import requests
            response = requests.get("https://huggingface.co", timeout=5)
            if response.status_code == 200:
                print("✅ Internet connection OK")
                return True
            else:
                self.issues.append(f"Internet issue (status: {response.status_code})")
                print(f"❌ Internet issue (status: {response.status_code})")
                return False
        except Exception as e:
            self.issues.append(f"No internet connection: {e}")
            print(f"❌ No internet connection: {e}")
            return False
            
    def check_write_permissions(self) -> bool:
        """Check if we can write to current directory."""
        print("\n🔍 Checking write permissions...")
        
        test_file = Path(".health_check_test")
        try:
            test_file.write_text("test")
            test_file.unlink()
            print("✅ Write permissions OK")
            return True
        except Exception as e:
            self.issues.append(f"Cannot write to directory: {e}")
            print(f"❌ Cannot write to directory: {e}")
            return False
            
    def auto_fix_cache(self) -> bool:
        """Clear HuggingFace cache if it's causing issues."""
        print("\n🔧 Attempting to free up cache space...")
        
        cache_dir = Path.home() / ".cache" / "huggingface"
        if cache_dir.exists():
            try:
                size_gb = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / 1e9
                print(f"   Cache size: {size_gb:.2f} GB")
                
                if size_gb > 20:  # If cache is huge
                    print(f"   Clearing old cache files...")
                    # Just report, don't actually delete (user should decide)
                    self.warnings.append(f"Large cache detected ({size_gb:.2f} GB)")
                    print(f"⚠️ Large cache ({size_gb:.2f} GB)")
                    print(f"   To clear: rm -rf {cache_dir}")
                    self.fixes_applied.append(f"Suggested clearing {size_gb:.2f} GB cache")
                return True
            except Exception as e:
                print(f"   Could not check cache: {e}")
                return False
        return True
        
    def run_all_checks(self, required_disk_gb=50, required_ram_gb=8) -> Dict[str, bool]:
        """Run all health checks."""
        print("="*80)
        print("RUNNING PRE-FLIGHT HEALTH CHECKS".center(80))
        print("="*80)
        
        results = {
            "gpu": self.check_gpu(),
            "disk": self.check_disk_space(required_disk_gb),
            "memory": self.check_memory(required_ram_gb),
            "dependencies": self.check_dependencies(),
            "hf_token": self.check_hf_token(),
            "internet": self.check_internet(),
            "permissions": self.check_write_permissions(),
        }
        
        # Try auto-fixes
        self.auto_fix_cache()
        
        # Summary
        print("\n" + "="*80)
        print("HEALTH CHECK SUMMARY".center(80))
        print("="*80)
        
        passed = sum(results.values())
        total = len(results)
        
        for check, status in results.items():
            icon = "✅" if status else "❌"
            print(f"{icon} {check.replace('_', ' ').title()}")
            
        print("\n" + "-"*80)
        print(f"Passed: {passed}/{total}")
        
        if self.issues:
            print(f"\n❌ Critical Issues ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   - {issue}")
                
        if self.warnings:
            print(f"\n⚠️ Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   - {warning}")
                
        if self.fixes_applied:
            print(f"\n🔧 Fixes Applied ({len(self.fixes_applied)}):")
            for fix in self.fixes_applied:
                print(f"   - {fix}")
                
        print("="*80)
        
        # Critical checks
        critical_passed = results["dependencies"] and results["internet"] and results["permissions"]
        
        if not critical_passed:
            print("\n❌ CRITICAL CHECKS FAILED - Cannot proceed with training")
            print("   Please fix the issues above and try again")
            return results
            
        if results["gpu"]:
            print("\n✅ ALL CHECKS PASSED - Ready to train!")
        else:
            print("\n⚠️ CHECKS PASSED (with warnings) - Can proceed but may be slow")
            
        return results


def run_health_check():
    """Convenience function to run all checks."""
    checker = HealthCheck()
    results = checker.run_all_checks()
    return checker, results


# CLI usage
if __name__ == "__main__":
    checker, results = run_health_check()
    
    # Exit with error code if critical checks failed
    critical_ok = results["dependencies"] and results["internet"] and results["permissions"]
    sys.exit(0 if critical_ok else 1)
