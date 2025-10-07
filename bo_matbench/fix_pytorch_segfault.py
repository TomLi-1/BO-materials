#!/usr/bin/env python3
"""
Script to diagnose and fix PyTorch segmentation fault issues.
This is a common problem with PyTorch installations in conda environments.
"""

import subprocess
import sys
import os

def check_environment():
    """Check current environment details."""
    print("🔍 ENVIRONMENT DIAGNOSIS")
    print("=" * 40)
    
    # Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Conda environment
    try:
        result = subprocess.run(['conda', 'list'], capture_output=True, text=True)
        if 'torch' in result.stdout:
            print("✅ PyTorch found in conda list")
            for line in result.stdout.split('\n'):
                if 'torch' in line and not line.startswith('#'):
                    print(f"   {line}")
        else:
            print("❌ PyTorch not found in conda list")
    except:
        print("⚠️  Cannot run conda list")
    
    # Check for conflicting installations
    print(f"\n🔍 Checking for conflicts...")
    try:
        result = subprocess.run([sys.executable, '-c', 'import sys; print(sys.path[:3])'], 
                              capture_output=True, text=True)
        print(f"Python path: {result.stdout.strip()}")
    except:
        print("❌ Cannot check Python path")

def test_minimal_imports():
    """Test imports one by one to isolate the issue."""
    print(f"\n🧪 TESTING MINIMAL IMPORTS")
    print("=" * 40)
    
    imports_to_test = [
        "import sys",
        "import os", 
        "import numpy",
        "import sklearn",
        "import warnings",
    ]
    
    for import_cmd in imports_to_test:
        try:
            print(f"Testing: {import_cmd}")
            result = subprocess.run([sys.executable, '-c', import_cmd], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"   ✅ OK")
            else:
                print(f"   ❌ Failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"   ❌ Timeout")
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    return True

def fix_pytorch_installation():
    """Provide instructions to fix PyTorch installation."""
    print(f"\n🔧 PYTORCH SEGFAULT FIX")
    print("=" * 40)
    
    print("This is a known issue with PyTorch in conda environments.")
    print("Here are the steps to fix it:")
    print()
    
    print("OPTION 1: Clean PyTorch Reinstall")
    print("-" * 30)
    print("conda activate bo_matbench")
    print("conda uninstall pytorch torchvision torchaudio -y")
    print("conda clean --all")
    print("conda install pytorch torchvision torchaudio cpuonly -c pytorch -y")
    print()
    
    print("OPTION 2: Use pip instead of conda for PyTorch")
    print("-" * 30) 
    print("conda activate bo_matbench")
    print("conda uninstall pytorch torchvision torchaudio -y")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print()
    
    print("OPTION 3: Create fresh environment")
    print("-" * 30)
    print("conda create -n bo_matbench_new python=3.10 -y")
    print("conda activate bo_matbench_new")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print("pip install botorch gpytorch pymatgen matminer scikit-learn pandas numpy matplotlib")
    print()
    
    print("OPTION 4: Use CPU-only PyTorch (safest)")
    print("-" * 30)
    print("conda activate bo_matbench")
    print("conda uninstall pytorch torchvision torchaudio -y")
    print("conda install pytorch torchvision torchaudio cpuonly -c pytorch -y")
    print("# Then restart your terminal")
    print()

def test_pytorch_specifically():
    """Test PyTorch import in subprocess to avoid crashing this script."""
    print(f"\n🤖 TESTING PYTORCH IMPORT")
    print("=" * 40)
    
    pytorch_tests = [
        "import torch",
        "import torch; print(torch.__version__)",
        "import torch; print('PyTorch version:', torch.__version__)",
        "import torch; x = torch.tensor([1.0]); print('Tensor created:', x)",
    ]
    
    for i, test_cmd in enumerate(pytorch_tests):
        print(f"Test {i+1}: {test_cmd}")
        try:
            result = subprocess.run([sys.executable, '-c', test_cmd], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"   ✅ Success: {result.stdout.strip()}")
            else:
                print(f"   ❌ Failed with return code {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr.strip()}")
                break
        except subprocess.TimeoutExpired:
            print(f"   ❌ Timeout (likely segfault)")
            break
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            break
    else:
        print(f"\n🎉 PyTorch is working! The issue might be elsewhere.")
        return True
    
    print(f"\n💥 PyTorch segfault confirmed!")
    return False

def main():
    """Main diagnostic function."""
    print("🩺 PYTORCH SEGFAULT DIAGNOSIS & FIX")
    print("=" * 50)
    
    # Check environment
    check_environment()
    
    # Test basic imports first
    if not test_minimal_imports():
        print("\n❌ Basic imports failing - deeper Python environment issue")
        return
    
    # Test PyTorch specifically
    pytorch_works = test_pytorch_specifically()
    
    if not pytorch_works:
        fix_pytorch_installation()
        
        print(f"\n📋 RECOMMENDED ACTION:")
        print("1. Try OPTION 1 (clean reinstall) first")
        print("2. If that fails, try OPTION 2 (pip install)")
        print("3. If still failing, use OPTION 3 (fresh environment)")
        print("4. After fixing, test with: python -c 'import torch; print(torch.__version__)'")
        print()
        print("Note: This is a common PyTorch/conda compatibility issue,")
        print("not a problem with our BO pipeline code.")
    else:
        print(f"\n🤔 PyTorch works fine. The segfault might be caused by:")
        print("- BoTorch/GPyTorch compatibility issues")
        print("- Memory issues with large computations")
        print("- Other package conflicts")
        print()
        print("Try: conda list | grep torch")
        print("And: conda list | grep scipy")

if __name__ == "__main__":
    main()