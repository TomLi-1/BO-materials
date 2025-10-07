#!/usr/bin/env python3
"""
Diagnostic script to identify what's causing segmentation faults.
Tests each component individually to isolate the problem.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def test_basic_imports():
    """Test basic imports one by one."""
    print("🔍 Testing basic imports...")
    
    imports_to_test = [
        ("warnings", "import warnings"),
        ("torch", "import torch"),
        ("numpy", "import numpy as np"),
        ("sklearn", "from sklearn.metrics import mean_absolute_error"),
        ("pymatgen", "from pymatgen.core import Structure"),
        ("matminer basic", "from matminer.featurizers.composition import ElementProperty"),
    ]
    
    for name, import_cmd in imports_to_test:
        try:
            print(f"   Testing {name}...")
            exec(import_cmd)
            print(f"   ✅ {name} OK")
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
            return False
    
    return True

def test_botorch_imports():
    """Test BoTorch imports specifically."""
    print("\n🤖 Testing BoTorch imports...")
    
    botorch_imports = [
        ("botorch.models", "from botorch.models import SingleTaskGP"),
        ("botorch.fit", "from botorch.fit import fit_gpytorch_mll"),
        ("botorch.acquisition", "from botorch.acquisition import ExpectedImprovement"),
        ("gpytorch", "import gpytorch"),
    ]
    
    for name, import_cmd in botorch_imports:
        try:
            print(f"   Testing {name}...")
            exec(import_cmd)
            print(f"   ✅ {name} OK")
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
            return False
    
    return True

def test_data_loading():
    """Test data loading."""
    print("\n📊 Testing data loading...")
    
    try:
        from src.data_loader import load_formation_energy
        print("   Loading small dataset...")
        X_train, X_test, y_train, y_test = load_formation_energy()
        
        # Use tiny subset
        X_small = X_train[:2]
        print(f"   ✅ Data loading OK ({len(X_small)} samples)")
        return X_small, y_train[:2]
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return None, None

def test_featurizers():
    """Test different featurizers."""
    print("\n🧬 Testing featurizers...")
    
    X_small, y_small = test_data_loading()
    if X_small is None:
        return False
    
    # Test safe featurizer
    try:
        print("   Testing safe Matbench featurizer...")
        from src.safe_matbench_featurizer import make_safe_matbench_featurizer
        featurizer = make_safe_matbench_featurizer(embedding_dim=64)  # Smaller for testing
        features = featurizer.transform(X_small)
        print(f"   ✅ Safe featurizer OK ({features.shape})")
    except Exception as e:
        print(f"   ❌ Safe featurizer failed: {e}")
        return False
    
    # Test basic Magpie
    try:
        print("   Testing basic Magpie featurizer...")
        from src.featurizer import MatminerTransformer, make_magpie_featurizer
        transformer = MatminerTransformer(make_magpie_featurizer())
        feats = transformer.transform(X_small)
        print(f"   ✅ Magpie featurizer OK")
    except Exception as e:
        print(f"   ❌ Magpie featurizer failed: {e}")
        return False
    
    return True

def test_gp_model():
    """Test GP model creation."""
    print("\n🤖 Testing GP model...")
    
    try:
        import torch
        from src.surrogate import fit_surrogate
        
        # Create dummy data
        X_dummy = torch.randn(10, 5, dtype=torch.float32)
        y_dummy = torch.randn(10, dtype=torch.float32)
        
        print("   Creating GP model...")
        gp = fit_surrogate(X_dummy, y_dummy)
        print("   ✅ GP model OK")
        return True
    except Exception as e:
        print(f"   ❌ GP model failed: {e}")
        return False

def test_bo_components():
    """Test BO components."""
    print("\n🎯 Testing BO components...")
    
    try:
        from src.surrogate import OptimizerParameters
        from src.bo_loop import bayes_optimize
        
        params = OptimizerParameters(
            total_sample_budget=5,  # Very small
            initialization_budget=2
        )
        print("   ✅ BO parameters OK")
        
        # Don't actually run BO (could be expensive), just test import
        print("   ✅ BO loop import OK")
        return True
    except Exception as e:
        print(f"   ❌ BO components failed: {e}")
        return False

def minimal_run_test():
    """Test a minimal version of the main pipeline."""
    print("\n🔬 Testing minimal pipeline...")
    
    try:
        # Test just the setup parts of run.py without the heavy computation
        import torch
        import time
        from src.data_loader import load_formation_energy
        from src.safe_matbench_featurizer import make_safe_matbench_featurizer
        from src.surrogate import OptimizerParameters
        
        print("   Loading minimal data...")
        X_train, X_test, y_train, y_test = load_formation_energy()
        X_train_small = X_train[:3]
        X_test_small = X_test[:2]
        y_train_small = y_train[:3]
        y_test_small = y_test[:2]
        
        print("   Testing featurization...")
        transformer = make_safe_matbench_featurizer(embedding_dim=32)  # Very small
        X_train_feats = transformer.transform(X_train_small)
        X_test_feats = transformer.transform(X_test_small)
        
        print("   Converting to tensors...")
        X_train_t = torch.tensor(X_train_feats, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_small.values, dtype=torch.float32)
        
        print("   ✅ Minimal pipeline OK")
        return True
        
    except Exception as e:
        print(f"   ❌ Minimal pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests."""
    print("🩺 SEGMENTATION FAULT DIAGNOSTIC")
    print("=" * 50)
    
    # Run tests in order of complexity
    tests = [
        ("Basic Imports", test_basic_imports),
        ("BoTorch Imports", test_botorch_imports),
        ("Data Loading", lambda: test_data_loading()[0] is not None),
        ("Featurizers", test_featurizers),
        ("GP Model", test_gp_model),
        ("BO Components", test_bo_components),
        ("Minimal Pipeline", minimal_run_test),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            success = test_func()
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                print(f"\n🚨 ISSUE FOUND: {test_name} is likely causing the segfault!")
                break
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            print(f"\n🚨 SEGFAULT LIKELY HERE: {test_name}")
            import traceback
            traceback.print_exc()
            break
    else:
        print(f"\n🤔 All tests passed individually - issue may be in combination or size")
        print(f"Try reducing dataset size in run.py or running with smaller parameters")

if __name__ == "__main__":
    main()