#!/usr/bin/env python3

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("🧪 Testing feature selection fix...")

try:
    import numpy as np
    from src.multi_objective_surrogate import select_features_multi_objective
    
    # Create test data that caused the issue
    X = np.random.randn(3, 5)  # 3 samples, 5 features
    y = np.random.randn(3)     # 3 targets
    
    print(f"Test data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test MI feature selection
    print("Testing MI feature selection...")
    X_selected, feat_idx = select_features_multi_objective(X, y, method="MI", k=3)
    print(f"✅ MI selection successful: selected {len(feat_idx)} features")
    
    # Test LASSO feature selection  
    print("Testing LASSO feature selection...")
    X_selected, feat_idx = select_features_multi_objective(X, y, method="LASSO", k=3)
    print(f"✅ LASSO selection successful: selected {len(feat_idx)} features")
    
    print("✅ Feature selection fix works!")
    
except Exception as e:
    print(f"❌ Feature selection failed: {e}")
    import traceback
    traceback.print_exc()