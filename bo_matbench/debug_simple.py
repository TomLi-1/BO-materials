#!/usr/bin/env python3

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("🔍 Simple debug test...")

try:
    import torch
    import numpy as np
    from src.multi_objective_data_loader import load_bandgap_dataset
    from src.featurizer import MatminerTransformer, make_magpie_featurizer
    
    print("✅ All imports successful")
    
    # Load small dataset
    X_train, X_test, y_train, y_test, train_meta, test_meta = load_bandgap_dataset()
    X_train = X_train[:20]  # Very small
    X_test = X_test[:10]
    y_train = y_train[:20]
    y_test = y_test[:10]
    
    print(f"✅ Loaded {len(X_train)} + {len(X_test)} samples")
    
    # Featurize
    transformer = MatminerTransformer(make_magpie_featurizer())
    print("✅ Transformer created")
    
    X_train_feats = transformer.transform(X_train)
    print(f"✅ Train featurization done: {len(X_train_feats)} samples")
    
    X_test_feats = transformer.transform(X_test)
    print(f"✅ Test featurization done: {len(X_test_feats)} samples")
    
    # Convert to arrays
    X_train_feats = np.array(X_train_feats)
    X_test_feats = np.array(X_test_feats)
    print(f"✅ Numpy conversion: {X_train_feats.shape}, {X_test_feats.shape}")
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train_feats, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_feats, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
    print(f"✅ Tensor conversion successful")
    
    # Test pool creation
    X_pool = torch.cat([X_train_t, X_test_t], dim=0)
    y_pool = torch.cat([y_train_t, y_test_t], dim=0)
    print(f"✅ Pool created: {X_pool.shape}, {y_pool.shape}")
    
    # Test basic BO components
    from src.multi_objective_surrogate import MultiObjectiveParameters
    params = MultiObjectiveParameters(
        total_sample_budget=8,
        initialization_budget=5,
        num_sparsity_feats=10
    )
    print(f"✅ Parameters created")
    
    print("🎉 All basic components work! The issue is in the BO loop itself.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()