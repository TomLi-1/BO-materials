#!/usr/bin/env python3
"""Simple test of GNN featurizer to isolate any issues"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import torch

print("🧪 Simple GNN Test")
print("=" * 30)

try:
    # 1. Load data
    print("Loading data...")
    from src.multi_objective_data_loader import load_bandgap_dataset
    X_train, X_test, y_train, y_test, _, _ = load_bandgap_dataset()
    
    # Take small subset
    X_train = X_train[:50]
    X_test = X_test[:30]
    y_train = y_train[:50]
    y_test = y_test[:30]
    print(f"✅ Data loaded: {len(X_train)} + {len(X_test)} samples")
    
    # 2. Test GNN featurizer
    print("\nTesting GNN featurizer...")
    from src.gnn_featurizer import make_gnn_featurizer
    
    transformer = make_gnn_featurizer(embedding_dim=32, pooling_strategy="max")
    print("✅ GNN transformer created")
    
    # Featurize
    X_train_feats = transformer.transform(X_train)
    X_test_feats = transformer.transform(X_test)
    print(f"✅ Featurization complete: {X_train_feats.shape}, {X_test_feats.shape}")
    
    # 3. Convert to tensors
    X_train_t = torch.tensor(X_train_feats, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_feats, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
    print(f"✅ Tensor conversion: {X_train_t.shape}, {X_test_t.shape}")
    
    # 4. Pool data
    X_pool = torch.cat([X_train_t, X_test_t], dim=0)
    y_pool = torch.cat([y_train_t, y_test_t], dim=0)
    print(f"✅ Data pooling: {X_pool.shape}, {y_pool.shape}")
    
    print(f"\n🎉 SUCCESS! GNN featurizer working properly")
    print(f"Final feature dimensions: {X_pool.shape[1]}")
    print(f"Sample features: {X_pool[0][:5].tolist()}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()