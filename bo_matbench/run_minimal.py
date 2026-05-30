#!/usr/bin/env python3
"""
Minimal version of run.py to test if the main pipeline works without segfaults.
Uses smaller dataset and fewer BO iterations.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import warnings
import random
from botorch.optim.fit import OptimizationWarning

from src.data_loader import load_formation_energy
from src.safe_matbench_featurizer import make_safe_matbench_featurizer
from src.surrogate import OptimizerParameters, evaluate_model_predictions, fit_surrogate, select_features
from src.bo_loop import bayes_optimize
import torch
import time
import numpy as np

warnings.filterwarnings("ignore", category=OptimizationWarning)

def main():
    print("🧪 MINIMAL BO PIPELINE TEST")
    print("=" * 40)
    
    # Configuration
    featurizer_type = "matbench"
    high_energy_method = "standard"
    
    print(f"🔧 Using featurizer: {featurizer_type}")
    print(f"🔧 High-energy method: {high_energy_method}")
    
    # 1) Load VERY small dataset
    print("\n📊 Loading minimal dataset...")
    X_train, X_test, y_train, y_test = load_formation_energy()
    
    # Use tiny subset for testing
    subset_size = 50  # Very small
    X_train = X_train[:subset_size]
    X_test = X_test[:20]
    y_train = y_train[:subset_size]
    y_test = y_test[:20]
    
    print(f"✅ Data loaded: {len(X_train)} train, {len(X_test)} test")
    print(f"Energy range: [{y_train.min():.3f}, {y_train.max():.3f}] eV/atom")
    
    # 2) Featurize
    print(f"\n🧬 Featurizing {len(X_train)} train + {len(X_test)} test samples...")
    
    start_time = time.time()
    transformer = make_safe_matbench_featurizer(embedding_dim=64)  # Smaller embedding
    X_train_feats = transformer.transform(X_train)
    X_test_feats = transformer.transform(X_test)
    
    # Convert to numpy arrays if needed
    X_train_feats = np.array(X_train_feats)
    X_test_feats = np.array(X_test_feats)
    
    feature_dim = X_train_feats.shape[1]
    featurization_time = time.time() - start_time
    print(f"✅ Safe Matbench-inspired features extracted: {feature_dim} dimensions")
    print(f"⏱️  Featurization time: {featurization_time:.2f} seconds")
    
    # 3) Convert to tensors
    X_train_t = torch.tensor(X_train_feats, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_feats, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
    
    X_pool = torch.cat([X_train_t, X_test_t], dim=0)
    y_pool = torch.cat([y_train_t, y_test_t], dim=0)
    
    print(f"📊 Pool: {X_pool.shape}, energy range: [{y_pool.min():.3f}, {y_pool.max():.3f}]")
    
    # 4) Minimal BO parameters
    params = OptimizerParameters(
        sparsity_method="MI",
        acq_fun="EI",
        num_sparsity_feats=15,  # Smaller
        initialization_budget=5,  # Smaller
        total_sample_budget=15,  # Much smaller for testing
    )
    
    print(f"\n🤖 Running minimal BO (budget: {params.total_sample_budget})...")
    
    # 5) Initialize and run BO
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)
    generator = torch.Generator().manual_seed(params.seed)
    init_idx = torch.randperm(len(X_pool), generator=generator)[:params.initialization_budget]
    X_init = X_pool[init_idx]
    y_init = y_pool[init_idx]
    pool_mask = torch.ones(len(X_pool), dtype=torch.bool)
    pool_mask[init_idx] = False
    X_pool_remaining = X_pool[pool_mask]
    y_pool_remaining = y_pool[pool_mask]
    
    print(f"   Initial best: {y_init.min().item():.4f} eV/atom")
    
    try:
        data_X, data_y = bayes_optimize(X_init, y_init, X_pool_remaining, y_pool_remaining, params)
        print(f"✅ BO complete! Final best: {data_y.min().item():.4f} eV/atom")
        
        # 6) Quick evaluation
        print(f"\n📊 Quick model evaluation...")
        
        # Feature selection
        final_X_fs, final_feat_idx = select_features(
            data_X.numpy(), data_y.numpy(),
            method=params.sparsity_method, k=params.num_sparsity_feats,
            random_state=params.seed
        )
        final_X_fs_torch = torch.tensor(final_X_fs, dtype=torch.float32)
        
        # Fit model
        print(f"🔧 Using standard surrogate modeling...")
        final_gp = fit_surrogate(final_X_fs_torch, data_y)
        
        # Create test set
        train_mask = torch.zeros(len(X_pool), dtype=torch.bool)
        for i, x_train in enumerate(data_X):
            distances = torch.norm(X_pool - x_train.unsqueeze(0), dim=1)
            train_mask[torch.argmin(distances)] = True
        
        test_mask = ~train_mask
        X_test_full = X_pool[test_mask]
        y_test_full = y_pool[test_mask]
        
        if len(X_test_full) > 0:
            print(f"Evaluating on {len(X_test_full)} held-out test samples...")
            results = evaluate_model_predictions(
                final_gp, X_test_full, y_test_full, 
                feat_idx=final_feat_idx, verbose=True
            )
            
            print(f"\n🎉 MINIMAL PIPELINE SUCCESS!")
            print(f"   MAE: {results['mae']:.3f} eV/atom")
            print(f"   R²: {results['r2']:.3f}")
        else:
            print(f"⚠️  No test data for evaluation")
            print(f"\n🎉 MINIMAL PIPELINE SUCCESS!")
        
    except Exception as e:
        print(f"❌ BO failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
