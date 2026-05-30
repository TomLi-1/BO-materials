#!/usr/bin/env python3
"""
Test robust formation energy modeling improvements.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import numpy as np
import warnings
from botorch.optim.fit import OptimizationWarning

from src.data_loader import load_formation_energy
from src.featurizer import MatminerTransformer, make_magpie_featurizer
from src.surrogate import (
    fit_surrogate, fit_robust_surrogate, 
    evaluate_model_predictions, evaluate_robust_model,
    select_features
)

warnings.filterwarnings("ignore", category=OptimizationWarning)

def main():
    print("🧪 TESTING ROBUST FORMATION ENERGY MODELING")
    print("=" * 60)
    
    # Load data
    print("📊 Loading formation energy data...")
    X_train, X_test, y_train, y_test = load_formation_energy()
    
    # Take subset for testing
    subset_size = 5000
    X_train = X_train[:subset_size]
    y_train = y_train[:subset_size]
    X_test = X_test[:1000]
    y_test = y_test[:1000]
    
    print(f"Using subset: {len(X_train)} train, {len(X_test)} test")
    print(f"Energy range: [{y_train.min():.3f}, {y_train.max():.3f}] eV/atom")
    
    # Featurize
    print("\n🧬 Featurizing...")
    transformer = MatminerTransformer(make_magpie_featurizer())
    X_train_feats = transformer.transform(X_train)
    X_test_feats = transformer.transform(X_test)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train_feats, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_feats, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
    
    # Feature selection
    print("\n🔍 Selecting features...")
    X_train_feats_np = np.array(X_train_feats)
    X_reduced, feat_idx = select_features(
        X_train_feats_np, y_train.values, method="MI", k=30
    )
    X_train_reduced = torch.tensor(X_reduced, dtype=torch.float32)
    X_test_reduced = X_test_t[:, feat_idx]
    
    print(f"Selected {len(feat_idx)} features")
    
    # Compare standard vs robust modeling
    print("\n" + "="*50)
    print("COMPARING STANDARD vs ROBUST MODELING")
    print("="*50)
    
    # 1. Standard modeling
    print("\n1️⃣ STANDARD GP MODELING:")
    try:
        standard_gp = fit_surrogate(X_train_reduced, y_train_t)
        standard_results = evaluate_model_predictions(
            standard_gp, X_test_reduced, y_test_t, verbose=True
        )
    except Exception as e:
        print(f"❌ Standard modeling failed: {e}")
        standard_results = None
    
    # 2. Robust modeling
    print("\n2️⃣ ROBUST GP MODELING:")
    try:
        robust_gp, robust_info = fit_robust_surrogate(X_train_reduced, y_train_t)
        robust_results = evaluate_robust_model(
            robust_gp, X_test_reduced, y_test_t, robust_info, verbose=True
        )
    except Exception as e:
        print(f"❌ Robust modeling failed: {e}")
        robust_results = None
    
    # Comparison
    if standard_results and robust_results:
        print("\n" + "="*40)
        print("📊 COMPARISON SUMMARY")
        print("="*40)
        print(f"{'Method':<20} {'MAE':<8} {'RMSE':<8} {'R²':<8}")
        print("-" * 45)
        print(f"{'Standard GP':<20} {standard_results['mae']:<8.3f} {standard_results['rmse']:<8.3f} {standard_results['r2']:<8.3f}")
        print(f"{'Robust GP':<20} {robust_results['mae']:<8.3f} {robust_results['rmse']:<8.3f} {robust_results['r2']:<8.3f}")
        
        mae_improvement = (standard_results['mae'] - robust_results['mae']) / standard_results['mae'] * 100
        rmse_improvement = (standard_results['rmse'] - robust_results['rmse']) / standard_results['rmse'] * 100
        
        print(f"\n🎯 Improvements:")
        print(f"   MAE:  {mae_improvement:+.1f}% ({'better' if mae_improvement > 0 else 'worse'})")
        print(f"   RMSE: {rmse_improvement:+.1f}% ({'better' if rmse_improvement > 0 else 'worse'})")
        
        if robust_info:
            print(f"\n🔧 Robust modeling details:")
            print(f"   Outliers removed: {robust_info['outliers_removed']}")
            print(f"   Clean/Total samples: {robust_info['clean_size']}/{robust_info['original_size']}")
        
        print(f"\n✅ Robust modeling test complete!")
        
        if mae_improvement > 5:
            print(f"🎉 Significant improvement achieved!")
        elif mae_improvement > 0:
            print(f"📈 Modest improvement achieved")
        else:
            print(f"📊 No improvement over standard method")
    
    else:
        print("❌ Could not complete comparison")

if __name__ == "__main__":
    main()