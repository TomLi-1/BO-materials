#!/usr/bin/env python3
"""
Simple test of baseline (reverted) surrogate modeling without full BO loop.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import warnings
from botorch.optim.fit import OptimizationWarning

from src.data_loader import load_formation_energy
from src.featurizer import MatminerTransformer, make_magpie_featurizer
from src.surrogate import fit_surrogate, select_features, evaluate_model_predictions
import torch
import numpy as np

warnings.filterwarnings("ignore", category=OptimizationWarning)

def main():
    print("🧪 TESTING BASELINE (REVERTED) SURROGATE")
    print("=" * 40)
    
    # Small dataset for quick test
    subset_size = 1000
    
    # 1) Load data
    X_train, X_test, y_train, y_test = load_formation_energy()
    
    X_train = X_train[:subset_size]
    X_test = X_test[:200]
    y_train = y_train[:subset_size]
    y_test = y_test[:200]
    
    print(f"✅ Data: {len(X_train)} train, {len(X_test)} test")
    print(f"Energy range: [{y_train.min():.3f}, {y_train.max():.3f}] eV/atom")
    
    # 2) Featurize
    print(f"\n🧬 Featurizing...")
    transformer = MatminerTransformer(make_magpie_featurizer())
    X_train_feats = np.array(transformer.transform(X_train))
    X_test_feats = np.array(transformer.transform(X_test))
    
    print(f"✅ Features: {X_train_feats.shape[1]} dimensions")
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train_feats, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_feats, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
    
    # 3) Feature selection
    print(f"\n🔧 Feature selection...")
    X_selected, feat_idx = select_features(
        X_train_feats, y_train.values, method="MI", k=30
    )
    X_selected_t = torch.tensor(X_selected, dtype=torch.float32)
    
    print(f"✅ Selected {len(feat_idx)} features")
    
    # 4) Fit baseline GP
    print(f"\n🤖 Fitting baseline GP...")
    gp = fit_surrogate(X_selected_t, y_train_t)
    
    # 5) Evaluate
    print(f"\n📊 Evaluating...")
    results = evaluate_model_predictions(
        gp, X_test_t, y_test_t, feat_idx=feat_idx, verbose=True
    )
    
    # 6) Check high-energy performance
    high_energy_mask = y_test.values > 0.0
    if high_energy_mask.sum() > 0:
        print(f"\n🔥 High-energy analysis ({high_energy_mask.sum()} unstable materials):")
        y_true_high = y_test.values[high_energy_mask]
        y_pred_high = results['predictions'][high_energy_mask]
        mae_high = np.mean(np.abs(y_true_high - y_pred_high))
        print(f"MAE (high-energy): {mae_high:.3f} eV/atom")
        print(f"Average high-energy prediction: {y_pred_high.mean():.3f} eV/atom")
        print(f"Average high-energy true value: {y_true_high.mean():.3f} eV/atom")
    
    print(f"\n✅ BASELINE TEST COMPLETE!")
    print(f"Overall MAE: {results['mae']:.3f} eV/atom")

if __name__ == "__main__":
    main()