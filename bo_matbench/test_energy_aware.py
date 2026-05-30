#!/usr/bin/env python3
"""
Test energy-aware surrogate modeling approaches against standard GP.
Compares performance on formation energy prediction, especially at high energies.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import warnings
import random
from botorch.optim.fit import OptimizationWarning

from src.data_loader import load_formation_energy
from src.featurizer import MatminerTransformer, make_magpie_featurizer
from src.surrogate import OptimizerParameters, fit_surrogate, select_features, evaluate_model_predictions
from src.energy_aware_surrogate import fit_energy_aware_surrogate, evaluate_energy_aware_model
from src.bo_loop import bayes_optimize
import torch
import time
import numpy as np

warnings.filterwarnings("ignore", category=OptimizationWarning)

def main():
    print("🧪 TESTING ENERGY-AWARE SURROGATE MODELING")
    print("=" * 50)
    
    subset_size = 5000  # Reasonable size for testing
    
    # 1) Load and prepare data
    X_train, X_test, y_train, y_test = load_formation_energy()
    
    # Take subset for testing
    X_train = X_train[:subset_size]
    X_test = X_test[:1000]
    y_train = y_train[:subset_size]
    y_test = y_test[:1000]
    
    print(f"✅ Data loaded: {len(X_train)} train, {len(X_test)} test")
    print(f"Energy range: [{y_train.min():.3f}, {y_train.max():.3f}] eV/atom")
    
    # Check distribution across energy ranges
    stable = (y_train < -1.0).sum()
    intermediate = ((y_train >= -1.0) & (y_train < 0.5)).sum()
    unstable = (y_train >= 0.5).sum()
    print(f"Energy distribution: Stable={stable}, Intermediate={intermediate}, Unstable={unstable}")
    
    # 2) Featurize
    print(f"\n🧬 Featurizing with Magpie features...")
    start_time = time.time()
    transformer = MatminerTransformer(make_magpie_featurizer())
    X_train_feats = transformer.transform(X_train)
    X_test_feats = transformer.transform(X_test)
    
    X_train_feats = np.array(X_train_feats)
    X_test_feats = np.array(X_test_feats)
    
    feature_dim = X_train_feats.shape[1]
    featurization_time = time.time() - start_time
    print(f"✅ Magpie features: {feature_dim} dimensions ({featurization_time:.2f}s)")
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train_feats, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_feats, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
    
    # 3) Combine into pool for mini BO
    X_pool = torch.cat([X_train_t, X_test_t], dim=0)
    y_pool = torch.cat([y_train_t, y_test_t], dim=0)
    
    print(f"📊 Pool: {X_pool.shape}, energy range: [{y_pool.min():.3f}, {y_pool.max():.3f}]")
    
    # 4) Run mini BO to get selected training data
    print(f"\n🤖 Running mini BO for data selection...")
    params = OptimizerParameters(
        sparsity_method="MI",
        acq_fun="EI",
        num_sparsity_feats=30,
        initialization_budget=20,
        total_sample_budget=100,  # Reasonable for testing
    )
    
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
    
    data_X, data_y = bayes_optimize(X_init, y_init, X_pool_remaining, y_pool_remaining, params)
    
    print(f"✅ BO complete! Best found: {data_y.min().item():.4f} eV/atom")
    
    # 5) Feature selection
    final_X_fs, final_feat_idx = select_features(
        data_X.numpy(), data_y.numpy(),
        method=params.sparsity_method, k=params.num_sparsity_feats,
        random_state=params.seed
    )
    final_X_fs_torch = torch.tensor(final_X_fs, dtype=torch.float32)
    
    # 6) Create test set (remainder of pool)
    train_mask = torch.zeros(len(X_pool), dtype=torch.bool)
    for i, x_train in enumerate(data_X):
        distances = torch.norm(X_pool - x_train.unsqueeze(0), dim=1)
        train_mask[torch.argmin(distances)] = True
    
    test_mask = ~train_mask
    X_test_full = X_pool[test_mask]
    y_test_full = y_pool[test_mask]
    
    if len(X_test_full) == 0:
        print("⚠️  No test data available for evaluation")
        return
    
    print(f"\n📊 COMPARING SURROGATE MODELS")
    print("=" * 40)
    print(f"Training on {len(data_X)} selected samples")
    print(f"Testing on {len(X_test_full)} held-out samples")
    
    # 7) Standard GP baseline
    print(f"\n🔧 STANDARD GP:")
    standard_gp = fit_surrogate(final_X_fs_torch, data_y)
    standard_results = evaluate_model_predictions(
        standard_gp, X_test_full, y_test_full,
        feat_idx=final_feat_idx, verbose=True
    )
    
    # 8) Energy-aware stratified ensemble
    print(f"\n🔧 ENERGY-STRATIFIED ENSEMBLE GP:")
    stratified_model = fit_energy_aware_surrogate(
        final_X_fs_torch, data_y, method="stratified"
    )
    stratified_results = evaluate_energy_aware_model(
        stratified_model, X_test_full, y_test_full,
        feat_idx=final_feat_idx, verbose=True
    )
    
    # 9) Compare results
    print(f"\n📊 MODEL COMPARISON:")
    print(f"{'Method':<20} {'MAE':<8} {'RMSE':<8} {'R²':<8}")
    print("-" * 45)
    print(f"{'Standard GP':<20} {standard_results['mae']:<8.3f} {standard_results['rmse']:<8.3f} {standard_results['r2']:<8.3f}")
    print(f"{'Stratified GP':<20} {stratified_results['mae']:<8.3f} {stratified_results['rmse']:<8.3f} {stratified_results['r2']:<8.3f}")
    
    # Calculate improvements
    mae_improvement = (standard_results['mae'] - stratified_results['mae']) / standard_results['mae'] * 100
    rmse_improvement = (standard_results['rmse'] - stratified_results['rmse']) / standard_results['rmse'] * 100
    r2_improvement = (stratified_results['r2'] - standard_results['r2']) / abs(standard_results['r2']) * 100
    
    print(f"\n🎯 ENERGY-AWARE IMPROVEMENTS:")
    print(f"MAE improvement:  {mae_improvement:+.1f}%")
    print(f"RMSE improvement: {rmse_improvement:+.1f}%") 
    print(f"R² improvement:   {r2_improvement:+.1f}%")
    
    # 10) Analyze high-energy performance specifically
    print(f"\n🔥 HIGH-ENERGY PERFORMANCE ANALYSIS:")
    high_energy_mask = y_test_full.numpy() > 0.0  # Focus on unstable materials
    
    if high_energy_mask.sum() > 0:
        print(f"Analyzing {high_energy_mask.sum()} unstable materials (E > 0.0 eV/atom)")
        
        y_true_high = y_test_full.numpy()[high_energy_mask]
        std_pred_high = standard_results['predictions'][high_energy_mask]
        strat_pred_high = stratified_results['predictions'][high_energy_mask]
        
        std_mae_high = np.mean(np.abs(y_true_high - std_pred_high))
        strat_mae_high = np.mean(np.abs(y_true_high - strat_pred_high))
        
        high_energy_improvement = (std_mae_high - strat_mae_high) / std_mae_high * 100
        
        print(f"Standard GP MAE (high-energy):   {std_mae_high:.3f} eV/atom")
        print(f"Stratified GP MAE (high-energy): {strat_mae_high:.3f} eV/atom")
        print(f"High-energy MAE improvement:    {high_energy_improvement:+.1f}%")
    else:
        print("No unstable materials in test set")
    
    print(f"\n✅ ENERGY-AWARE MODELING TEST COMPLETE!")

if __name__ == "__main__":
    main()
