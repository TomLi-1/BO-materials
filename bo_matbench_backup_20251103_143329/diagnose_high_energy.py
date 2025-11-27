#!/usr/bin/env python3
"""
Diagnostic tool to analyze high formation energy prediction failures.
This will help us understand exactly where and why the model fails.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import warnings
from botorch.optim.fit import OptimizationWarning

from src.data_loader import load_formation_energy
from src.featurizer import MatminerTransformer, make_magpie_featurizer
from src.surrogate import fit_surrogate, select_features, evaluate_model_predictions
from src.weighted_surrogate import fit_weighted_surrogate, evaluate_weighted_model
import torch
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=OptimizationWarning)

def analyze_high_energy_failures():
    print("🔍 DIAGNOSING HIGH FORMATION ENERGY PREDICTION FAILURES")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_formation_energy()
    
    # Use subset for faster analysis
    subset_size = 3000
    X_train = X_train[:subset_size]
    X_test = X_test[:500] 
    y_train = y_train[:subset_size]
    y_test = y_test[:500]
    
    print(f"✅ Data: {len(X_train)} train, {len(X_test)} test")
    
    # Analyze energy distribution
    print(f"\n📊 ENERGY DISTRIBUTION ANALYSIS:")
    y_all = np.concatenate([y_train.values, y_test.values])
    
    ranges = [
        ("Very Stable", -np.inf, -2.0),
        ("Stable", -2.0, -0.5),
        ("Marginally Stable", -0.5, 0.0), 
        ("Unstable", 0.0, 1.0),
        ("Very Unstable", 1.0, np.inf)
    ]
    
    for name, min_e, max_e in ranges:
        mask = (y_all >= min_e) & (y_all < max_e)
        count = mask.sum()
        percentage = count / len(y_all) * 100
        print(f"   {name:16}: {count:5d} ({percentage:4.1f}%)")
    
    # Focus on high-energy materials
    high_energy_mask = y_all > 0.0
    print(f"\n🔥 HIGH-ENERGY MATERIALS (E > 0.0 eV/atom):")
    print(f"   Count: {high_energy_mask.sum()}")
    print(f"   Percentage: {high_energy_mask.sum() / len(y_all) * 100:.1f}%")
    print(f"   Energy range: [{y_all[high_energy_mask].min():.3f}, {y_all[high_energy_mask].max():.3f}]")
    print(f"   Mean energy: {y_all[high_energy_mask].mean():.3f} eV/atom")
    
    # Featurize
    print(f"\n🧬 Featurizing...")
    transformer = MatminerTransformer(make_magpie_featurizer())
    X_train_feats = np.array(transformer.transform(X_train))
    X_test_feats = np.array(transformer.transform(X_test))
    
    X_train_t = torch.tensor(X_train_feats, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_feats, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
    
    # Feature selection
    X_selected, feat_idx = select_features(X_train_feats, y_train.values, method="MI", k=30)
    X_selected_t = torch.tensor(X_selected, dtype=torch.float32)
    
    print(f"✅ Selected {len(feat_idx)} features")
    
    # Compare models
    models_to_test = [
        ("Standard GP", lambda: fit_surrogate(X_selected_t, y_train_t)),
        ("Weighted GP (quadratic)", lambda: fit_weighted_surrogate(X_selected_t, y_train_t, weight_strategy="quadratic")),
        ("Weighted GP (threshold)", lambda: fit_weighted_surrogate(X_selected_t, y_train_t, weight_strategy="threshold")),
        ("Weighted GP (exponential)", lambda: fit_weighted_surrogate(X_selected_t, y_train_t, weight_strategy="exponential"))
    ]
    
    results = {}
    
    for model_name, model_func in models_to_test:
        print(f"\n🤖 TESTING {model_name.upper()}:")
        
        try:
            model = model_func()
            
            if "Weighted" in model_name:
                result = evaluate_weighted_model(model, X_test_t, y_test_t, feat_idx=feat_idx, verbose=True)
            else:
                result = evaluate_model_predictions(model, X_test_t, y_test_t, feat_idx=feat_idx, verbose=True)
            
            results[model_name] = result
            
            # Analyze high-energy performance specifically
            high_energy_test_mask = y_test.values > 0.0
            if high_energy_test_mask.sum() > 0:
                y_true_high = y_test.values[high_energy_test_mask]
                y_pred_high = result['predictions'][high_energy_test_mask]
                
                mae_high = np.mean(np.abs(y_true_high - y_pred_high))
                bias_high = np.mean(y_pred_high - y_true_high)  # Negative = underestimation
                
                print(f"   🔥 High-energy performance:")
                print(f"      Samples: {high_energy_test_mask.sum()}")
                print(f"      MAE: {mae_high:.3f} eV/atom")
                print(f"      Bias: {bias_high:.3f} eV/atom {'(underestimation)' if bias_high < 0 else '(overestimation)'}")
                print(f"      True mean: {y_true_high.mean():.3f} eV/atom")
                print(f"      Pred mean: {y_pred_high.mean():.3f} eV/atom")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Summary comparison
    if len(results) > 1:
        print(f"\n📊 MODEL COMPARISON SUMMARY:")
        print(f"{'Model':<25} {'Overall MAE':<12} {'High-E MAE':<12} {'High-E Bias':<12}")
        print("-" * 65)
        
        for model_name, result in results.items():
            overall_mae = result['mae']
            
            high_energy_test_mask = y_test.values > 0.0
            if high_energy_test_mask.sum() > 0:
                y_true_high = y_test.values[high_energy_test_mask]
                y_pred_high = result['predictions'][high_energy_test_mask]
                high_mae = np.mean(np.abs(y_true_high - y_pred_high))
                high_bias = np.mean(y_pred_high - y_true_high)
            else:
                high_mae = high_bias = 0.0
            
            print(f"{model_name:<25} {overall_mae:<12.3f} {high_mae:<12.3f} {high_bias:<12.3f}")
    
    # Find best model for high-energy
    if results:
        best_model = None
        best_high_mae = float('inf')
        
        for model_name, result in results.items():
            high_energy_test_mask = y_test.values > 0.0
            if high_energy_test_mask.sum() > 0:
                y_true_high = y_test.values[high_energy_test_mask]
                y_pred_high = result['predictions'][high_energy_test_mask]
                high_mae = np.mean(np.abs(y_true_high - y_pred_high))
                
                if high_mae < best_high_mae:
                    best_high_mae = high_mae
                    best_model = model_name
        
        if best_model:
            print(f"\n🏆 BEST MODEL FOR HIGH-ENERGY PREDICTIONS: {best_model}")
            print(f"   High-energy MAE: {best_high_mae:.3f} eV/atom")
    
    print(f"\n✅ DIAGNOSTIC ANALYSIS COMPLETE!")

if __name__ == "__main__":
    analyze_high_energy_failures()