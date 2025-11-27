#!/usr/bin/env python3
"""
Test enhanced featurizer for improved accuracy.
Compare against basic approaches.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def test_enhanced_vs_basic():
    """Compare enhanced vs basic featurizers for accuracy."""
    print("🎯 ENHANCED FEATURIZER ACCURACY TEST")
    print("=" * 40)
    
    try:
        from src.data_loader import load_formation_energy
        from src.featurizer import MatminerTransformer, make_magpie_featurizer
        from src.enhanced_matbench_featurizer import make_enhanced_matbench_featurizer
        from src.surrogate import fit_surrogate, evaluate_model_predictions, select_features
        import torch
        import numpy as np
        
        # Load small dataset for comparison
        X_train, X_test, y_train, y_test = load_formation_energy()
        
        # Use reasonable subset
        subset_size = 200
        X_train_small = X_train[:subset_size]
        X_test_small = X_test[:50]
        y_train_small = y_train[:subset_size]
        y_test_small = y_test[:50]
        
        print(f"✅ Data: {len(X_train_small)} train, {len(X_test_small)} test")
        print(f"Energy range: [{y_train_small.min():.3f}, {y_train_small.max():.3f}] eV/atom")
        
        # Test 1: Basic Magpie
        print(f"\n🔧 Testing Basic Magpie...")
        magpie_transformer = MatminerTransformer(make_magpie_featurizer())
        X_train_magpie = np.array(magpie_transformer.transform(X_train_small))
        X_test_magpie = np.array(magpie_transformer.transform(X_test_small))
        
        # Feature selection and GP fitting
        X_magpie_selected, feat_idx_magpie = select_features(X_train_magpie, y_train_small.values, method="MI", k=30)
        X_magpie_t = torch.tensor(X_magpie_selected, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_small.values, dtype=torch.float32)
        
        gp_magpie = fit_surrogate(X_magpie_t, y_train_t)
        
        X_test_magpie_t = torch.tensor(X_test_magpie, dtype=torch.float32)
        y_test_t = torch.tensor(y_test_small.values, dtype=torch.float32)
        
        results_magpie = evaluate_model_predictions(
            gp_magpie, X_test_magpie_t, y_test_t, 
            feat_idx=feat_idx_magpie, verbose=False
        )
        
        print(f"   Magpie MAE: {results_magpie['mae']:.3f} eV/atom")
        print(f"   Magpie R²:  {results_magpie['r2']:.3f}")
        
        # Test 2: Enhanced Matbench
        print(f"\n🚀 Testing Enhanced Matbench...")
        enhanced_transformer = make_enhanced_matbench_featurizer(embedding_dim=128)  # Smaller for testing
        X_train_enhanced = enhanced_transformer.transform(X_train_small, fit=True)
        X_test_enhanced = enhanced_transformer.transform(X_test_small, fit=False)
        
        # Feature selection and GP fitting
        X_enhanced_selected, feat_idx_enhanced = select_features(X_train_enhanced, y_train_small.values, method="MI", k=30)
        X_enhanced_t = torch.tensor(X_enhanced_selected, dtype=torch.float32)
        
        gp_enhanced = fit_surrogate(X_enhanced_t, y_train_t)
        
        X_test_enhanced_t = torch.tensor(X_test_enhanced, dtype=torch.float32)
        
        results_enhanced = evaluate_model_predictions(
            gp_enhanced, X_test_enhanced_t, y_test_t, 
            feat_idx=feat_idx_enhanced, verbose=False
        )
        
        print(f"   Enhanced MAE: {results_enhanced['mae']:.3f} eV/atom")
        print(f"   Enhanced R²:  {results_enhanced['r2']:.3f}")
        
        # Compare results
        print(f"\n📊 ACCURACY COMPARISON:")
        print(f"{'Method':<15} {'MAE':<8} {'R²':<8} {'Features':<10}")
        print("-" * 45)
        print(f"{'Basic Magpie':<15} {results_magpie['mae']:<8.3f} {results_magpie['r2']:<8.3f} {X_train_magpie.shape[1]:<10}")
        print(f"{'Enhanced':<15} {results_enhanced['mae']:<8.3f} {results_enhanced['r2']:<8.3f} {X_train_enhanced.shape[1]:<10}")
        
        # Calculate improvement
        mae_improvement = (results_magpie['mae'] - results_enhanced['mae']) / results_magpie['mae'] * 100
        r2_improvement = (results_enhanced['r2'] - results_magpie['r2']) / abs(results_magpie['r2']) * 100
        
        print(f"\n🎯 IMPROVEMENT:")
        print(f"MAE improvement:  {mae_improvement:+.1f}%")
        print(f"R² improvement:   {r2_improvement:+.1f}%")
        
        if mae_improvement > 5:
            print(f"✅ Significant improvement! Enhanced approach is better.")
        elif mae_improvement > 0:
            print(f"✅ Modest improvement. Enhanced approach is slightly better.")
        else:
            print(f"⚠️  No improvement. May need larger dataset or different approach.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_vs_basic()
    
    if success:
        print(f"\n🚀 Use featurizer_type='enhanced' in run.py for better accuracy!")
    else:
        print(f"\n❌ Fix issues before using enhanced approach")