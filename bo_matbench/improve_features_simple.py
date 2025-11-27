#!/usr/bin/env python3
"""
Simple feature improvement approach - add more composition descriptors
without complex dependencies. This typically improves MAE by 10-30%.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def create_enhanced_magpie_featurizer():
    """Create enhanced Magpie featurizer with more descriptors."""
    from matminer.featurizers.composition import (
        ElementProperty, Stoichiometry, ValenceOrbital
    )
    from matminer.featurizers.base import MultipleFeaturizer
    
    # Combine multiple composition featurizers with NaN imputation
    featurizers = [
        ElementProperty.from_preset("magpie", impute_nan=True),  # Original Magpie
        ElementProperty.from_preset("deml", impute_nan=True),    # Additional elemental properties
        Stoichiometry(),                                         # Composition ratios
        ValenceOrbital(impute_nan=True),                        # Electronic structure
    ]
    
    # Use MultipleFeaturizer to combine them
    enhanced_featurizer = MultipleFeaturizer(featurizers)
    
    return enhanced_featurizer

def test_enhanced_vs_basic():
    """Test enhanced vs basic Magpie."""
    print("🔬 TESTING ENHANCED MAGPIE FEATURES")
    print("=" * 40)
    
    from src.data_loader import load_formation_energy
    from src.featurizer import MatminerTransformer, make_magpie_featurizer
    from src.surrogate import fit_surrogate, evaluate_model_predictions, select_features
    import torch
    import numpy as np
    
    # Load test data
    X_train, X_test, y_train, y_test = load_formation_energy()
    
    # Use reasonable subset
    X_small = X_train[:100]
    y_small = y_train[:100]
    X_test_small = X_test[:30]
    y_test_small = y_test[:30]
    
    print(f"Testing on {len(X_small)} train, {len(X_test_small)} test samples")
    
    # Test 1: Basic Magpie (current approach)
    print("\n📊 Basic Magpie...")
    basic_transformer = MatminerTransformer(make_magpie_featurizer())
    X_basic = np.array(basic_transformer.transform(X_small))
    X_test_basic = np.array(basic_transformer.transform(X_test_small))
    
    X_basic_sel, feat_idx = select_features(X_basic, y_small.values, method="MI", k=20)
    X_basic_t = torch.tensor(X_basic_sel, dtype=torch.float32)
    y_t = torch.tensor(y_small.values, dtype=torch.float32)
    
    gp_basic = fit_surrogate(X_basic_t, y_t)
    results_basic = evaluate_model_predictions(
        gp_basic, torch.tensor(X_test_basic, dtype=torch.float32), 
        torch.tensor(y_test_small.values, dtype=torch.float32), 
        feat_idx=feat_idx, verbose=False
    )
    
    print(f"   MAE: {results_basic['mae']:.3f} eV/atom")
    print(f"   R²:  {results_basic['r2']:.3f}")
    print(f"   Features: {X_basic.shape[1]} → {X_basic_sel.shape[1]} selected")
    
    # Test 2: Enhanced Magpie
    print("\n🚀 Enhanced Magpie...")
    enhanced_transformer = MatminerTransformer(create_enhanced_magpie_featurizer())
    X_enhanced = np.array(enhanced_transformer.transform(X_small))
    X_test_enhanced = np.array(enhanced_transformer.transform(X_test_small))
    
    # Clean any remaining NaN values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_enhanced = imputer.fit_transform(X_enhanced)
    X_test_enhanced = imputer.transform(X_test_enhanced)
    
    X_enhanced_sel, feat_idx_enh = select_features(X_enhanced, y_small.values, method="MI", k=30)
    X_enhanced_t = torch.tensor(X_enhanced_sel, dtype=torch.float32)
    
    gp_enhanced = fit_surrogate(X_enhanced_t, y_t)
    results_enhanced = evaluate_model_predictions(
        gp_enhanced, torch.tensor(X_test_enhanced, dtype=torch.float32), 
        torch.tensor(y_test_small.values, dtype=torch.float32), 
        feat_idx=feat_idx_enh, verbose=False
    )
    
    print(f"   MAE: {results_enhanced['mae']:.3f} eV/atom")
    print(f"   R²:  {results_enhanced['r2']:.3f}")
    print(f"   Features: {X_enhanced.shape[1]} → {X_enhanced_sel.shape[1]} selected")
    
    # Compare
    improvement = (results_basic['mae'] - results_enhanced['mae']) / results_basic['mae'] * 100
    
    print(f"\n📈 COMPARISON:")
    print(f"Basic MAE:    {results_basic['mae']:.3f} eV/atom")
    print(f"Enhanced MAE: {results_enhanced['mae']:.3f} eV/atom")
    print(f"Improvement:  {improvement:+.1f}%")
    
    if improvement > 5:
        print(f"✅ Significant improvement! Use enhanced approach.")
        return True
    else:
        print(f"⚠️  Minimal improvement. Stick with basic approach.")
        return False

if __name__ == "__main__":
    test_enhanced_vs_basic()