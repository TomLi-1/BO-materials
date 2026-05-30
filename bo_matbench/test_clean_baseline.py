#!/usr/bin/env python3
"""
Test the clean baseline to make sure it works.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def test_clean_baseline():
    """Test clean baseline BO pipeline."""
    print("🧪 TESTING CLEAN BASELINE")
    print("=" * 30)
    
    try:
        from src.data_loader import load_formation_energy
        from src.featurizer import MatminerTransformer, make_magpie_featurizer
        from src.surrogate import fit_surrogate, evaluate_model_predictions, select_features
        import torch
        import numpy as np
        
        # Very small test
        X_train, X_test, y_train, y_test = load_formation_energy()
        X_small = X_train[:20]
        y_small = y_train[:20]
        
        print(f"✅ Data: {len(X_small)} samples")
        
        # Featurize
        transformer = MatminerTransformer(make_magpie_featurizer())
        X_feats = np.array(transformer.transform(X_small))
        
        print(f"✅ Features: {X_feats.shape}")
        
        # Select features
        X_selected, feat_idx = select_features(X_feats, y_small.values, method="MI", k=15)
        
        # Convert to tensors
        X_t = torch.tensor(X_selected, dtype=torch.float32)
        y_t = torch.tensor(y_small.values, dtype=torch.float32)
        
        print(f"✅ Selected features: {X_selected.shape}")
        
        # Fit GP
        gp = fit_surrogate(X_t, y_t)
        print(f"✅ GP model fitted")
        
        # Test prediction
        results = evaluate_model_predictions(gp, X_t, y_t, feat_idx=None, verbose=False)
        print(f"✅ Evaluation: MAE={results['mae']:.3f}, R²={results['r2']:.3f}")
        
        print(f"\n🎉 CLEAN BASELINE WORKS!")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_clean_baseline()