#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("Testing bandgap data loading...")

try:
    from src.multi_objective_data_loader import load_bandgap_dataset
    print("✅ Import successful")
    
    print("\n🔍 Loading dataset...")
    X_train, X_test, y_train, y_test, train_meta, test_meta = load_bandgap_dataset()
    
    print(f"\n📊 Dataset info:")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Train composition type: {type(X_train[0])}")
    print(f"   Sample composition: {X_train[0]}")
    
    # Test featurization
    print(f"\n🧬 Testing featurization...")
    from src.featurizer import MatminerTransformer, make_magpie_featurizer
    
    transformer = MatminerTransformer(make_magpie_featurizer())
    print(f"✅ Transformer created")
    
    # Try with just first few samples
    print(f"🔬 Testing on first 3 samples...")
    test_comps = X_train[:3]
    print(f"   Sample composition types: {[type(comp) for comp in test_comps]}")
    
    try:
        features = transformer.transform(test_comps)
        print(f"✅ Featurization successful! Type: {type(features)}")
        
        # Check if features are valid
        import numpy as np
        features_array = np.array(features)
        print(f"   Feature array shape: {features_array.shape}")
        print(f"   Feature range: [{features_array.min():.3f}, {features_array.max():.3f}]")
        print(f"   Any NaN values: {np.isnan(features_array).any()}")
        
    except Exception as feat_error:
        print(f"❌ Featurization error: {feat_error}")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()