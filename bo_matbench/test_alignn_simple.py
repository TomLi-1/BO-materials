#!/usr/bin/env python3
"""
Simple test of ALIGNN featurizer integration.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader import load_formation_energy
from src.alignn_featurizer import make_alignn_featurizer, ALIGNN_AVAILABLE

def test_alignn_featurizer():
    """Test ALIGNN featurizer with small dataset."""
    print("🧪 Testing ALIGNN Featurizer Integration")
    print("=" * 40)
    
    # Check availability
    print(f"ALIGNN Available: {ALIGNN_AVAILABLE}")
    
    # Load small dataset
    X_train, X_test, y_train, y_test = load_formation_energy()
    
    # Use tiny subset for testing
    X_small = X_train[:10]
    print(f"✅ Loaded {len(X_small)} test structures")
    
    # Test featurizer
    try:
        print("\n🔧 Creating ALIGNN featurizer...")
        featurizer = make_alignn_featurizer(embedding_dim=256, use_pretrained=True)
        
        print("🧬 Extracting features...")
        features = featurizer.transform(X_small)
        
        print(f"✅ Feature extraction successful!")
        print(f"   Shape: {features.shape}")
        print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"   Feature mean: {features.mean():.3f}")
        
        if ALIGNN_AVAILABLE:
            print("✅ Using ALIGNN embeddings")
        else:
            print("⚠️  Using fallback (Magpie) features")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nTo fix this, run:")
        print("python setup_alignn.py")

if __name__ == "__main__":
    test_alignn_featurizer()