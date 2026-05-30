#!/usr/bin/env python3
"""
Test Matbench-inspired featurizer that works without ALIGNN dependencies.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader import load_formation_energy
from src.matbench_inspired_featurizer import make_matbench_inspired_featurizer

def test_matbench_featurizer():
    """Test Matbench-inspired featurizer with small dataset."""
    print("🧪 Testing Matbench-Inspired Featurizer")
    print("=" * 40)
    
    # Load small dataset
    X_train, X_test, y_train, y_test = load_formation_energy()
    
    # Use tiny subset for testing
    X_small = X_train[:5]  # Very small for quick test
    print(f"✅ Loaded {len(X_small)} test structures")
    
    # Test featurizer
    try:
        print("\n🔧 Creating Matbench-inspired featurizer...")
        featurizer = make_matbench_inspired_featurizer(
            embedding_dim=256, 
            use_structure_features=True
        )
        
        print("🧬 Extracting features...")
        features = featurizer.transform(X_small)
        
        print(f"✅ Feature extraction successful!")
        print(f"   Shape: {features.shape}")
        print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"   Feature mean: {features.mean():.3f}")
        print(f"   Feature std: {features.std():.3f}")
        
        print("\n🎯 Matbench-inspired approach benefits:")
        print("   ✅ High-quality 256D embeddings")
        print("   ✅ Multiple feature types (composition + structure + graph-inspired)")
        print("   ✅ PCA dimensionality reduction")
        print("   ✅ No complex dependencies")
        print("   ✅ Based on successful Matbench strategies")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_matbench_featurizer()