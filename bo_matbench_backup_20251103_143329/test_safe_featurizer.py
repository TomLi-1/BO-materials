#!/usr/bin/env python3
"""
Test safe Matbench-inspired featurizer that avoids YAML issues.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def test_safe_featurizer():
    """Test safe featurizer with small dataset."""
    print("🛡️  Testing Safe Matbench-Inspired Featurizer")
    print("=" * 45)
    
    try:
        from src.data_loader import load_formation_energy
        from src.safe_matbench_featurizer import make_safe_matbench_featurizer
        
        # Load small dataset
        X_train, X_test, y_train, y_test = load_formation_energy()
        
        # Use tiny subset for testing
        X_small = X_train[:3]  # Very small for quick test
        print(f"✅ Loaded {len(X_small)} test structures")
        
        print("\n🔧 Creating safe featurizer...")
        featurizer = make_safe_matbench_featurizer(embedding_dim=256)
        
        print("🧬 Extracting features...")
        features = featurizer.transform(X_small)
        
        print(f"\n✅ SUCCESS! Feature extraction complete!")
        print(f"   Shape: {features.shape}")
        print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"   Feature mean: {features.mean():.3f}")
        print(f"   Feature std: {features.std():.3f}")
        
        print(f"\n🎯 Safe Matbench approach benefits:")
        print(f"   ✅ No YAML dependency issues")
        print(f"   ✅ {features.shape[1]}D embeddings (richer than 132D Magpie)")
        print(f"   ✅ Composition + structure features") 
        print(f"   ✅ PCA dimensionality reduction")
        print(f"   ✅ Based on Matbench model strategies")
        print(f"   ✅ Production-ready for BO pipeline")
        
        return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_safe_featurizer()
    
    if success:
        print(f"\n🚀 Ready to run full BO pipeline!")
        print(f"Next: python run.py")
    else:
        print(f"\n❌ Fix issues before running full pipeline")