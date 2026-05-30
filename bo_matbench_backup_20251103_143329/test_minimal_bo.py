#!/usr/bin/env python3

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("🔍 Testing minimal BO components...")

try:
    import torch
    print("✅ PyTorch imported")
    
    # Test imports
    from src.multi_objective_surrogate import MultiObjectiveParameters, fit_surrogate_target_based
    print("✅ Surrogate imports successful")
    
    from src.multi_objective_bo_loop import bandgap_targeting_bo
    print("✅ BO loop imports successful")
    
    # Create minimal test data
    print("🧪 Creating test data...")
    torch.manual_seed(42)
    
    # Minimal test data (10 samples, 5 features)
    X_pool = torch.randn(10, 5)
    y_pool = torch.rand(10) * 3.0  # Random bandgaps 0-3 eV
    
    # Select 3 initial samples
    X_init = X_pool[:3]
    y_init = y_pool[:3]
    
    print(f"Test data created: {X_pool.shape}, y range [{y_pool.min():.2f}, {y_pool.max():.2f}]")
    
    # Minimal BO parameters
    params = MultiObjectiveParameters(
        bandgap_target=1.5,
        bandgap_tolerance=0.3,
        include_classification=False,
        num_sparsity_feats=3,  # Very small
        total_sample_budget=6,  # Tiny budget: 3 initial + 3 BO
        initialization_budget=3,
        acq_fun="EI_target"
    )
    
    print(f"✅ Parameters created: {params.total_sample_budget - params.initialization_budget} BO iterations")
    
    # Test BO function
    print(f"\n🚀 Testing BO function...")
    
    try:
        data_X, data_y = bandgap_targeting_bo(
            X_init, y_init,
            X_pool, y_pool,
            params
        )
        print(f"✅ BO completed! Found {len(data_y)} samples")
        print(f"Results: y_range = [{data_y.min():.3f}, {data_y.max():.3f}]")
        
    except Exception as bo_error:
        print(f"❌ BO function failed: {bo_error}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ Import or setup failed: {e}")
    import traceback
    traceback.print_exc()