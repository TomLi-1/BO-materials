#!/usr/bin/env python3
"""
Quick test to verify GNN implementation works in bandgap targeting context
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import warnings
import torch
from botorch.optim.fit import OptimizationWarning

# Import modules  
from src.multi_objective_data_loader import load_bandgap_dataset
from src.gnn_featurizer import make_gnn_featurizer
from src.multi_objective_surrogate import MultiObjectiveParameters
from src.multi_objective_bo_loop import bandgap_targeting_bo, evaluate_target_achievement

warnings.filterwarnings("ignore", category=OptimizationWarning)

def main():
    print("🎯 GNN BANDGAP TARGETING TEST")
    print("=" * 40)
    
    # Configuration
    bandgap_target = 1.5
    bandgap_tolerance = 0.2
    
    try:
        # 1) Load small dataset
        print("📊 Loading dataset...")
        X_train, X_test, y_train, y_test, _, _ = load_bandgap_dataset()
        
        # Take very small subset for quick test
        X_train = X_train[:50]
        X_test = X_test[:30] 
        y_train = y_train[:50]
        y_test = y_test[:30]
        
        print(f"✅ Loaded {len(X_train) + len(X_test)} samples")
        
        # 2) Featurize with GNN
        print("🧬 GNN Featurization...")
        transformer = make_gnn_featurizer(embedding_dim=16, pooling_strategy="max")
        
        X_train_feats = transformer.transform(X_train)
        X_test_feats = transformer.transform(X_test)
        print(f"✅ Features: {X_train_feats.shape[1]}D")
        
        # 3) Convert to tensors
        X_train_t = torch.tensor(X_train_feats, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_feats, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
        
        # 4) Pool data
        X_pool = torch.cat([X_train_t, X_test_t], dim=0)
        y_pool = torch.cat([y_train_t, y_test_t], dim=0)
        
        print(f"📊 Pool: {X_pool.shape}, target range check...")
        
        # Check target materials
        in_target = ((y_pool >= bandgap_target - bandgap_tolerance) & 
                    (y_pool <= bandgap_target + bandgap_tolerance)).sum().item()
        print(f"   Target materials: {in_target}/{len(y_pool)} ({in_target/len(y_pool):.1%})")
        
        # 5) Quick BO test
        print("🤖 Quick BO test...")
        params = MultiObjectiveParameters(
            dataset_name="test",
            bandgap_target=bandgap_target,
            bandgap_tolerance=bandgap_tolerance, 
            include_classification=False,
            sparsity_method="MI",
            num_sparsity_feats=10,
            total_sample_budget=15,
            initialization_budget=5,
            acq_fun="EI_target"
        )
        
        # Initialize
        init_idx = torch.randperm(len(X_pool))[:params.initialization_budget]
        X_init = X_pool[init_idx]
        y_init = y_pool[init_idx]
        
        # Run mini BO
        data_X, data_y = bandgap_targeting_bo(X_init, y_init, X_pool, y_pool, params)
        
        # Results
        achievement = evaluate_target_achievement(data_y, bandgap_target, bandgap_tolerance)
        
        print(f"\n🎯 RESULTS:")
        print(f"   Success rate: {achievement['success_rate']:.1%}")
        print(f"   Best found: {achievement['best_bandgap']:.3f} eV") 
        print(f"   Target distance: {achievement['min_distance']:.3f} eV")
        
        print(f"\n✅ GNN IMPLEMENTATION WORKING!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()