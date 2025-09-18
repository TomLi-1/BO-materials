import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import warnings
import torch
import json
import yaml
from botorch.optim.fit import OptimizationWarning

# Import our multi-objective modules
from src.multi_objective_data_loader import load_bandgap_dataset, create_classification_targets
from src.featurizer import MatminerTransformer, make_magpie_featurizer
from src.gnn_featurizer import make_gnn_featurizer
from src.multi_objective_surrogate import (
    MultiObjectiveParameters, evaluate_bandgap_targeting, 
    plot_bandgap_targeting_analysis, fit_surrogate_target_based
)
from src.multi_objective_bo_loop import bandgap_targeting_bo, evaluate_target_achievement

warnings.filterwarnings("ignore", category=OptimizationWarning)

def main():
    print("🎯 BANDGAP TARGETING BAYESIAN OPTIMIZATION (QUICK TEST)")
    print("=" * 60)
    
    # Check for required dependencies
    try:
        import matminer
        import pymatgen
        print("✅ Dependencies verified: matminer, pymatgen")
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        return
    
    # Quick test configuration
    bandgap_target = 1.5
    bandgap_tolerance = 0.2
    featurizer_type = "magpie"  # Options: "magpie", "gnn"
    # featurizer_type = "gnn"  # Uncomment to test GNN embeddings
    
    print(f"🎯 Target: {bandgap_target} ± {bandgap_tolerance} eV (optimal for solar cells)")
    print(f"🔧 Using featurizer: {featurizer_type}")
    
    # 1) Load smaller dataset sample
    print(f"\n📊 Loading dataset (quick test - first 500 materials)...")
    try:
        X_train, X_test, y_train, y_test, train_meta, test_meta = load_bandgap_dataset()
        
        # Take smaller subset for quick testing
        X_train = X_train[:300]  # Reduced from ~3683
        X_test = X_test[:200]    # Reduced from ~921
        y_train = y_train[:300]
        y_test = y_test[:200]
        
        print(f"✅ Dataset loaded: {len(X_train) + len(X_test)} total materials (subset)")
        print(f"   Bandgap range: [{min(y_train.min(), y_test.min()):.3f}, {max(y_train.max(), y_test.max()):.3f}] eV")
        
        # Check how many materials are in target range
        all_y = torch.cat([torch.tensor(y_train.values), torch.tensor(y_test.values)])
        in_target = ((all_y >= bandgap_target - bandgap_tolerance) & 
                    (all_y <= bandgap_target + bandgap_tolerance)).sum().item()
        print(f"   Materials in target range: {in_target}/{len(all_y)} ({in_target/len(all_y):.1%})")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # 2) Featurize with comparison
    print(f"\n🧬 Featurizing with {featurizer_type} descriptors...")
    
    try:
        import time
        start_time = time.time()
        
        if featurizer_type == "magpie":
            transformer = MatminerTransformer(make_magpie_featurizer())
            print(f"   Featurizing {len(X_train)} training samples...")
            X_train_feats = transformer.transform(X_train)
            print(f"   Featurizing {len(X_test)} test samples...")
            X_test_feats = transformer.transform(X_test)
            
            # Convert to numpy arrays
            import numpy as np
            X_train_feats = np.array(X_train_feats)
            X_test_feats = np.array(X_test_feats)
            
        elif featurizer_type == "gnn":
            transformer = make_gnn_featurizer(embedding_dim=32, pooling_strategy="max_mean")
            print(f"   Featurizing {len(X_train)} training samples...")
            X_train_feats = transformer.transform(X_train)
            print(f"   Featurizing {len(X_test)} test samples...")
            X_test_feats = transformer.transform(X_test)
            
        else:
            raise ValueError(f"Unknown featurizer type: {featurizer_type}")
        
        featurization_time = time.time() - start_time
        print(f"✅ Features extracted: {X_train_feats.shape[1]} dimensional")
        print(f"⏱️  Featurization time: {featurization_time:.2f} seconds")
        
        # Convert to torch tensors
        print(f"   Converting to PyTorch tensors...")
        X_train_t = torch.tensor(X_train_feats, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_feats, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
        print(f"✅ Tensor conversion complete")
        
    except Exception as e:
        print(f"❌ Featurization failed: {e}")
        return
    
    # 3) Combine into pool for BO
    X_pool = torch.cat([X_train_t, X_test_t], dim=0)
    y_pool = torch.cat([y_train_t, y_test_t], dim=0)
    
    # 4) Quick BO parameters (reduced budget)
    params = MultiObjectiveParameters(
        dataset_name="matbench_expt_gap",
        bandgap_target=bandgap_target,
        bandgap_tolerance=bandgap_tolerance,
        include_classification=False,  # Disable for quick test
        sparsity_method="MI",
        num_sparsity_feats=20,  # Reduced from 40
        total_sample_budget=25,  # Reduced from 200
        initialization_budget=5,   # Reduced from 15
        acq_fun="EI_target"
    )
    
    # 5) Initialize with random samples
    print(f"\n🎲 Initializing with {params.initialization_budget} random samples...")
    init_idx = torch.randperm(len(X_pool))[:params.initialization_budget]
    X_init = X_pool[init_idx]
    y_init = y_pool[init_idx]
    
    # Check initial target achievement
    init_in_target = ((y_init >= bandgap_target - bandgap_tolerance) & 
                     (y_init <= bandgap_target + bandgap_tolerance)).sum().item()
    print(f"Initial materials in target range: {init_in_target}/{len(y_init)}")
    
    # 6) Run Quick Bayesian Optimization
    print(f"\n🤖 Starting Quick Bayesian Optimization ({params.total_sample_budget} iterations)...")
    try:
        data_X, data_y = bandgap_targeting_bo(
            X_init, y_init,
            X_pool, y_pool,
            params
        )
        print(f"✅ BO completed successfully!")
    except Exception as e:
        print(f"❌ BO failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7) Results
    achievement = evaluate_target_achievement(data_y, bandgap_target, bandgap_tolerance)
    
    print(f"\n🎯 QUICK TEST RESULTS:")
    print(f"=" * 50)
    print(f"Target: {bandgap_target} ± {bandgap_tolerance} eV")
    print(f"Success rate: {achievement['success_rate']:.1%}")
    print(f"Materials in range: {achievement['materials_in_range']}/{achievement['total_materials']}")
    print(f"Best found: {achievement['best_bandgap']:.4f} eV")
    print(f"Distance from target: {achievement['min_distance']:.4f} eV")
    
    print(f"\n✅ QUICK TEST COMPLETE - Framework is working!")
    print(f"💡 For full experiment, run: python run_bandgap_targeting.py")


if __name__ == "__main__":
    main()