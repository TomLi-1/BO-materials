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
from src.multi_objective_surrogate import (
    MultiObjectiveParameters, evaluate_bandgap_targeting, 
    plot_bandgap_targeting_analysis, fit_surrogate_target_based
)
from src.multi_objective_bo_loop import bandgap_targeting_bo, evaluate_target_achievement

warnings.filterwarnings("ignore", category=OptimizationWarning)

def main():
    print("🎯 BANDGAP TARGETING BAYESIAN OPTIMIZATION")
    print("=" * 60)
    
    # Check for required dependencies
    try:
        import matminer
        import pymatgen
        print("✅ Dependencies verified: matminer, pymatgen")
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("\n💡 To install missing dependencies:")
        print("   pip install matminer>=0.9.0 pymatgen>=2023.0.0")
        print("   OR")
        print("   conda install -c conda-forge matminer pymatgen")
        print(f"\n📖 See SETUP_INSTRUCTIONS.md for detailed installation guide")
        return
    
    # Load configuration
    config_path = os.path.join("src", "multi_objective_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded configuration from {config_path}")
    else:
        config = {}
        print("⚠️ Using default configuration")
    
    # Extract target bandgap from config
    bandgap_target = config.get('bandgap_target', {}).get('value', 1.5)
    bandgap_tolerance = config.get('bandgap_target', {}).get('tolerance', 0.2)
    
    print(f"🎯 Target: {bandgap_target} ± {bandgap_tolerance} eV (optimal for solar cells)")
    
    # 1) Load bandgap dataset
    print(f"\n📊 Loading dataset...")
    try:
        dataset_name = config.get('dataset', {}).get('primary', {}).get('name', 'matbench_expt_gap')
        X_train, X_test, y_train, y_test, train_meta, test_meta = load_bandgap_dataset(
            dataset_name=dataset_name
        )
        print(f"✅ Dataset loaded: {len(X_train) + len(X_test)} total materials")
        print(f"   Bandgap range: [{min(y_train.min(), y_test.min()):.3f}, {max(y_train.max(), y_test.max()):.3f}] eV")
        
        # Check how many materials are in target range
        all_y = torch.cat([torch.tensor(y_train.values), torch.tensor(y_test.values)])
        in_target = ((all_y >= bandgap_target - bandgap_tolerance) & 
                    (all_y <= bandgap_target + bandgap_tolerance)).sum().item()
        print(f"   Materials in target range: {in_target}/{len(all_y)} ({in_target/len(all_y):.1%})")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Make sure matminer is installed: pip install matminer")
        return
    
    # 2) Featurize using Magpie descriptors
    print(f"\n🧬 Featurizing with Magpie descriptors...")
    transformer = MatminerTransformer(make_magpie_featurizer())
    
    try:
        print(f"   Featurizing {len(X_train)} training samples...")
        X_train_feats = transformer.transform(X_train)
        print(f"   Featurizing {len(X_test)} test samples...")
        X_test_feats = transformer.transform(X_test)
        
        # Convert to numpy arrays first
        import numpy as np
        print(f"   Converting to numpy arrays...")
        X_train_feats = np.array(X_train_feats)
        X_test_feats = np.array(X_test_feats)
        print(f"✅ Features extracted: {X_train_feats.shape[1]} dimensional")
        
        # Convert to torch tensors
        print(f"   Converting to PyTorch tensors...")
        X_train_t = torch.tensor(X_train_feats, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_feats, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
        print(f"✅ Tensor conversion complete")
        
    except Exception as e:
        print(f"❌ Featurization failed: {e}")
        print("💡 This might take a while for large datasets...")
        return
    
    # 3) Handle classification targets if enabled
    y_class_train_t = None
    y_class_test_t = None
    
    if config.get('dataset', {}).get('secondary', {}).get('enabled', False):
        print(f"\n🏷️ Setting up classification targets...")
        try:
            class_task = config.get('dataset', {}).get('secondary', {}).get('name', 'spacegroup_classification')
            if 'spacegroup' in class_task.lower():
                task_type = 'spacegroup'
            else:
                task_type = 'crystal_system'
                
            if train_meta and test_meta:
                train_labels, train_mapping = create_classification_targets(train_meta, task_type)
                test_labels, test_mapping = create_classification_targets(test_meta, task_type)
                
                y_class_train_t = torch.tensor(train_labels, dtype=torch.long)
                y_class_test_t = torch.tensor(test_labels, dtype=torch.long)
                
                print(f"✅ {task_type} classification: {train_mapping['num_classes']} classes")
            else:
                print("⚠️ No structural metadata available for classification")
        except Exception as e:
            print(f"⚠️ Classification setup failed: {e}")
    
    # 4) Combine into pool for BO
    X_pool = torch.cat([X_train_t, X_test_t], dim=0)
    y_pool = torch.cat([y_train_t, y_test_t], dim=0)
    
    if y_class_train_t is not None and y_class_test_t is not None:
        y_class_pool = torch.cat([y_class_train_t, y_class_test_t], dim=0)
    else:
        y_class_pool = None
    
    # 5) BO parameters
    params = MultiObjectiveParameters(
        dataset_name=dataset_name,
        bandgap_target=bandgap_target,
        bandgap_tolerance=bandgap_tolerance,
        include_classification=y_class_pool is not None,
        classification_weight=config.get('dataset', {}).get('secondary', {}).get('priority', 0.3),
        sparsity_method=config.get('bo', {}).get('feature_selection', {}).get('method', 'MI'),
        num_sparsity_feats=config.get('bo', {}).get('feature_selection', {}).get('num_features', 40),
        total_sample_budget=config.get('bo', {}).get('total_sample_budget', 200),
        initialization_budget=config.get('bo', {}).get('initialization_budget', 15),
        acq_fun="EI_target"
    )
    
    # 6) Initialize with random samples
    print(f"\n🎲 Initializing with {params.initialization_budget} random samples...")
    init_idx = torch.randperm(len(X_pool))[:params.initialization_budget]
    X_init = X_pool[init_idx]
    y_init = y_pool[init_idx]
    y_class_init = y_class_pool[init_idx] if y_class_pool is not None else None
    
    # Check initial target achievement
    init_in_target = ((y_init >= bandgap_target - bandgap_tolerance) & 
                     (y_init <= bandgap_target + bandgap_tolerance)).sum().item()
    print(f"Initial materials in target range: {init_in_target}/{len(y_init)}")
    
    # 7) Run Bayesian Optimization
    print(f"\n🤖 Starting Bayesian Optimization...")
    data_X, data_y = bandgap_targeting_bo(
        X_init, y_init,
        X_pool, y_pool,
        params,
        y_class_init, y_class_pool
    )
    
    # 8) Evaluate target achievement
    achievement = evaluate_target_achievement(data_y, bandgap_target, bandgap_tolerance)
    
    print(f"\n🎯 BANDGAP TARGETING RESULTS:")
    print(f"=" * 50)
    print(f"Target: {bandgap_target} ± {bandgap_tolerance} eV")
    print(f"Success rate: {achievement['success_rate']:.1%}")
    print(f"Materials in range: {achievement['materials_in_range']}/{achievement['total_materials']}")
    print(f"Best found: {achievement['best_bandgap']:.4f} eV")
    print(f"Distance from target: {achievement['min_distance']:.4f} eV")
    
    # 9) Model evaluation on held-out test data
    print(f"\n📊 EVALUATING MODEL PERFORMANCE...")
    print(f"=" * 50)
    
    # Create test set from remaining pool
    train_mask = torch.zeros(len(X_pool), dtype=torch.bool)
    for i, x_train in enumerate(data_X):
        distances = torch.norm(X_pool - x_train.unsqueeze(0), dim=1)
        train_mask[torch.argmin(distances)] = True
    
    test_mask = ~train_mask
    X_test_final = X_pool[test_mask]
    y_test_final = y_pool[test_mask]
    
    if len(X_test_final) > 0:
        # Use final feature selection for evaluation
        from src.multi_objective_surrogate import select_features_multi_objective
        final_X_fs, final_feat_idx = select_features_multi_objective(
            data_X.numpy(), data_y.numpy(),
            method=params.sparsity_method, k=params.num_sparsity_feats
        )
        final_X_fs_torch = torch.tensor(final_X_fs, dtype=torch.float32)
        
        # Fit final model
        final_gp = fit_surrogate_target_based(final_X_fs_torch, data_y, bandgap_target)
        
        # Evaluate
        results = evaluate_bandgap_targeting(
            final_gp, X_test_final, y_test_final,
            target_value=bandgap_target, tolerance=bandgap_tolerance,
            feat_idx=final_feat_idx, verbose=True
        )
        
        # Create visualization
        plot_save_path = os.path.join("src", "bandgap_targeting_evaluation.png")
        plot_bandgap_targeting_analysis(results, save_path=plot_save_path)
        
    else:
        print("⚠️ No test data available (all data used in BO)")
        results = {'mae': 'N/A', 'rmse': 'N/A', 'r2': 'N/A', 'target_f1': 'N/A'}
    
    # 10) Save results
    experiment_results = {
        'date': '2025-09-06',
        'experiment_type': 'bandgap_targeting',
        'dataset': dataset_name,
        'target_config': {
            'bandgap_target': bandgap_target,
            'tolerance': bandgap_tolerance,
        },
        'dataset_stats': {
            'total_materials': len(X_pool),
            'bandgap_range': [y_pool.min().item(), y_pool.max().item()],
            'materials_in_target_range': in_target,
            'target_percentage': in_target / len(all_y)
        },
        'bo_config': {
            'total_budget': params.total_sample_budget,
            'initialization_budget': params.initialization_budget,
            'acquisition_function': params.acq_fun,
            'feature_selection': params.sparsity_method,
            'num_features': params.num_sparsity_feats,
            'include_classification': params.include_classification
        },
        'results': {
            'success_rate': achievement['success_rate'],
            'materials_found_in_range': achievement['materials_in_range'],
            'best_bandgap_found': achievement['best_bandgap'],
            'min_distance_from_target': achievement['min_distance'],
            'mean_distance_from_target': achievement['mean_distance']
        },
        'model_evaluation': {
            'test_mae': results.get('mae', 'N/A'),
            'test_rmse': results.get('rmse', 'N/A'),
            'test_r2': results.get('r2', 'N/A'),
            'target_detection_f1': results.get('target_f1', 'N/A'),
            'test_samples': len(X_test_final) if len(X_test_final) > 0 else 0
        }
    }
    
    # Save JSON results
    results_file = "bandgap_targeting_results.json"
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Save summary
    summary_file = "bandgap_targeting_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("BANDGAP TARGETING BO EXPERIMENT SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Target: {bandgap_target} ± {bandgap_tolerance} eV\n")
        f.write(f"Dataset: {dataset_name} ({len(X_pool)} materials)\n")
        f.write(f"Success Rate: {achievement['success_rate']:.1%}\n")
        f.write(f"Materials Found in Range: {achievement['materials_in_range']}/{achievement['total_materials']}\n")
        f.write(f"Best Material: {achievement['best_bandgap']:.4f} eV\n")
        f.write(f"Model Test MAE: {results.get('mae', 'N/A')}\n")
        f.write(f"Target Detection F1: {results.get('target_f1', 'N/A')}\n")
    
    print(f"📄 Summary saved to: {summary_file}")
    print(f"\n✅ BANDGAP TARGETING EXPERIMENT COMPLETE!")


if __name__ == "__main__":
    main()