import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import warnings
from botorch.optim.fit import OptimizationWarning

from src.data_loader import load_formation_energy
from src.featurizer import MatminerTransformer, make_magpie_featurizer
from src.gnn_featurizer import GNNTransformer, make_gnn_featurizer
from src.surrogate import OptimizerParameters, evaluate_model_predictions, plot_prediction_analysis, fit_surrogate, fit_surrogate_with_scaling, fit_robust_surrogate, evaluate_robust_model, select_features
from src.bo_loop import bayes_optimize
import torch
import time

warnings.filterwarnings("ignore", category=OptimizationWarning)

def main():
    # Configuration for featurizer type
    featurizer_type = "magpie"  # Options: "magpie", "gnn"
    # featurizer_type = "gnn"  # Uncomment to test GNN embeddings
    
    # Configuration for robust modeling
    use_robust_modeling = True  # Use robust surrogate to handle high formation energies
    
    print(f"🔧 Using featurizer: {featurizer_type}")
    print(f"🔧 Robust modeling: {use_robust_modeling}")
    
    # 1) Load Matbench formation-energy data
    X_train, X_test, y_train, y_test = load_formation_energy()

    # 2) Featurize with timing comparison
    print(f"\n🧬 Featurizing {len(X_train)} train + {len(X_test)} test samples...")
    
    start_time = time.time()
    if featurizer_type == "magpie":
        transformer = MatminerTransformer(make_magpie_featurizer())
        X_train_feats = transformer.transform(X_train)
        X_test_feats = transformer.transform(X_test)
        feature_dim = X_train_feats.shape[1]
        print(f"✅ Magpie features extracted: {feature_dim} dimensions")
        
    elif featurizer_type == "gnn":
        transformer = make_gnn_featurizer(embedding_dim=64, pooling_strategy="max_mean")
        X_train_feats = transformer.transform(X_train)
        X_test_feats = transformer.transform(X_test)
        feature_dim = X_train_feats.shape[1]
        print(f"✅ GNN features extracted: {feature_dim} dimensions")
        
    else:
        raise ValueError(f"Unknown featurizer type: {featurizer_type}")
    
    featurization_time = time.time() - start_time
    print(f"⏱️  Featurization time: {featurization_time:.2f} seconds")
    
    # to torch Tensors:
    X_train_t = torch.tensor(X_train_feats, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_feats, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test.values, dtype=torch.float32)

    # 3) Combine into a single pool
    X_pool = torch.cat([X_train_t, X_test_t], dim=0)
    y_pool = torch.cat([y_train_t,   y_test_t],   dim=0)

    # 4) BO parameters
    params = OptimizerParameters(
        sparsity_method="MI",
        acq_fun="EI",
        num_sparsity_feats=30,
        initialization_budget=10,
        total_sample_budget=200,
    )

    # 5) Pick an initial random seed set
    init_idx = torch.randperm(len(X_pool))[: params.initialization_budget]
    X_init   = X_pool[init_idx]
    y_init   = y_pool[init_idx]

    # 6) Run BO
    data_X, data_y = bayes_optimize(
        X_init, y_init,
        X_pool, y_pool,
        params
    )

    print(f"\nFOUND!! Most stable material (lowest formation energy) = {data_y.min().item():.4f} eV/atom")
    
    # Show ground truth statistics for manual comparison
    print(f"\n" + "="*50)
    print("GROUND TRUTH ANALYSIS")
    print("="*50)
    
    print(f"📊 Overall Dataset Statistics:")
    print(f"   Total samples: {len(y_pool)}")
    print(f"   Ground truth range: [{y_pool.min().item():.4f}, {y_pool.max().item():.4f}] eV/atom")
    print(f"   Ground truth mean: {y_pool.mean().item():.4f} eV/atom")
    print(f"   Ground truth std: {y_pool.std().item():.4f} eV/atom")
    
    print(f"\n🎯 Current BO Target (MINIMIZING formation energy for STABLE materials):")
    print(f"   BO found most stable: {data_y.min().item():.4f} eV/atom")
    print(f"   True global minimum: {y_pool.min().item():.4f} eV/atom")
    print(f"   Gap to global min: {data_y.min().item() - y_pool.min().item():.4f} eV/atom")
    
    print(f"\n✅ PHYSICS CORRECT: Formation Energy Interpretation")
    print(f"   Negative values (-): STABLE materials (thermodynamically favored)")
    print(f"   Positive values (+): UNSTABLE materials (decompose spontaneously)")
    print(f"   Your BO found: {data_y.min().item():.4f} eV/atom → {'STABLE' if data_y.min().item() < 0 else 'UNSTABLE'} material")
    
    print(f"\n🏆 TOP 5 Most Stable Materials (LOWEST formation energy - BO target):")
    sorted_y_stable, sorted_indices_stable = torch.sort(y_pool, descending=False)
    for i in range(min(5, len(sorted_y_stable))):
        stability = "VERY STABLE" if sorted_y_stable[i].item() < -2 else "STABLE" if sorted_y_stable[i].item() < 0 else "UNSTABLE"
        print(f"   Rank {i+1}: {sorted_y_stable[i].item():.4f} eV/atom (index {sorted_indices_stable[i].item()}) - {stability}")
    
    print(f"\n📊 BO Performance Analysis:")
    best_possible = sorted_y_stable[0].item()
    bo_found = data_y.min().item()
    if bo_found <= best_possible + 0.01:
        print(f"   🎉 EXCELLENT: BO found the global optimum!")
    else:
        gap_pct = ((bo_found - best_possible) / abs(best_possible)) * 100 if best_possible != 0 else 0
        print(f"   📈 Gap to optimum: {bo_found - best_possible:.4f} eV/atom ({gap_pct:.1f}%)")
        
    # Show where BO result ranks globally
    bo_rank = (sorted_y_stable < bo_found).sum().item() + 1
    print(f"   🥇 BO result ranks #{bo_rank} out of {len(y_pool)} materials")
    
    # Record results for experiment log
    experiment_results = {
        'date': '2025-09-06',
        'objective': 'minimize_formation_energy', 
        'dataset_size': len(y_pool),
        'data_range': [y_pool.min().item(), y_pool.max().item()],
        'data_mean': y_pool.mean().item(),
        'data_std': y_pool.std().item(),
        'bo_found_best': bo_found,
        'true_global_min': best_possible,
        'gap_to_optimum': bo_found - best_possible,
        'gap_percentage': gap_pct,
        'global_ranking': bo_rank,
        'total_materials': len(y_pool),
        'bo_config': {
            'initialization_budget': params.initialization_budget,
            'total_sample_budget': params.total_sample_budget,
            'acquisition_function': params.acq_fun,
            'sparsity_method': params.sparsity_method,
            'num_sparsity_feats': params.num_sparsity_feats
        }
    }
    
    # Now let's evaluate the final model on held-out test data
    print("\n" + "="*60)
    print("EVALUATING FINAL MODEL ON TEST DATA")
    print("="*60)
    
    # Use the final selected features for evaluation
    final_X_fs, final_feat_idx = select_features(
        data_X.numpy(),
        data_y.numpy(), 
        method=params.sparsity_method,
        k=params.num_sparsity_feats
    )
    final_X_fs_torch = torch.tensor(final_X_fs, dtype=torch.float32)
    
    # Fit final GP model with robust option
    if use_robust_modeling:
        print(f"🔧 Using robust surrogate modeling...")
        final_gp, robust_info = fit_robust_surrogate(final_X_fs_torch, data_y)
    else:
        print(f"🔧 Using standard surrogate modeling...")
        final_gp = fit_surrogate(final_X_fs_torch, data_y)
        robust_info = None
    
    # Create test set from remaining pool (excluding training data)
    # Find indices of training data in the full pool
    train_mask = torch.zeros(len(X_pool), dtype=torch.bool)
    for i, x_train in enumerate(data_X):
        # Find matching indices in X_pool
        distances = torch.norm(X_pool - x_train.unsqueeze(0), dim=1)
        train_mask[torch.argmin(distances)] = True
    
    # Test data is the complement
    test_mask = ~train_mask
    X_test_full = X_pool[test_mask]
    y_test = y_pool[test_mask]
    
    if len(X_test_full) > 0:
        print(f"Evaluating on {len(X_test_full)} held-out test samples...")
        
        # COMPARISON: Evaluate both reduced features and full Magpie features
        print(f"\n1. REDUCED FEATURES ({params.num_sparsity_feats} features with {params.sparsity_method}):")
        
        if use_robust_modeling and robust_info is not None:
            results_reduced = evaluate_robust_model(
                final_gp, X_test_full, y_test, 
                robust_info, feat_idx=final_feat_idx, verbose=True
            )
        else:
            results_reduced = evaluate_model_predictions(
                final_gp, X_test_full, y_test, 
                feat_idx=final_feat_idx, verbose=True
            )
        
        print(f"\n2. FULL MAGPIE FEATURES ({X_test_full.shape[1]} features):")
        # Debug: Check data properties
        print(f"Training data shape: {data_X.shape}")
        print(f"Training data range: [{data_X.min():.3f}, {data_X.max():.3f}]")
        print(f"Test data shape: {X_test_full.shape}")
        print(f"Test data range: [{X_test_full.min():.3f}, {X_test_full.max():.3f}]")
        
        try:
            # Fit GP on full features with robust handling
            if use_robust_modeling:
                full_gp, full_robust_info = fit_robust_surrogate(data_X, data_y)
                results_full = evaluate_robust_model(
                    full_gp, X_test_full, y_test, 
                    full_robust_info, feat_idx=None, verbose=True
                )
            else:
                full_gp, full_scaler = fit_surrogate_with_scaling(data_X, data_y)
                results_full = evaluate_model_predictions(
                    full_gp, X_test_full, y_test, 
                    feat_idx=None, scaler=full_scaler, verbose=True
                )
        except Exception as e:
            print(f"❌ Error in full Magpie evaluation: {e}")
            print("Falling back to reduced features only...")
            results_full = None
        
        # Create visualizations for both approaches
        plot_save_path_reduced = os.path.join(os.path.dirname(__file__), "src", "model_evaluation_reduced.png")
        plot_prediction_analysis(results_reduced, save_path=plot_save_path_reduced)
        
        if results_full is not None:
            plot_save_path_full = os.path.join(os.path.dirname(__file__), "src", "model_evaluation_full.png")
            try:
                plot_prediction_analysis(results_full, save_path=plot_save_path_full)
                print(f"✅ Full Magpie plot saved to: {plot_save_path_full}")
            except Exception as e:
                print(f"❌ Error creating full Magpie plot: {e}")
        else:
            print("⚠️  Skipping full Magpie plot due to evaluation error")
        
        # Additional analysis
        print(f"\nAdditional Analysis:")
        print(f"Number of test samples: {len(y_test)}")
        print(f"Test data range: [{y_test.min():.3f}, {y_test.max():.3f}] eV/atom")
        print(f"Training data range: [{data_y.min():.3f}, {data_y.max():.3f}] eV/atom")
        print(f"Selected features: {params.num_sparsity_feats} out of {X_pool.shape[1]} total features")
        
        # Compare the two approaches
        if results_full is not None:
            print(f"\n📊 PERFORMANCE COMPARISON:")
            print(f"{'Approach':<20} {'MAE':<8} {'RMSE':<8} {'R²':<8}")
            print("-" * 45)
            print(f"{'Reduced (30 feats)':<20} {results_reduced['mae']:<8.3f} {results_reduced['rmse']:<8.3f} {results_reduced['r2']:<8.3f}")
            print(f"{'Full Magpie':<20} {results_full['mae']:<8.3f} {results_full['rmse']:<8.3f} {results_full['r2']:<8.3f}")
            
            # Analyze the unexpected result
            mae_diff = results_full['mae'] - results_reduced['mae'] 
            if mae_diff > 0.3:
                print(f"\n🤔 UNEXPECTED RESULT: Full features perform WORSE!")
                print(f"Full Magpie MAE is {mae_diff:.3f} higher than reduced features")
                print("Possible causes:")
                print("1. ⚠️  OVERFITTING: Too many features ({}) vs limited training data ({})".format(
                    X_test_full.shape[1], len(data_X)))
                print("2. ⚠️  CURSE OF DIMENSIONALITY: GP struggles in high-dim space")
                print("3. ⚠️  NOISE FEATURES: Full Magpie includes irrelevant features")
                print("4. ⚠️  SCALING ISSUES: Different feature scales affecting GP")
                print("5. ⚠️  GP HYPERPARAMETERS: Not optimized for high-dimensional data")
                
                # Calculate feature-to-sample ratio
                feature_ratio = X_test_full.shape[1] / len(data_X)
                print(f"\n📊 Dimensionality Analysis:")
                print(f"Feature-to-sample ratio: {feature_ratio:.1f}:1")
                if feature_ratio > 2:
                    print("❗ High dimensionality warning: Features >> Training samples")
            else:
                print(f"\n✅ Feature selection helped! Reduced MAE by {-mae_diff:.3f}")
        else:
            print(f"\n📊 PERFORMANCE SUMMARY (Reduced Features Only):")
            print(f"{'Approach':<20} {'MAE':<8} {'RMSE':<8} {'R²':<8}")
            print("-" * 45)
            print(f"{'Reduced (30 feats)':<20} {results_reduced['mae']:<8.3f} {results_reduced['rmse']:<8.3f} {results_reduced['r2']:<8.3f}")
        
        # Check if MAE > 1.0 and provide diagnostic info
        if results_reduced['mae'] > 1.0:
            print(f"\n⚠️  HIGH MAE DETECTED ({results_reduced['mae']:.3f} > 1.0)")
            print("Potential issues to investigate:")
            print("1. Feature selection removing important information")
            print("2. Limited training data vs feature dimensionality") 
            print("3. Feature scaling/normalization problems")
            print("4. GP hyperparameters not well optimized")
            print("5. Data distribution mismatch between train/test")
        
    else:
        print("⚠️  No test data available for evaluation (all data used in BO)")
        results_reduced = {'mae': None, 'rmse': None, 'r2': None}
        results_full = None
        
    # Add model evaluation results to experiment record
    experiment_results['model_evaluation'] = {
        'reduced_features_mae': results_reduced['mae'] if results_reduced['mae'] is not None else 'N/A',
        'reduced_features_rmse': results_reduced['rmse'] if results_reduced['rmse'] is not None else 'N/A', 
        'reduced_features_r2': results_reduced['r2'] if results_reduced['r2'] is not None else 'N/A',
        'full_features_mae': results_full['mae'] if results_full is not None else 'N/A',
        'full_features_rmse': results_full['rmse'] if results_full is not None else 'N/A',
        'full_features_r2': results_full['r2'] if results_full is not None else 'N/A',
        'test_samples': len(y_test) if len(X_test_full) > 0 else 0
    }
    
    # Save results to JSON file for programmatic access
    import json
    results_file = os.path.join(os.path.dirname(__file__), "formation_energy_results.json")
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\n💾 Experiment results saved to: {results_file}")
    
    # Also save a human-readable summary
    summary_file = os.path.join(os.path.dirname(__file__), "experiment_summary.txt") 
    with open(summary_file, 'w') as f:
        f.write("FORMATION ENERGY BO EXPERIMENT SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Date: {experiment_results['date']}\n")
        f.write(f"Dataset size: {experiment_results['dataset_size']} materials\n")
        f.write(f"Formation energy range: [{experiment_results['data_range'][0]:.4f}, {experiment_results['data_range'][1]:.4f}] eV/atom\n\n")
        
        f.write("BO CONFIGURATION:\n")
        f.write(f"- Budget: {experiment_results['bo_config']['total_sample_budget']} total samples\n")
        f.write(f"- Initialization: {experiment_results['bo_config']['initialization_budget']} random samples\n")
        f.write(f"- Acquisition: {experiment_results['bo_config']['acquisition_function']} (minimize=True)\n")
        f.write(f"- Feature selection: {experiment_results['bo_config']['sparsity_method']} with {experiment_results['bo_config']['num_sparsity_feats']} features\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"- BO found most stable: {experiment_results['bo_found_best']:.4f} eV/atom\n")
        f.write(f"- True global minimum: {experiment_results['true_global_min']:.4f} eV/atom\n") 
        f.write(f"- Gap to optimum: {experiment_results['gap_to_optimum']:.4f} eV/atom ({experiment_results['gap_percentage']:.1f}%)\n")
        f.write(f"- Global ranking: #{experiment_results['global_ranking']} out of {experiment_results['total_materials']}\n\n")
        
        f.write("MODEL EVALUATION:\n")
        if experiment_results['model_evaluation']['reduced_features_mae'] != 'N/A':
            f.write(f"- Reduced features MAE: {experiment_results['model_evaluation']['reduced_features_mae']:.4f}\n")
            f.write(f"- Reduced features R²: {experiment_results['model_evaluation']['reduced_features_r2']:.4f}\n")
        if experiment_results['model_evaluation']['full_features_mae'] != 'N/A':
            f.write(f"- Full Magpie MAE: {experiment_results['model_evaluation']['full_features_mae']:.4f}\n") 
            f.write(f"- Full Magpie R²: {experiment_results['model_evaluation']['full_features_r2']:.4f}\n")
    
    print(f"📄 Human-readable summary saved to: {summary_file}")
    print(f"\n🎯 EXPERIMENT COMPLETE - Results documented for bandgap optimization reference!")


if __name__ == "__main__":
    main()