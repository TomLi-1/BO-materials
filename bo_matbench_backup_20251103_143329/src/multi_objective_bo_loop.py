import torch
import numpy as np
from multi_objective_surrogate import (
    fit_surrogate_target_based, make_multi_objective_acquisition, 
    select_features_multi_objective, MultiObjectiveParameters
)

try:
    from botorch.optim import optimize_acqf
except ImportError:
    from botorch.optim.optimize import optimize_acqf


def bandgap_targeting_bo(
    X_init: torch.Tensor,
    y_init: torch.Tensor,
    X_pool: torch.Tensor, 
    y_pool: torch.Tensor,
    params: MultiObjectiveParameters,
    y_classification_init: torch.Tensor = None,
    y_classification_pool: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bayesian Optimization for bandgap targeting with optional classification
    
    Args:
        X_init: Initial feature vectors
        y_init: Initial bandgap values
        X_pool: Full pool of feature vectors
        y_pool: Full pool of bandgap values  
        params: BO configuration parameters
        y_classification_init: Initial classification labels (optional)
        y_classification_pool: Full pool of classification labels (optional)
        
    Returns:
        data_X: Selected feature vectors
        data_y: Corresponding bandgap values
    """
    # Initialize data
    y0 = torch.as_tensor(y_init, dtype=torch.float32)
    data_X = X_init.clone()
    data_y = y0.clone()
    
    # Classification data if provided
    if y_classification_init is not None:
        data_y_class = y_classification_init.clone()
    else:
        data_y_class = None
    
    print(f"Starting bandgap targeting BO:")
    print(f"Target: {params.bandgap_target} ± {params.bandgap_tolerance} eV")
    print(f"Budget: {params.total_sample_budget} total samples")
    
    # Main BO loop
    for iteration in range(params.total_sample_budget - params.initialization_budget):
        print(f"\nBO Iteration {iteration + 1}/{params.total_sample_budget - params.initialization_budget}")
        
        # Feature selection (considering both objectives if classification included)
        if data_y_class is not None and params.include_classification:
            Xfs_np, feat_idx = select_features_multi_objective(
                data_X.numpy(),
                data_y.numpy(),
                data_y_class.numpy(),
                method=params.sparsity_method,
                k=params.num_sparsity_feats,
                classification_weight=params.classification_weight
            )
        else:
            # Regression only
            Xfs_np, feat_idx = select_features_multi_objective(
                data_X.numpy(),
                data_y.numpy(),
                method=params.sparsity_method,
                k=params.num_sparsity_feats
            )
        
        Xfs = torch.tensor(Xfs_np, dtype=torch.float32)
        
        # Fit GP surrogate model
        gp = fit_surrogate_target_based(
            Xfs, data_y, target_value=params.bandgap_target
        )
        
        # Create acquisition function
        acq = make_multi_objective_acquisition(
            gp, 
            target_value=params.bandgap_target,
            tolerance=params.bandgap_tolerance,
            acq_fun=params.acq_fun
        )
        
        # Optimize acquisition function
        bounds = torch.stack([Xfs.min(dim=0).values, Xfs.max(dim=0).values])
        
        # Ensure bounds are valid
        for i in range(bounds.shape[1]):
            if bounds[0, i] == bounds[1, i]:  # Identical bounds
                bounds[1, i] = bounds[0, i] + 1e-6
        
        try:
            cand_feat, acq_value = optimize_acqf(
                acq,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=100,
            )
            
            if cand_feat.dtype != X_pool.dtype:
                cand_feat = cand_feat.to(X_pool.dtype)
            
            # Find nearest point in pool
            dists = torch.cdist(cand_feat, X_pool[:, feat_idx])
            idx_near = torch.argmin(dists).item()
            
            # Add selected point to dataset
            new_X = X_pool[idx_near].unsqueeze(0)
            new_y = y_pool[idx_near].unsqueeze(0)
            
            data_X = torch.cat([data_X, new_X], dim=0)
            data_y = torch.cat([data_y, new_y], dim=0)
            
            # Add classification label if available
            if y_classification_pool is not None and data_y_class is not None:
                new_y_class = y_classification_pool[idx_near].unsqueeze(0)
                data_y_class = torch.cat([data_y_class, new_y_class], dim=0)
            
            # Progress reporting
            current_best = data_y.min().item() if params.acq_fun == "minimize" else data_y.max().item()
            target_distance = abs(data_y[-1].item() - params.bandgap_target)
            
            print(f"  Selected bandgap: {data_y[-1].item():.4f} eV")
            print(f"  Distance from target: {target_distance:.4f} eV")
            print(f"  Acquisition value: {acq_value.item():.6f}")
            
            # Check if we found materials in target range
            in_target_range = (
                (data_y >= params.bandgap_target - params.bandgap_tolerance) &
                (data_y <= params.bandgap_target + params.bandgap_tolerance)
            )
            num_in_target = in_target_range.sum().item()
            print(f"  Materials in target range: {num_in_target}/{len(data_y)}")
            
        except Exception as e:
            print(f"  ⚠️ Acquisition optimization failed: {e}")
            print(f"  Using random selection as fallback...")
            
            # Random fallback
            remaining_indices = list(range(len(X_pool)))
            # Remove already selected indices (approximate)
            selected_idx = torch.randint(0, len(remaining_indices), (1,)).item()
            idx_near = remaining_indices[selected_idx]
            
            new_X = X_pool[idx_near].unsqueeze(0) 
            new_y = y_pool[idx_near].unsqueeze(0)
            
            data_X = torch.cat([data_X, new_X], dim=0)
            data_y = torch.cat([data_y, new_y], dim=0)
    
    # Final summary
    final_in_target = (
        (data_y >= params.bandgap_target - params.bandgap_tolerance) &
        (data_y <= params.bandgap_target + params.bandgap_tolerance)
    ).sum().item()
    
    best_distance = torch.min(torch.abs(data_y - params.bandgap_target)).item()
    closest_bandgap = data_y[torch.argmin(torch.abs(data_y - params.bandgap_target))].item()
    
    print(f"\n🎯 BANDGAP TARGETING COMPLETE:")
    print(f"Target: {params.bandgap_target} ± {params.bandgap_tolerance} eV")
    print(f"Materials in target range: {final_in_target}/{len(data_y)}")
    print(f"Closest found: {closest_bandgap:.4f} eV (distance: {best_distance:.4f})")
    
    return data_X, data_y


def evaluate_target_achievement(data_y: torch.Tensor, target: float, tolerance: float):
    """
    Evaluate how well BO achieved the bandgap target
    
    Args:
        data_y: Collected bandgap values
        target: Target bandgap value
        tolerance: Acceptable tolerance
        
    Returns:
        Dictionary with achievement metrics
    """
    # Materials within target range
    in_range = ((data_y >= target - tolerance) & (data_y <= target + tolerance)).sum().item()
    total = len(data_y)
    success_rate = in_range / total
    
    # Distance metrics
    distances = torch.abs(data_y - target)
    min_distance = distances.min().item()
    mean_distance = distances.mean().item()
    
    # Best material found
    best_idx = torch.argmin(distances)
    best_bandgap = data_y[best_idx].item()
    
    # Progressive improvement (how distance changed over time)
    cumulative_best = []
    for i in range(len(data_y)):
        best_so_far = torch.min(distances[:i+1]).item()
        cumulative_best.append(best_so_far)
    
    return {
        'success_rate': success_rate,
        'materials_in_range': in_range,
        'total_materials': total,
        'min_distance': min_distance,
        'mean_distance': mean_distance,
        'best_bandgap': best_bandgap,
        'target_value': target,
        'tolerance': tolerance,
        'convergence_curve': cumulative_best
    }


if __name__ == "__main__":
    print("Multi-objective BO loop module loaded successfully!")
    print("Available functions:")
    print("- bandgap_targeting_bo(): Main BO loop for bandgap targeting")
    print("- evaluate_target_achievement(): Analyze target achievement metrics")