from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso

@dataclass
class OptimizerParameters:
    # Feature‐selection method: Mutual Information, Lasso, etc.
    sparsity_method: Literal["MI", "LASSO", "NONE"] = "MI"
    # Acquisition function: Expected Improvement, UCB, …
    acq_fun:       Literal["EI", "UCB", "Thompson"] = "EI"
    num_sparsity_feats: int   = 30
    multi_objective:    bool  = False
    constrained:        bool  = False
    total_sample_budget:int  = 200
    initialization_budget:int= 10
    seed:               int  = 42


def select_features(X: np.ndarray,
                    y: np.ndarray,
                    method: str,
                    k: int):
    """
    X: (n_samples, n_features)
    y: (n_samples,)
    method: "MI", "LASSO", or "NONE"
    k: number of features to keep
    returns: X_sel with only k columns
    """
    if method == "MI":
        mi = mutual_info_regression(X, y, random_state=0)
        idx = np.argsort(mi)[-k:]
    elif method == "LASSO":
        model = Lasso(alpha=1e-3, random_state=0).fit(X, y)
        coef = np.abs(model.coef_)
        idx  = np.argsort(coef)[-k:]
    else:  # NONE
        idx = np.arange(X.shape[1])
    return X[:, idx], idx

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

def fit_surrogate(train_X: torch.Tensor,
                  train_y: torch.Tensor):
    # 1) instantiate
    # compute dims
    feat_dim   = train_X.size(-1)                    # e.g. 132 features
    output_dim = train_y.unsqueeze(-1).size(-1)      # always 1 here

    # pass ints, not Tensors, into the transforms
    gp = SingleTaskGP(
        train_X,
        train_y.unsqueeze(-1),
        input_transform=Normalize(feat_dim),
        outcome_transform=Standardize(output_dim),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    # 2) fit in place
    fit_gpytorch_mll(mll)
    # 3) return the trained GP model (not the MLL)
    return gp


def fit_surrogate_with_scaling(train_X: torch.Tensor,
                              train_y: torch.Tensor):
    """
    Fit surrogate with explicit min-max scaling to avoid BoTorch warnings
    Returns both the GP model and the scaler for consistent test data preprocessing
    """
    # Apply min-max scaling
    X_np = train_X.numpy()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_np)
    train_X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Compute dims
    feat_dim   = train_X_scaled.size(-1)
    output_dim = train_y.unsqueeze(-1).size(-1)

    # Create GP without additional normalization (since we already scaled to [0,1])
    gp = SingleTaskGP(
        train_X_scaled,
        train_y.unsqueeze(-1),
        outcome_transform=Standardize(output_dim),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    return gp, scaler

def make_acquisition(gp, best_f: torch.Tensor, acq_fun: str, minimize: bool = False):
    if acq_fun == "EI":
        if minimize:
            # For minimization, we want Expected Improvement below the current best
            # BoTorch EI automatically handles this when best_f is the minimum
            return ExpectedImprovement(model=gp, best_f=best_f, maximize=False)
        else:
            return ExpectedImprovement(model=gp, best_f=best_f)
    elif acq_fun == "UCB":
        if minimize:
            # For minimization, we want Lower Confidence Bound (negative UCB)
            return UpperConfidenceBound(model=gp, beta=5.0, maximize=False)
        else:
            return UpperConfidenceBound(model=gp, beta=5.0)
    else:
        raise ValueError(f"Unknown acq_fun {acq_fun}")


def evaluate_model_predictions(gp, X_test: torch.Tensor, y_test: torch.Tensor, 
                              feat_idx: np.ndarray = None, scaler=None, verbose: bool = True):
    """
    Evaluate GP model predictions against ground truth values
    
    Args:
        gp: Trained GP model
        X_test: Test features (full or selected)
        y_test: Ground truth test values
        feat_idx: Feature indices used for selection (if applicable)
        verbose: Print metrics
    
    Returns:
        dict with predictions, ground_truth, and metrics
    """
    gp.eval()
    
    # Select features if feat_idx provided
    if feat_idx is not None:
        X_test_selected = X_test[:, feat_idx]
    else:
        X_test_selected = X_test
    
    # Apply scaling if scaler provided
    if scaler is not None:
        X_test_np = X_test_selected.numpy()
        X_test_scaled = scaler.transform(X_test_np)
        X_test_selected = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    # Get predictions
    with torch.no_grad():
        posterior = gp.posterior(X_test_selected)
        y_pred = posterior.mean.squeeze(-1)
        y_std = posterior.variance.sqrt().squeeze(-1)
    
    # Convert to numpy for metrics
    y_pred_np = y_pred.numpy()
    y_test_np = y_test.numpy()
    y_std_np = y_std.numpy()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_np, y_pred_np)
    mse = mean_squared_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_np, y_pred_np)
    
    if verbose:
        print(f"\nModel Evaluation Metrics:")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")
        print(f"Mean prediction uncertainty: {y_std_np.mean():.4f}")
    
    return {
        'predictions': y_pred_np,
        'ground_truth': y_test_np,
        'uncertainty': y_std_np,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def plot_prediction_analysis(results: dict, save_path: str = None):
    """
    Create comprehensive plots comparing predictions vs ground truth
    
    Args:
        results: Dict from evaluate_model_predictions
        save_path: Optional path to save the plot
    """
    y_pred = results['predictions']
    y_true = results['ground_truth']
    y_std = results['uncertainty']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Prediction Analysis', fontsize=16)
    
    # 1. Predictions vs Ground Truth (scatter)
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.6, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('Ground Truth (eV/atom)')
    ax1.set_ylabel('Predicted (eV/atom)')
    ax1.set_title('Predictions vs Ground Truth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add R² and MAE to the plot
    r2 = results['r2']
    mae = results['mae']
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Residuals plot
    ax2 = axes[0, 1]
    residuals = y_pred - y_true
    ax2.scatter(y_true, residuals, alpha=0.6, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Ground Truth (eV/atom)')
    ax2.set_ylabel('Residuals (Predicted - True)')
    ax2.set_title('Residuals vs Ground Truth')
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals histogram
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Residuals (eV/atom)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Residuals')
    ax3.grid(True, alpha=0.3)
    
    # 4. Uncertainty vs Absolute Error
    ax4 = axes[1, 1]
    abs_error = np.abs(residuals)
    ax4.scatter(y_std, abs_error, alpha=0.6, s=20)
    ax4.set_xlabel('Prediction Uncertainty (σ)')
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Model Uncertainty vs Absolute Error')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig


def fit_robust_surrogate(train_X: torch.Tensor, train_y: torch.Tensor, 
                        remove_outliers: bool = True, outlier_threshold: float = 3.0):
    """
    Fit a robust GP model that handles outliers and extreme values better.
    
    Args:
        train_X: Training features
        train_y: Training targets (formation energies)
        remove_outliers: Whether to remove outliers during training
        outlier_threshold: Z-score threshold for outlier detection
        
    Returns:
        Trained GP model and preprocessing information
    """
    print(f"🔧 Fitting robust surrogate model...")
    
    # Convert to numpy for preprocessing
    X_np = train_X.numpy()
    y_np = train_y.numpy()
    
    print(f"   Original data: {len(y_np)} samples")
    print(f"   Energy range: [{y_np.min():.3f}, {y_np.max():.3f}] eV/atom")
    
    # Outlier detection using robust statistics
    outlier_mask = np.ones(len(y_np), dtype=bool)
    if remove_outliers:
        # Use median absolute deviation for outlier detection
        median_y = np.median(y_np)
        mad_y = np.median(np.abs(y_np - median_y))
        
        if mad_y > 0:  # Avoid division by zero
            # Robust z-score
            robust_z_scores = 0.6745 * (y_np - median_y) / mad_y
            outlier_mask = np.abs(robust_z_scores) <= outlier_threshold
            
            outliers_removed = (~outlier_mask).sum()
            print(f"   Outliers removed: {outliers_removed} ({outliers_removed/len(y_np):.1%})")
            
            # Apply outlier mask
            X_clean = X_np[outlier_mask]
            y_clean = y_np[outlier_mask]
        else:
            X_clean, y_clean = X_np, y_np
    else:
        X_clean, y_clean = X_np, y_np
    
    print(f"   Clean data: {len(y_clean)} samples")
    
    # Robust feature scaling using RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Convert back to tensors
    train_X_robust = torch.tensor(X_scaled, dtype=torch.float32)
    train_y_robust = torch.tensor(y_clean, dtype=torch.float32)
    
    # Compute dims
    feat_dim = train_X_robust.size(-1)
    output_dim = train_y_robust.unsqueeze(-1).size(-1)
    
    # Fit GP with robust preprocessing
    gp = SingleTaskGP(
        train_X_robust,
        train_y_robust.unsqueeze(-1),
        outcome_transform=Standardize(output_dim),
    )
    
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    
    try:
        fit_gpytorch_mll(mll)
        print(f"✅ Robust GP training complete")
    except Exception as e:
        print(f"⚠️  GP training warning: {e}")
    
    # Store preprocessing info
    preprocessing_info = {
        'scaler': scaler,
        'outlier_mask': outlier_mask,
        'outliers_removed': (~outlier_mask).sum() if remove_outliers else 0,
        'original_size': len(y_np),
        'clean_size': len(y_clean)
    }
    
    return gp, preprocessing_info


def evaluate_robust_model(gp, X_test: torch.Tensor, y_test: torch.Tensor,
                         preprocessing_info: dict, feat_idx: np.ndarray = None, 
                         verbose: bool = True):
    """
    Evaluate robust model with proper preprocessing applied to test data.
    """
    gp.eval()
    
    # Select features if feat_idx provided
    if feat_idx is not None:
        X_test_selected = X_test[:, feat_idx]
    else:
        X_test_selected = X_test
    
    # Apply same robust scaling as training
    scaler = preprocessing_info['scaler']
    X_test_np = X_test_selected.numpy()
    X_test_scaled = scaler.transform(X_test_np)
    X_test_robust = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    # Get predictions
    with torch.no_grad():
        posterior = gp.posterior(X_test_robust)
        y_pred = posterior.mean.squeeze(-1)
        y_std = posterior.variance.sqrt().squeeze(-1)
    
    # Convert to numpy for metrics
    y_pred_np = y_pred.numpy()
    y_test_np = y_test.numpy()
    y_std_np = y_std.numpy()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_np, y_pred_np)
    mse = mean_squared_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_np, y_pred_np)
    
    # Analyze performance by energy range
    if verbose:
        print(f"\n📊 Robust Model Evaluation:")
        print(f"MAE:  {mae:.4f} eV/atom")
        print(f"RMSE: {rmse:.4f} eV/atom")
        print(f"R²:   {r2:.4f}")
        print(f"Mean uncertainty: {y_std_np.mean():.4f}")
        
        analyze_performance_by_energy_range(y_test_np, y_pred_np, y_std_np)
    
    return {
        'predictions': y_pred_np,
        'ground_truth': y_test_np,
        'uncertainty': y_std_np,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'preprocessing_info': preprocessing_info
    }


def analyze_performance_by_energy_range(y_true, y_pred, y_std):
    """Analyze model performance across different formation energy ranges."""
    print(f"\n📈 Performance by formation energy range:")
    
    # Define energy ranges based on thermodynamic stability
    ranges = [
        ("Very Stable", -np.inf, -2.0),
        ("Stable", -2.0, -0.5), 
        ("Marginally Stable", -0.5, 0.0),
        ("Unstable", 0.0, 1.0),
        ("Very Unstable", 1.0, np.inf)
    ]
    
    for name, min_e, max_e in ranges:
        mask = (y_true >= min_e) & (y_true < max_e)
        if mask.sum() == 0:
            continue
            
        y_range = y_true[mask]
        pred_range = y_pred[mask]
        std_range = y_std[mask]
        
        mae_range = np.mean(np.abs(y_range - pred_range))
        rmse_range = np.sqrt(np.mean((y_range - pred_range)**2))
        avg_uncertainty = np.mean(std_range)
        
        print(f"   {name:16} ({mask.sum():5d}): MAE={mae_range:.3f}, RMSE={rmse_range:.3f}, σ={avg_uncertainty:.3f}")
        
        # Flag problematic ranges
        if mae_range > 1.0:
            print(f"      ⚠️  High error in {name} range!")
