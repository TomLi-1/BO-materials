import torch
import numpy as np
from dataclasses import dataclass
from typing import Literal, Union, List, Dict, Any
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# BoTorch imports
from botorch.models import SingleTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.utils.multi_objective import is_non_dominated

@dataclass
class MultiObjectiveParameters:
    """Configuration for multi-objective BO experiments"""
    # Dataset configuration
    dataset_name: str = "matbench_expt_gap"
    experiment_type: Literal["bandgap_target", "multi_objective"] = "bandgap_target"
    
    # Bandgap target configuration
    bandgap_target: float = 1.5  # eV - optimal for solar cells
    bandgap_tolerance: float = 0.2  # ±0.2 eV acceptable
    
    # Classification configuration
    include_classification: bool = True
    classification_task: Literal["spacegroup", "crystal_system"] = "spacegroup"
    classification_weight: float = 0.3  # Relative to regression task
    
    # BO parameters
    sparsity_method: Literal["MI", "LASSO", "NONE"] = "MI"
    acq_fun: Literal["EI_target", "UCB", "multi_objective"] = "EI_target"
    num_sparsity_feats: int = 40
    total_sample_budget: int = 200
    initialization_budget: int = 15
    seed: int = 42


def select_features_multi_objective(X: np.ndarray, y_regression: np.ndarray, 
                                  y_classification: np.ndarray = None,
                                  method: str = "MI", k: int = 40, 
                                  classification_weight: float = 0.3):
    """
    Feature selection considering both regression and classification objectives
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y_regression: Regression targets (e.g., bandgap values)
        y_classification: Classification targets (e.g., spacegroup labels)
        method: Feature selection method
        k: Number of features to select
        classification_weight: Weight for classification vs regression
        
    Returns:
        Selected features and feature indices
    """
    if method == "MI":
        # Mutual information for regression
        # Adjust n_neighbors for small datasets
        n_neighbors = min(3, max(1, len(X) - 1))
        mi_reg = mutual_info_regression(X, y_regression, n_neighbors=n_neighbors, random_state=0)
        
        if y_classification is not None:
            # Remove invalid classification labels (-1)
            valid_mask = y_classification >= 0
            if np.sum(valid_mask) > 0:
                X_valid = X[valid_mask]
                y_class_valid = y_classification[valid_mask]
                
                # Mutual information for classification
                mi_class = np.zeros(X.shape[1])
                if len(np.unique(y_class_valid)) > 1:  # Ensure multiple classes
                    n_neighbors_class = min(3, max(1, len(X_valid) - 1))
                    mi_class_valid = mutual_info_classif(X_valid, y_class_valid, n_neighbors=n_neighbors_class, random_state=0)
                    mi_class = mi_class_valid
                
                # Combine MI scores
                mi_combined = (1 - classification_weight) * mi_reg + classification_weight * mi_class
                idx = np.argsort(mi_combined)[-k:]
            else:
                # Fall back to regression only
                idx = np.argsort(mi_reg)[-k:]
        else:
            # Regression only
            idx = np.argsort(mi_reg)[-k:]
            
    elif method == "LASSO":
        # Use LASSO on regression target (primary objective)
        model = Lasso(alpha=1e-3, random_state=0).fit(X, y_regression)
        coef = np.abs(model.coef_)
        idx = np.argsort(coef)[-k:]
        
    else:  # NONE
        idx = np.arange(min(k, X.shape[1]))
        
    return X[:, idx], idx


def fit_surrogate_target_based(train_X: torch.Tensor, train_y: torch.Tensor, 
                               target_value: float = 1.5):
    """
    Fit GP surrogate optimized for target-based acquisition
    
    Args:
        train_X: Training features
        train_y: Training targets (bandgap values)
        target_value: Target bandgap value
        
    Returns:
        Trained GP model
    """
    # Standard GP fitting
    feat_dim = train_X.size(-1)
    output_dim = train_y.unsqueeze(-1).size(-1)
    
    gp = SingleTaskGP(
        train_X,
        train_y.unsqueeze(-1),
        input_transform=Normalize(feat_dim),
        outcome_transform=Standardize(output_dim),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    return gp


class TargetBasedAcquisition:
    """
    Custom acquisition function for targeting specific bandgap values
    """
    def __init__(self, gp, target_value: float, tolerance: float = 0.2):
        self.gp = gp
        self.target_value = target_value
        self.tolerance = tolerance
        
    def __call__(self, X):
        """
        Acquisition function that prefers materials near the target bandgap
        
        Args:
            X: Candidate points to evaluate
            
        Returns:
            Acquisition values (higher = more desirable)
        """
        self.gp.eval()
        
        with torch.no_grad():
            posterior = self.gp.posterior(X)
            mean = posterior.mean.squeeze(-1)
            std = posterior.variance.sqrt().squeeze(-1)
        
        # Distance from target
        distance_from_target = torch.abs(mean - self.target_value)
        
        # Probability of being within tolerance
        prob_in_range = (
            torch.erf((self.target_value + self.tolerance - mean) / (std * np.sqrt(2))) -
            torch.erf((self.target_value - self.tolerance - mean) / (std * np.sqrt(2)))
        ) / 2
        
        # Combine exploration (uncertainty) with target proximity
        acquisition = prob_in_range * std / (1 + distance_from_target)
        
        return acquisition


def make_multi_objective_acquisition(gp, target_value: float = 1.5, 
                                   tolerance: float = 0.2, acq_fun: str = "EI_target"):
    """
    Create acquisition function for multi-objective optimization
    
    Args:
        gp: Trained GP model
        target_value: Target bandgap value
        tolerance: Acceptable range around target
        acq_fun: Type of acquisition function
        
    Returns:
        Acquisition function
    """
    if acq_fun == "EI_target":
        return TargetBasedAcquisition(gp, target_value, tolerance)
    elif acq_fun == "UCB":
        return UpperConfidenceBound(model=gp, beta=5.0)
    else:
        raise ValueError(f"Unknown acquisition function: {acq_fun}")


def evaluate_bandgap_targeting(gp, X_test: torch.Tensor, y_test: torch.Tensor,
                              target_value: float = 1.5, tolerance: float = 0.2,
                              feat_idx: np.ndarray = None, scaler=None, verbose: bool = True):
    """
    Evaluate GP model for bandgap targeting task
    
    Args:
        gp: Trained GP model
        X_test: Test features
        y_test: Test bandgap values  
        target_value: Target bandgap value
        tolerance: Acceptable range
        feat_idx: Selected feature indices
        scaler: Feature scaler
        verbose: Print metrics
        
    Returns:
        Evaluation results dictionary
    """
    gp.eval()
    
    # Feature selection and scaling
    if feat_idx is not None:
        X_test_selected = X_test[:, feat_idx]
    else:
        X_test_selected = X_test
        
    if scaler is not None:
        X_test_np = X_test_selected.numpy()
        X_test_scaled = scaler.transform(X_test_np)
        X_test_selected = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    # Get predictions
    with torch.no_grad():
        posterior = gp.posterior(X_test_selected)
        y_pred = posterior.mean.squeeze(-1)
        y_std = posterior.variance.sqrt().squeeze(-1)
    
    # Convert to numpy
    y_pred_np = y_pred.numpy()
    y_test_np = y_test.numpy()
    y_std_np = y_std.numpy()
    
    # Standard regression metrics
    mae = mean_absolute_error(y_test_np, y_pred_np)
    mse = mean_squared_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_np, y_pred_np)
    
    # Target-specific metrics
    target_min = target_value - tolerance
    target_max = target_value + tolerance
    
    # How many test materials are actually in target range?
    test_in_target = np.sum((y_test_np >= target_min) & (y_test_np <= target_max))
    test_target_rate = test_in_target / len(y_test_np)
    
    # How many predictions are in target range?
    pred_in_target = np.sum((y_pred_np >= target_min) & (y_pred_np <= target_max))
    pred_target_rate = pred_in_target / len(y_pred_np)
    
    # True positives for target detection
    true_positives = np.sum(
        ((y_test_np >= target_min) & (y_test_np <= target_max)) &
        ((y_pred_np >= target_min) & (y_pred_np <= target_max))
    )
    
    target_precision = true_positives / max(pred_in_target, 1)
    target_recall = true_positives / max(test_in_target, 1)
    target_f1 = 2 * target_precision * target_recall / max(target_precision + target_recall, 1e-10)
    
    if verbose:
        print(f"\nBandgap Targeting Evaluation:")
        print(f"Standard Regression Metrics:")
        print(f"  MAE: {mae:.4f} eV")
        print(f"  RMSE: {rmse:.4f} eV")
        print(f"  R²: {r2:.4f}")
        print(f"\nTarget-Specific Metrics (target: {target_value}±{tolerance} eV):")
        print(f"  Test materials in target range: {test_in_target}/{len(y_test_np)} ({test_target_rate:.1%})")
        print(f"  Predictions in target range: {pred_in_target}/{len(y_pred_np)} ({pred_target_rate:.1%})")
        print(f"  Target detection F1: {target_f1:.4f}")
        print(f"  Target precision: {target_precision:.4f}")
        print(f"  Target recall: {target_recall:.4f}")
    
    return {
        'predictions': y_pred_np,
        'ground_truth': y_test_np,
        'uncertainty': y_std_np,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'target_value': target_value,
        'tolerance': tolerance,
        'test_in_target': test_in_target,
        'pred_in_target': pred_in_target,
        'target_f1': target_f1,
        'target_precision': target_precision,
        'target_recall': target_recall
    }


def plot_bandgap_targeting_analysis(results: dict, save_path: str = None):
    """
    Create visualizations for bandgap targeting analysis
    
    Args:
        results: Results dictionary from evaluate_bandgap_targeting
        save_path: Path to save the plot
    """
    y_pred = results['predictions']
    y_true = results['ground_truth']
    y_std = results['uncertainty']
    target_value = results['target_value']
    tolerance = results['tolerance']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Bandgap Targeting Analysis (Target: {target_value}±{tolerance:.1f} eV)', fontsize=16)
    
    # Target range bounds
    target_min = target_value - tolerance
    target_max = target_value + tolerance
    
    # 1. Predictions vs Ground Truth with target zone
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.6, s=20, c='blue', label='Predictions')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    # Highlight target zone
    ax1.axvspan(target_min, target_max, alpha=0.2, color='green', label='Target range')
    ax1.axhspan(target_min, target_max, alpha=0.2, color='green')
    
    ax1.set_xlabel('Ground Truth Bandgap (eV)')
    ax1.set_ylabel('Predicted Bandgap (eV)')
    ax1.set_title('Predictions vs Ground Truth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add metrics text
    ax1.text(0.05, 0.95, f'R² = {results["r2"]:.3f}\nMAE = {results["mae"]:.3f} eV\nF1 = {results["target_f1"]:.3f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Residuals plot with target highlighting
    ax2 = axes[0, 1]
    residuals = y_pred - y_true
    colors = ['green' if (target_min <= yt <= target_max) else 'blue' for yt in y_true]
    ax2.scatter(y_true, residuals, alpha=0.6, s=20, c=colors)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.axvspan(target_min, target_max, alpha=0.2, color='green', label='Target range')
    ax2.set_xlabel('Ground Truth Bandgap (eV)')
    ax2.set_ylabel('Residuals (Predicted - True)')
    ax2.set_title('Residuals (Green = Target Range Materials)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Target zone analysis
    ax3 = axes[1, 0]
    # Create bins for bandgap ranges
    bins = np.linspace(min_val, max_val, 20)
    ax3.hist(y_true, bins=bins, alpha=0.7, label='Ground Truth', edgecolor='black')
    ax3.hist(y_pred, bins=bins, alpha=0.7, label='Predictions', edgecolor='black')
    ax3.axvspan(target_min, target_max, alpha=0.3, color='green', label='Target range')
    ax3.axvline(target_value, color='red', linestyle='--', linewidth=2, label='Target value')
    ax3.set_xlabel('Bandgap (eV)')
    ax3.set_ylabel('Count')
    ax3.set_title('Bandgap Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Uncertainty vs Distance from Target
    ax4 = axes[1, 1]
    distance_from_target = np.abs(y_true - target_value)
    ax4.scatter(distance_from_target, y_std, alpha=0.6, s=20)
    ax4.set_xlabel('Distance from Target (eV)')
    ax4.set_ylabel('Prediction Uncertainty')
    ax4.set_title('Model Uncertainty vs Target Distance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig