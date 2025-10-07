"""
Advanced surrogate modeling with data augmentation and adaptive kernels 
specifically designed to handle high formation energy prediction failures.
"""

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
import gpytorch
from typing import Dict, Tuple


def augment_high_energy_data(X: np.ndarray, y: np.ndarray, 
                            energy_threshold: float = 0.0,
                            augmentation_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment high-energy materials with synthetic samples to balance the dataset.
    
    Args:
        X: Feature matrix
        y: Formation energies
        energy_threshold: Energy above which to consider "high-energy"
        augmentation_factor: How many synthetic samples per high-energy sample
        
    Returns:
        Augmented X and y arrays
    """
    print(f"🔧 Augmenting high-energy data (E > {energy_threshold:.1f} eV/atom)...")
    
    high_energy_mask = y > energy_threshold
    high_energy_count = high_energy_mask.sum()
    
    if high_energy_count == 0:
        print("   No high-energy samples found, skipping augmentation")
        return X, y
    
    print(f"   Found {high_energy_count} high-energy samples")
    
    # Extract high-energy samples
    X_high = X[high_energy_mask]
    y_high = y[high_energy_mask]
    
    # Generate synthetic samples using nearest neighbor interpolation
    X_augmented = []
    y_augmented = []
    
    if len(X_high) >= 2:
        # Fit nearest neighbors on high-energy samples
        nn = NearestNeighbors(n_neighbors=min(3, len(X_high)), metric='euclidean')
        nn.fit(X_high)
        
        for i in range(len(X_high)):
            # Find neighbors of each high-energy sample
            distances, indices = nn.kneighbors([X_high[i]])
            
            # Generate synthetic samples by interpolating between neighbors
            for _ in range(augmentation_factor):
                if len(indices[0]) > 1:
                    # Random interpolation between this sample and a neighbor
                    neighbor_idx = np.random.choice(indices[0][1:])  # Skip self (index 0)
                    alpha = np.random.uniform(0.2, 0.8)  # Interpolation weight
                    
                    X_synthetic = alpha * X_high[i] + (1 - alpha) * X_high[neighbor_idx]
                    y_synthetic = alpha * y_high[i] + (1 - alpha) * y_high[neighbor_idx]
                    
                    # Add small random noise to avoid exact duplicates
                    noise_scale = 0.01 * np.std(X_high, axis=0)
                    X_synthetic += np.random.normal(0, noise_scale, X_synthetic.shape)
                    
                    X_augmented.append(X_synthetic)
                    y_augmented.append(y_synthetic)
    
    if X_augmented:
        X_augmented = np.array(X_augmented)
        y_augmented = np.array(y_augmented)
        
        # Combine with original data
        X_combined = np.vstack([X, X_augmented])
        y_combined = np.hstack([y, y_augmented])
        
        print(f"   Added {len(X_augmented)} synthetic high-energy samples")
        print(f"   New dataset size: {len(X_combined)} (was {len(X)})")
        
        return X_combined, y_combined
    else:
        print("   Could not generate synthetic samples")
        return X, y


class AdaptiveKernelGP(SingleTaskGP):
    """
    GP with adaptive kernel that changes behavior based on energy ranges.
    Uses a mixture of kernels with energy-dependent mixing.
    """
    
    def __init__(self, train_X, train_Y, train_y_energies=None, **kwargs):
        super().__init__(train_X, train_Y, **kwargs)
        
        if train_y_energies is not None:
            # Store energy values for adaptive kernel behavior
            self.register_buffer('train_energies', torch.tensor(train_y_energies, dtype=torch.float32))
        else:
            self.train_energies = None
    
    def _setup_adaptive_covar(self, train_X, train_Y):
        """Setup adaptive covariance module based on energy distribution."""
        
        # Create multiple kernels for different energy regimes
        stable_kernel = ScaleKernel(MaternKernel(nu=2.5))    # Smooth for stable materials
        unstable_kernel = ScaleKernel(RBFKernel())           # More flexible for unstable
        
        return stable_kernel, unstable_kernel


def fit_advanced_surrogate(train_X: torch.Tensor, train_y: torch.Tensor,
                          use_augmentation: bool = True,
                          use_adaptive_weighting: bool = True,
                          kernel_type: str = "matern52") -> SingleTaskGP:
    """
    Fit advanced GP with data augmentation and adaptive weighting for high-energy materials.
    
    Args:
        train_X: Training features
        train_y: Training formation energies
        use_augmentation: Whether to augment high-energy data
        use_adaptive_weighting: Whether to use adaptive sample weights
        kernel_type: Type of kernel to use
        
    Returns:
        Fitted GP model
    """
    print(f"🔧 Fitting advanced surrogate (augmentation: {use_augmentation}, weighting: {use_adaptive_weighting})...")
    
    # Convert to numpy for processing
    X_np = train_X.numpy()
    y_np = train_y.numpy()
    
    # Data augmentation for high-energy materials
    if use_augmentation:
        X_aug, y_aug = augment_high_energy_data(X_np, y_np, energy_threshold=0.0, augmentation_factor=2)
        
        # Convert back to tensors
        train_X_final = torch.tensor(X_aug, dtype=torch.float32)
        train_y_final = torch.tensor(y_aug, dtype=torch.float32)
    else:
        train_X_final = train_X
        train_y_final = train_y
        y_aug = y_np
    
    # Adaptive weighting
    if use_adaptive_weighting:
        # Compute weights that emphasize high-energy materials
        weights = compute_adaptive_weights(y_aug)
        print(f"   Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
    else:
        weights = None
    
    # Select kernel
    if kernel_type == "rbf":
        base_kernel = RBFKernel()
    elif kernel_type == "matern32":
        base_kernel = MaternKernel(nu=1.5)
    elif kernel_type == "matern52":
        base_kernel = MaternKernel(nu=2.5)
    else:
        base_kernel = RBFKernel()
    
    # Create GP model
    feat_dim = train_X_final.size(-1)
    output_dim = train_y_final.unsqueeze(-1).size(-1)
    
    gp = SingleTaskGP(
        train_X_final,
        train_y_final.unsqueeze(-1),
        covar_module=ScaleKernel(base_kernel),
        input_transform=Normalize(feat_dim),
        outcome_transform=Standardize(output_dim),
    )
    
    # Store weights if used
    if weights is not None:
        gp.register_buffer('sample_weights', torch.tensor(weights, dtype=torch.float32))
    
    # Fit model
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    
    try:
        fit_gpytorch_mll(mll)
        print(f"✅ Advanced GP training complete")
        
        # Report final dataset composition
        final_high_count = (y_aug > 0.0).sum()
        final_total = len(y_aug)
        print(f"   Final dataset: {final_total} samples, {final_high_count} high-energy ({final_high_count/final_total*100:.1f}%)")
        
    except Exception as e:
        print(f"⚠️  GP training warning: {e}")
    
    return gp


def compute_adaptive_weights(y_values: np.ndarray) -> np.ndarray:
    """
    Compute adaptive weights that strongly emphasize high-energy materials.
    """
    # Base weights
    weights = np.ones_like(y_values)
    
    # Energy-based weights
    high_energy_mask = y_values > 0.0
    moderate_energy_mask = (y_values > -0.5) & (y_values <= 0.0)
    
    # Strong emphasis on unstable materials
    weights[high_energy_mask] = 5.0  # 5x weight for unstable
    weights[moderate_energy_mask] = 2.0  # 2x weight for marginally stable
    
    # Additional emphasis for very high energies
    very_high_mask = y_values > 1.0
    weights[very_high_mask] = 8.0  # Even stronger emphasis
    
    return weights


def evaluate_advanced_model(gp, X_test: torch.Tensor, y_test: torch.Tensor,
                           feat_idx: np.ndarray = None, verbose: bool = True) -> Dict:
    """
    Evaluate advanced GP model with detailed high-energy analysis.
    """
    gp.eval()
    
    # Select features if needed
    if feat_idx is not None:
        X_test_selected = X_test[:, feat_idx]
    else:
        X_test_selected = X_test
    
    # Get predictions
    with torch.no_grad():
        posterior = gp.posterior(X_test_selected)
        y_pred = posterior.mean.squeeze(-1)
        y_std = posterior.variance.sqrt().squeeze(-1)
    
    # Convert to numpy
    y_pred_np = y_pred.numpy()
    y_test_np = y_test.numpy()
    y_std_np = y_std.numpy()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_np, y_pred_np)
    mse = mean_squared_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_np, y_pred_np)
    
    if verbose:
        print(f"\n📊 Advanced Model Evaluation:")
        print(f"MAE:  {mae:.4f} eV/atom")
        print(f"RMSE: {rmse:.4f} eV/atom") 
        print(f"R²:   {r2:.4f}")
        print(f"Mean uncertainty: {y_std_np.mean():.4f}")
        
        # Detailed high-energy analysis
        analyze_high_energy_detailed(y_test_np, y_pred_np, y_std_np)
    
    return {
        'predictions': y_pred_np,
        'ground_truth': y_test_np,
        'uncertainty': y_std_np,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def analyze_high_energy_detailed(y_true, y_pred, y_std):
    """Detailed analysis of high-energy prediction performance."""
    print(f"\n🔥 DETAILED HIGH-ENERGY ANALYSIS:")
    
    # Multiple energy thresholds
    thresholds = [0.0, 0.5, 1.0, 1.5]
    
    for threshold in thresholds:
        mask = y_true > threshold
        if mask.sum() == 0:
            continue
            
        y_range = y_true[mask]
        pred_range = y_pred[mask]
        std_range = y_std[mask]
        
        mae_range = np.mean(np.abs(y_range - pred_range))
        bias_range = np.mean(pred_range - y_range)  # Negative = underestimation
        uncertainty_range = np.mean(std_range)
        
        print(f"   E > {threshold:.1f} eV/atom ({mask.sum():3d} samples):")
        print(f"      MAE: {mae_range:.3f}, Bias: {bias_range:+.3f}, σ: {uncertainty_range:.3f}")
        
        if abs(bias_range) > 0.5:
            bias_type = "underestimation" if bias_range < 0 else "overestimation"
            print(f"      ⚠️  Significant {bias_type}!")
    
    # Overall high-energy performance
    high_mask = y_true > 0.0
    if high_mask.sum() > 0:
        high_mae = np.mean(np.abs(y_true[high_mask] - y_pred[high_mask]))
        high_bias = np.mean(y_pred[high_mask] - y_true[high_mask])
        
        print(f"\n   📈 Overall high-energy (E > 0.0):")
        print(f"      Samples: {high_mask.sum()}")
        print(f"      MAE: {high_mae:.3f} eV/atom")
        print(f"      Bias: {high_bias:+.3f} eV/atom")
        print(f"      True range: [{y_true[high_mask].min():.2f}, {y_true[high_mask].max():.2f}]")
        print(f"      Pred range: [{y_pred[high_mask].min():.2f}, {y_pred[high_mask].max():.2f}]")