"""
Energy-aware surrogate modeling approaches to address high formation energy prediction failures.

This module implements alternative strategies to the robust modeling approach:
1. Ensemble GP with energy-stratified training
2. Custom kernels with energy-dependent length scales
3. Multi-fidelity GP with uncertainty-weighted predictions
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel
import gpytorch
from typing import List, Tuple, Dict, Any


class EnergyStratifiedGP:
    """
    Ensemble of GP models trained on different energy ranges to handle
    the formation energy prediction more effectively across the full range.
    """
    
    def __init__(self, energy_thresholds: List[float] = [-1.0, 0.5]):
        """
        Args:
            energy_thresholds: Energy values to split the data into ranges
                              e.g., [-1.0, 0.5] creates ranges: 
                              stable (<-1.0), intermediate (-1.0 to 0.5), unstable (>0.5)
        """
        self.energy_thresholds = sorted(energy_thresholds)
        self.models = {}
        self.data_stats = {}
        
    def _get_energy_range_idx(self, energy: float) -> int:
        """Get the energy range index for a given energy value."""
        for i, threshold in enumerate(self.energy_thresholds):
            if energy < threshold:
                return i
        return len(self.energy_thresholds)  # Highest energy range
    
    def fit(self, train_X: torch.Tensor, train_y: torch.Tensor):
        """
        Fit separate GP models for different energy ranges.
        """
        print(f"🔧 Fitting energy-stratified GP ensemble...")
        
        # Convert to numpy for easier processing
        X_np = train_X.numpy()
        y_np = train_y.numpy()
        
        # Group data by energy ranges
        energy_ranges = {}
        for i in range(len(self.energy_thresholds) + 1):
            energy_ranges[i] = {'X': [], 'y': [], 'indices': []}
        
        # Assign each sample to an energy range
        for idx, energy in enumerate(y_np):
            range_idx = self._get_energy_range_idx(energy)
            energy_ranges[range_idx]['X'].append(X_np[idx])
            energy_ranges[range_idx]['y'].append(energy)
            energy_ranges[range_idx]['indices'].append(idx)
        
        # Convert to arrays and fit models
        for range_idx, data in energy_ranges.items():
            if len(data['y']) < 5:  # Skip ranges with too few samples
                print(f"   Range {range_idx}: Skipped (only {len(data['y'])} samples)")
                continue
                
            X_range = torch.tensor(np.array(data['X']), dtype=torch.float32)
            y_range = torch.tensor(np.array(data['y']), dtype=torch.float32)
            
            # Store statistics
            y_min, y_max = float(y_range.min()), float(y_range.max())
            self.data_stats[range_idx] = {
                'count': len(data['y']),
                'energy_range': (y_min, y_max),
                'mean_energy': float(y_range.mean()),
                'std_energy': float(y_range.std())
            }
            
            # Fit GP model for this range
            feat_dim = X_range.size(-1)
            output_dim = 1
            
            gp = SingleTaskGP(
                X_range,
                y_range.unsqueeze(-1),
                input_transform=Normalize(feat_dim),
                outcome_transform=Standardize(output_dim),
            )
            
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            
            try:
                fit_gpytorch_mll(mll)
                self.models[range_idx] = gp
                
                range_name = self._get_range_name(range_idx)
                print(f"   Range {range_idx} ({range_name}): {len(data['y'])} samples, "
                      f"energy [{y_min:.2f}, {y_max:.2f}] eV/atom")
                
            except Exception as e:
                print(f"   Range {range_idx}: Failed to fit ({e})")
        
        print(f"✅ Fitted {len(self.models)} energy-stratified models")
        return self
    
    def _get_range_name(self, range_idx: int) -> str:
        """Get descriptive name for energy range."""
        if range_idx == 0:
            return f"Very Stable (< {self.energy_thresholds[0]})"
        elif range_idx == len(self.energy_thresholds):
            return f"Unstable (> {self.energy_thresholds[-1]})"
        else:
            return f"Intermediate ({self.energy_thresholds[range_idx-1]:.1f} to {self.energy_thresholds[range_idx]:.1f})"
    
    def predict(self, X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make ensemble predictions by weighting models based on energy ranges.
        
        For test points, we don't know the true energy, so we:
        1. Get predictions from all available models
        2. Weight predictions based on model confidence and energy likelihood
        """
        all_models = list(self.models.values())
        if not all_models:
            raise ValueError("No models fitted!")
        
        predictions = []
        uncertainties = []
        
        for model in all_models:
            model.eval()
            with torch.no_grad():
                posterior = model.posterior(X_test)
                pred = posterior.mean.squeeze(-1)
                uncertainty = posterior.variance.sqrt().squeeze(-1)
                predictions.append(pred)
                uncertainties.append(uncertainty)
        
        # Stack predictions and uncertainties
        pred_stack = torch.stack(predictions)  # (n_models, n_test)
        unc_stack = torch.stack(uncertainties)  # (n_models, n_test)
        
        # Weight predictions by inverse uncertainty (more confident models get higher weight)
        weights = 1.0 / (unc_stack + 1e-6)  # Add small epsilon to avoid division by zero
        weights = weights / weights.sum(dim=0, keepdim=True)  # Normalize weights
        
        # Weighted ensemble prediction
        ensemble_pred = (weights * pred_stack).sum(dim=0)
        
        # Uncertainty estimation: combine individual uncertainties
        ensemble_uncertainty = torch.sqrt((weights**2 * unc_stack**2).sum(dim=0))
        
        return ensemble_pred, ensemble_uncertainty


class EnergyAwareKernel(gpytorch.kernels.Kernel):
    """
    Custom kernel that adapts length scales based on formation energy regions.
    Uses a mixture of RBF kernels with different length scales for different energy ranges.
    """
    
    def __init__(self, base_kernel=None, energy_thresholds=[-1.0, 0.5], **kwargs):
        super().__init__(**kwargs)
        
        self.energy_thresholds = energy_thresholds
        if base_kernel is None:
            base_kernel = RBFKernel()
        
        # Create multiple kernels with different length scales
        self.stable_kernel = ScaleKernel(base_kernel)    # For stable materials
        self.unstable_kernel = ScaleKernel(base_kernel)  # For unstable materials
        
        # Parameters for energy-dependent mixing
        self.register_parameter(
            name="mixing_weight",
            parameter=torch.nn.Parameter(torch.tensor(0.5))
        )
    
    def forward(self, x1, x2, diag=False, **params):
        # This is a simplified implementation - in practice, you'd need
        # access to energy values to properly weight the kernels
        # For now, use a simple mixture
        
        k_stable = self.stable_kernel(x1, x2, diag=diag, **params)
        k_unstable = self.unstable_kernel(x1, x2, diag=diag, **params)
        
        # Simple mixture (in practice, this would be energy-dependent)
        weight = torch.sigmoid(self.mixing_weight)
        return weight * k_stable + (1 - weight) * k_unstable


def fit_energy_aware_surrogate(train_X: torch.Tensor, train_y: torch.Tensor, 
                              method: str = "stratified") -> Any:
    """
    Fit energy-aware surrogate model using specified method.
    
    Args:
        train_X: Training features
        train_y: Training formation energies
        method: "stratified" for ensemble approach, "adaptive_kernel" for custom kernel
        
    Returns:
        Fitted model
    """
    
    if method == "stratified":
        # Use energy-stratified ensemble approach
        model = EnergyStratifiedGP(energy_thresholds=[-1.5, -0.5, 0.5])
        return model.fit(train_X, train_y)
        
    elif method == "adaptive_kernel":
        # Use custom energy-adaptive kernel (simplified implementation)
        print("🔧 Fitting GP with energy-adaptive kernel...")
        
        feat_dim = train_X.size(-1)
        output_dim = 1
        
        # Create custom kernel
        custom_kernel = EnergyAwareKernel()
        
        gp = SingleTaskGP(
            train_X,
            train_y.unsqueeze(-1),
            covar_module=custom_kernel,
            input_transform=Normalize(feat_dim),
            outcome_transform=Standardize(output_dim),
        )
        
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        print("✅ Energy-adaptive kernel GP fitted")
        return gp
    
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_energy_aware_model(model, X_test: torch.Tensor, y_test: torch.Tensor,
                               feat_idx: np.ndarray = None, verbose: bool = True) -> Dict:
    """
    Evaluate energy-aware model predictions.
    """
    # Select features if needed
    if feat_idx is not None:
        X_test_selected = X_test[:, feat_idx]
    else:
        X_test_selected = X_test
    
    # Get predictions based on model type
    if isinstance(model, EnergyStratifiedGP):
        y_pred, y_std = model.predict(X_test_selected)
        y_pred_np = y_pred.numpy()
        y_std_np = y_std.numpy()
    else:
        # Standard GP model
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X_test_selected)
            y_pred = posterior.mean.squeeze(-1)
            y_std = posterior.variance.sqrt().squeeze(-1)
            y_pred_np = y_pred.numpy()
            y_std_np = y_std.numpy()
    
    # Convert test data to numpy
    y_test_np = y_test.numpy()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_np, y_pred_np)
    mse = mean_squared_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_np, y_pred_np)
    
    if verbose:
        print(f"\n📊 Energy-Aware Model Evaluation:")
        print(f"MAE:  {mae:.4f} eV/atom")
        print(f"RMSE: {rmse:.4f} eV/atom") 
        print(f"R²:   {r2:.4f}")
        print(f"Mean uncertainty: {y_std_np.mean():.4f}")
        
        # Analyze performance by energy range
        analyze_energy_range_performance(y_test_np, y_pred_np, y_std_np)
    
    return {
        'predictions': y_pred_np,
        'ground_truth': y_test_np,
        'uncertainty': y_std_np,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def analyze_energy_range_performance(y_true, y_pred, y_std):
    """Analyze model performance across formation energy ranges."""
    print(f"\n📈 Performance by formation energy range:")
    
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
        
        if mae_range > 1.0:
            print(f"      ⚠️  High error in {name} range!")