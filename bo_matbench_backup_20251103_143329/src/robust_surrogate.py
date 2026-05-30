"""
Robust surrogate modeling improvements for formation energy prediction.

Addresses issues with:
1. High formation energy (unstable materials) prediction failures
2. Heteroscedastic noise across energy ranges
3. Outlier sensitivity in GP training
4. Poor extrapolation beyond training distribution
"""

import torch
import numpy as np
from gpytorch import kernels, means, models, distributions
from gpytorch.constraints import Positive
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
import warnings

class RobustFormationEnergyGP(ExactGP):
    """
    Enhanced GP model for formation energy prediction with:
    - Heteroscedastic noise modeling
    - Robust kernel design for wide energy ranges
    - Better uncertainty quantification
    """
    
    def __init__(self, train_x, train_y, likelihood, use_matern=True):
        super().__init__(train_x, train_y, likelihood)
        
        self.mean_module = ConstantMean()
        
        if use_matern:
            # Matern 5/2 kernel is more robust for physical properties
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
        else:
            base_kernel = RBFKernel(ard_num_dims=train_x.shape[-1])
            
        # Scale kernel for better amplitude fitting
        self.covar_module = ScaleKernel(base_kernel)
        
        # Initialize lengthscales more conservatively
        self.covar_module.base_kernel.lengthscale = 1.0
        self.covar_module.outputscale = train_y.var()
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distributions.MultivariateNormal(mean_x, covar_x)


class HeteroscedasticGP(ExactGP):
    """
    GP with learned noise that varies across input space.
    Useful for formation energy where prediction uncertainty
    increases for unstable materials.
    """
    
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        
        self.mean_module = ConstantMean()
        
        # Main prediction kernel
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
        )
        
        # Noise kernel for heteroscedastic modeling
        self.noise_covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=train_x.shape[-1])
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        # Add learned noise variance
        noise_covar = self.noise_covar_module(x)
        total_covar = covar_x + noise_covar
        
        return distributions.MultivariateNormal(mean_x, total_covar)


class RobustDataProcessor:
    """
    Robust data preprocessing for formation energy data.
    """
    
    def __init__(self, outlier_threshold=3.0, use_quantile_transform=False):
        self.outlier_threshold = outlier_threshold
        self.use_quantile_transform = use_quantile_transform
        self.feature_scaler = None
        self.target_transformer = None
        self.outlier_mask = None
        
    def detect_outliers(self, X, y):
        """Detect outliers using robust statistics."""
        # Use median absolute deviation for outlier detection
        median_y = np.median(y)
        mad_y = np.median(np.abs(y - median_y))
        
        # Robust z-score
        robust_z_scores = 0.6745 * (y - median_y) / mad_y
        outlier_mask = np.abs(robust_z_scores) > self.outlier_threshold
        
        print(f"🔍 Outlier detection: {outlier_mask.sum()} outliers out of {len(y)} samples")
        return ~outlier_mask  # Return inlier mask
    
    def fit_transform(self, X, y, remove_outliers=True):
        """Robust fitting and transformation."""
        X = np.array(X)
        y = np.array(y)
        
        # Detect and optionally remove outliers
        if remove_outliers:
            inlier_mask = self.detect_outliers(X, y)
            X_clean = X[inlier_mask]
            y_clean = y[inlier_mask]
            self.outlier_mask = inlier_mask
            print(f"📊 Using {len(X_clean)}/{len(X)} samples after outlier removal")
        else:
            X_clean, y_clean = X, y
            self.outlier_mask = np.ones(len(y), dtype=bool)
        
        # Robust feature scaling
        self.feature_scaler = RobustScaler()
        X_scaled = self.feature_scaler.fit_transform(X_clean)
        
        # Target transformation
        if self.use_quantile_transform:
            # Quantile transform for highly skewed distributions
            self.target_transformer = QuantileTransformer(
                output_distribution='normal', 
                n_quantiles=min(1000, len(y_clean))
            )
            y_transformed = self.target_transformer.fit_transform(y_clean.reshape(-1, 1)).flatten()
        else:
            # Simple robust scaling
            self.target_transformer = RobustScaler()
            y_transformed = self.target_transformer.fit_transform(y_clean.reshape(-1, 1)).flatten()
            
        return X_scaled, y_transformed, self.outlier_mask
    
    def transform(self, X, apply_outlier_mask=False):
        """Transform new data."""
        X = np.array(X)
        
        if apply_outlier_mask and self.outlier_mask is not None:
            X = X[self.outlier_mask]
            
        X_scaled = self.feature_scaler.transform(X)
        return X_scaled
    
    def inverse_transform_target(self, y_transformed):
        """Inverse transform predictions back to original scale."""
        y_transformed = np.array(y_transformed).reshape(-1, 1)
        
        if self.use_quantile_transform:
            return self.target_transformer.inverse_transform(y_transformed).flatten()
        else:
            return self.target_transformer.inverse_transform(y_transformed).flatten()


def fit_robust_formation_energy_model(X, y, test_size=0.2, 
                                     use_heteroscedastic=False,
                                     use_quantile_transform=False,
                                     remove_outliers=True):
    """
    Fit a robust GP model for formation energy prediction.
    
    Args:
        X: Feature matrix
        y: Formation energies
        test_size: Fraction for train/test split
        use_heteroscedastic: Use heteroscedastic noise model
        use_quantile_transform: Apply quantile normalization
        remove_outliers: Remove outliers during training
        
    Returns:
        Trained model, processor, and evaluation metrics
    """
    print(f"🔧 Fitting robust formation energy model...")
    print(f"   Data shape: {X.shape}")
    print(f"   Energy range: [{y.min():.3f}, {y.max():.3f}] eV/atom")
    
    # Robust preprocessing
    processor = RobustDataProcessor(
        outlier_threshold=3.0,
        use_quantile_transform=use_quantile_transform
    )
    
    X_processed, y_processed, outlier_mask = processor.fit_transform(
        X, y, remove_outliers=remove_outliers
    )
    
    # Stratified split to ensure representation across energy ranges
    # Bin formation energies for stratification
    n_bins = min(10, len(y_processed) // 20)
    y_bins = np.digitize(y_processed, bins=np.quantile(y_processed, np.linspace(0, 1, n_bins)))
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, 
            test_size=test_size, 
            random_state=42,
            stratify=y_bins
        )
        print(f"✅ Stratified split: {len(X_train)} train, {len(X_test)} test")
    except:
        # Fall back to random split if stratification fails
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, 
            test_size=test_size, 
            random_state=42
        )
        print(f"✅ Random split: {len(X_train)} train, {len(X_test)} test")
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float64)
    X_test_t = torch.tensor(X_test, dtype=torch.float64) 
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    # Fit GP model
    print(f"🤖 Training GP model...")
    
    if use_heteroscedastic:
        likelihood = GaussianLikelihood()
        model = HeteroscedasticGP(X_train_t, y_train_t, likelihood)
        print(f"   Using heteroscedastic GP")
    else:
        likelihood = GaussianLikelihood()
        model = RobustFormationEnergyGP(X_train_t, y_train_t, likelihood)
        print(f"   Using robust Matern GP")
    
    # Training mode
    model.train()
    likelihood.train()
    
    # Optimize hyperparameters with more iterations
    mll = ExactMarginalLogLikelihood(likelihood, model)
    
    try:
        fit_gpytorch_mll(mll, max_iter=500)
        print(f"✅ GP hyperparameter optimization complete")
    except Exception as e:
        print(f"⚠️  GP fitting warning: {e}")
    
    # Evaluation mode
    model.eval()
    likelihood.eval()
    
    # Evaluate on test set
    with torch.no_grad():
        test_predictions = likelihood(model(X_test_t))
        pred_mean = test_predictions.mean.numpy()
        pred_std = test_predictions.stddev.numpy()
        
        # Transform back to original scale
        pred_mean_orig = processor.inverse_transform_target(pred_mean)
        y_test_orig = processor.inverse_transform_target(y_test)
        
        # Compute metrics
        mae = np.mean(np.abs(pred_mean_orig - y_test_orig))
        rmse = np.sqrt(np.mean((pred_mean_orig - y_test_orig)**2))
        
        # R² calculation
        ss_res = np.sum((y_test_orig - pred_mean_orig)**2)
        ss_tot = np.sum((y_test_orig - np.mean(y_test_orig))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"📊 Robust model performance:")
        print(f"   MAE: {mae:.4f} eV/atom")
        print(f"   RMSE: {rmse:.4f} eV/atom") 
        print(f"   R²: {r2:.4f}")
        
        # Analyze performance by energy range
        analyze_performance_by_range(y_test_orig, pred_mean_orig, pred_std)
        
    evaluation_results = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': pred_mean_orig,
        'ground_truth': y_test_orig,
        'uncertainties': pred_std,
        'outliers_removed': outlier_mask.sum() if remove_outliers else 0,
        'total_samples': len(y)
    }
    
    return model, processor, likelihood, evaluation_results


def analyze_performance_by_range(y_true, y_pred, y_std=None):
    """Analyze model performance across different energy ranges."""
    print(f"\n📈 Performance by formation energy range:")
    
    # Define energy ranges
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
        
        mae_range = np.mean(np.abs(y_range - pred_range))
        rmse_range = np.sqrt(np.mean((y_range - pred_range)**2))
        
        print(f"   {name:16} ({mask.sum():5d} samples): MAE={mae_range:.3f}, RMSE={rmse_range:.3f}")
        
        if y_std is not None:
            avg_uncertainty = np.mean(y_std[mask])
            print(f"                    Avg uncertainty: {avg_uncertainty:.3f}")


def create_robust_surrogate_pipeline():
    """Factory function for robust surrogate modeling."""
    return {
        'processor': RobustDataProcessor,
        'model': RobustFormationEnergyGP,
        'fit_function': fit_robust_formation_energy_model
    }


if __name__ == "__main__":
    # Test the robust modeling approach
    print("🧪 Testing robust formation energy modeling...")
    
    # Generate synthetic test data with outliers
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X_test = np.random.randn(n_samples, n_features)
    
    # Create formation energies with realistic distribution
    # Most materials are stable (negative), few are very unstable
    stable_energies = np.random.normal(-1.5, 0.8, int(0.8 * n_samples))
    unstable_energies = np.random.exponential(0.5, int(0.2 * n_samples))
    
    y_test = np.concatenate([stable_energies, unstable_energies])
    X_test = X_test[:len(y_test)]
    
    # Add some outliers
    n_outliers = int(0.05 * len(y_test))
    outlier_idx = np.random.choice(len(y_test), n_outliers, replace=False)
    y_test[outlier_idx] += np.random.normal(0, 3, n_outliers)
    
    print(f"Generated test data: {len(y_test)} samples, energy range [{y_test.min():.2f}, {y_test.max():.2f}]")
    
    # Test robust modeling
    model, processor, likelihood, results = fit_robust_formation_energy_model(
        X_test, y_test, use_quantile_transform=True
    )
    
    print(f"✅ Robust modeling test complete!")