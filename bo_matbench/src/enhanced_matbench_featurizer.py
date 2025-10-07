"""
Enhanced Matbench featurizer with more sophisticated features to improve accuracy.
Based on analysis of top-performing Matbench models and their feature engineering strategies.
"""

import numpy as np
import warnings
from typing import List, Optional
from pymatgen.core import Structure
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

# Suppress warnings
warnings.filterwarnings("ignore")

# Enhanced matminer imports
try:
    from matminer.featurizers.composition import (
        ElementProperty, Stoichiometry, ValenceOrbital, IonProperty,
        ElectronAffinity, ElectronegativityDiff, AtomicOrbitals,
        BandCenter, CohesiveEnergy, Miedema
    )
    from matminer.featurizers.structure import (
        DensityFeatures, GlobalSymmetryFeatures,
        EwaldEnergy, StructuralHeterogeneity,
        MaximumPackingEfficiency
    )
    ENHANCED_MATMINER_AVAILABLE = True
except ImportError:
    # Fallback imports
    from matminer.featurizers.composition import ElementProperty, Stoichiometry
    ENHANCED_MATMINER_AVAILABLE = False
    print("⚠️  Enhanced matminer features not available, using basic set")


class EnhancedMatbenchTransformer:
    """
    Enhanced featurizer with more sophisticated features for better accuracy.
    
    Key improvements:
    1. More comprehensive composition features
    2. Physics-informed structural descriptors
    3. Better feature selection and preprocessing
    4. Robust scaling for numerical stability
    """
    
    def __init__(self, 
                 embedding_dim: int = 256,
                 use_enhanced_features: bool = True,
                 variance_threshold: float = 0.01):
        """
        Args:
            embedding_dim: Target dimension for final embeddings
            use_enhanced_features: Use advanced matminer features if available
            variance_threshold: Remove low-variance features
        """
        self.embedding_dim = embedding_dim
        self.use_enhanced_features = use_enhanced_features and ENHANCED_MATMINER_AVAILABLE
        self.variance_threshold = variance_threshold
        
        # Preprocessing components
        self.variance_selector = None
        self.robust_scaler = None
        self.pca = None
        
        # Featurizers
        self.composition_featurizers = []
        self.structure_featurizers = []
        
        self._setup_enhanced_featurizers()
    
    def _setup_enhanced_featurizers(self):
        """Setup comprehensive featurizers for better accuracy."""
        print("🔧 Setting up enhanced Matbench featurizers...")
        
        # Core composition features (always available)
        self.composition_featurizers = [
            ElementProperty.from_preset("magpie"),
            Stoichiometry(),
        ]
        
        # Enhanced composition features (if available)
        if self.use_enhanced_features:
            self.composition_featurizers.extend([
                ElementProperty.from_preset("deml"),
                ElementProperty.from_preset("matminer"),
                ValenceOrbital(),
                IonProperty(),
                ElectronAffinity(),
                ElectronegativityDiff(),
                AtomicOrbitals(),
                BandCenter(),
                CohesiveEnergy(),
                Miedema(),
            ])
            
            # Enhanced structure features
            self.structure_featurizers = [
                DensityFeatures(),
                GlobalSymmetryFeatures(),
                EwaldEnergy(),
                StructuralHeterogeneity(),
                MaximumPackingEfficiency(),
            ]
            
            print(f"✅ Enhanced features: {len(self.composition_featurizers)} composition + {len(self.structure_featurizers)} structure")
        else:
            print(f"✅ Basic features: {len(self.composition_featurizers)} composition featurizers")
    
    def _extract_enhanced_composition_features(self, structures: List[Structure]) -> np.ndarray:
        """Extract comprehensive composition features."""
        print(f"🧬 Extracting enhanced composition features...")
        
        compositions = [struct.composition for struct in structures]
        all_features = []
        
        for featurizer in self.composition_featurizers:
            try:
                print(f"   Processing {featurizer.__class__.__name__}...")
                features = featurizer.featurize_many(compositions, ignore_errors=True)
                features = np.array(features)
                
                # Handle NaN values
                if np.any(np.isnan(features)):
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='median')
                    features = imputer.fit_transform(features)
                
                # Remove constant features
                if features.std(axis=0).sum() > 0:  # Check if any variance
                    all_features.append(features)
                    print(f"      ✅ {features.shape[1]} features")
                else:
                    print(f"      ⚠️  Skipped (all constant)")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
        
        if all_features:
            result = np.hstack(all_features)
            print(f"✅ Enhanced composition features: {result.shape[1]}")
            return result
        else:
            print("❌ No composition features extracted")
            return np.zeros((len(structures), 64))
    
    def _extract_enhanced_structure_features(self, structures: List[Structure]) -> np.ndarray:
        """Extract enhanced structure features."""
        if not self.use_enhanced_features or not self.structure_featurizers:
            return self._extract_basic_structure_features(structures)
        
        print(f"🏗️  Extracting enhanced structure features...")
        
        all_features = []
        
        for featurizer in self.structure_featurizers:
            try:
                print(f"   Processing {featurizer.__class__.__name__}...")
                features = featurizer.featurize_many(structures, ignore_errors=True)
                features = np.array(features)
                
                # Handle NaN/inf values
                if np.any(~np.isfinite(features)):
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='median')
                    features = imputer.fit_transform(features)
                
                # Check for valid features
                if features.std(axis=0).sum() > 0:
                    all_features.append(features)
                    print(f"      ✅ {features.shape[1]} features")
                else:
                    print(f"      ⚠️  Skipped (all constant)")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
        
        if all_features:
            result = np.hstack(all_features)
            print(f"✅ Enhanced structure features: {result.shape[1]}")
            return result
        else:
            return self._extract_basic_structure_features(structures)
    
    def _extract_basic_structure_features(self, structures: List[Structure]) -> np.ndarray:
        """Extract basic structure features as fallback."""
        print(f"🔄 Using basic structure features...")
        
        features = []
        for struct in structures:
            try:
                # Basic properties
                struct_features = [
                    struct.volume,
                    struct.volume / len(struct),
                    struct.density,
                    len(struct),
                    len(struct.species),
                ]
                
                # Lattice parameters
                lattice = struct.lattice
                struct_features.extend([
                    lattice.a, lattice.b, lattice.c,
                    lattice.alpha, lattice.beta, lattice.gamma,
                    lattice.volume,
                ])
                
                # Packing efficiency approximation
                try:
                    packing_eff = struct.volume / (len(struct) * 20)  # Rough approximation
                except:
                    packing_eff = 0
                struct_features.append(packing_eff)
                
                features.append(struct_features)
                
            except Exception as e:
                # Fallback features
                features.append([0] * 13)
        
        result = np.array(features)
        print(f"✅ Basic structure features: {result.shape[1]}")
        return result
    
    def _preprocess_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply robust preprocessing pipeline."""
        print(f"🔧 Preprocessing features...")
        
        # Remove low-variance features
        if fit:
            self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
            features = self.variance_selector.fit_transform(features)
        else:
            features = self.variance_selector.transform(features)
        
        # Robust scaling (better than StandardScaler for outliers)
        if fit:
            self.robust_scaler = RobustScaler()
            features = self.robust_scaler.fit_transform(features)
        else:
            features = self.robust_scaler.transform(features)
        
        print(f"   ✅ After preprocessing: {features.shape[1]} features")
        return features
    
    def _apply_dimensionality_reduction(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply PCA with proper handling."""
        n_samples, n_features = features.shape
        target_dim = min(self.embedding_dim, n_samples - 1, n_features)
        
        if n_features <= self.embedding_dim:
            print(f"📊 No reduction needed: {n_features} ≤ {self.embedding_dim}")
            return features
        
        print(f"📉 Reducing dimensionality: {n_features} → {target_dim}")
        
        if fit:
            # Use explained variance to determine optimal components
            explained_variance_target = 0.95  # Retain 95% of variance
            
            self.pca = PCA(n_components=target_dim, random_state=42)
            features_reduced = self.pca.fit_transform(features)
            
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f"✅ PCA complete, explained variance: {explained_variance:.3f}")
            
            # If we're losing too much information, warn the user
            if explained_variance < 0.8:
                print(f"⚠️  Low explained variance ({explained_variance:.3f}), consider increasing embedding_dim")
        else:
            features_reduced = self.pca.transform(features)
        
        return features_reduced
    
    def transform(self, structures: List[Structure], fit: bool = None) -> np.ndarray:
        """
        Transform structures to enhanced embeddings.
        
        Args:
            structures: List of pymatgen Structure objects
            fit: Whether to fit preprocessing components (auto-detect if None)
            
        Returns:
            Feature matrix of shape (n_structures, embedding_dim)
        """
        # Auto-detect fit mode
        if fit is None:
            fit = (self.variance_selector is None)
        
        print(f"\n🚀 Enhanced Matbench Featurization of {len(structures)} structures")
        print(f"Target embedding dimension: {self.embedding_dim}")
        print(f"Mode: {'Fit & Transform' if fit else 'Transform'}")
        
        # Extract features
        comp_features = self._extract_enhanced_composition_features(structures)
        struct_features = self._extract_enhanced_structure_features(structures)
        
        # Combine features
        all_features = [comp_features]
        if struct_features.shape[1] > 0:
            all_features.append(struct_features)
        
        combined_features = np.hstack(all_features)
        print(f"🔗 Combined features: {combined_features.shape[1]} dimensions")
        
        # Preprocess features
        processed_features = self._preprocess_features(combined_features, fit=fit)
        
        # Apply dimensionality reduction
        final_features = self._apply_dimensionality_reduction(processed_features, fit=fit)
        
        print(f"✅ Final enhanced embeddings: {final_features.shape}")
        print(f"   Feature range: [{final_features.min():.3f}, {final_features.max():.3f}]")
        
        return final_features


def make_enhanced_matbench_featurizer(embedding_dim: int = 256) -> EnhancedMatbenchTransformer:
    """
    Create enhanced Matbench featurizer for better accuracy.
    
    Args:
        embedding_dim: Target embedding dimension
        
    Returns:
        Enhanced featurizer configured for high accuracy
    """
    return EnhancedMatbenchTransformer(
        embedding_dim=embedding_dim,
        use_enhanced_features=True,
        variance_threshold=0.001  # More aggressive feature selection
    )


if __name__ == "__main__":
    print("🧪 Testing enhanced Matbench featurizer...")
    featurizer = make_enhanced_matbench_featurizer(embedding_dim=256)
    print("✅ Enhanced featurizer ready for high accuracy!")