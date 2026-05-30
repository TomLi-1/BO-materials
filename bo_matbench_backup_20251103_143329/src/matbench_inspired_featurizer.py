"""
Matbench-inspired featurizer that creates high-quality embeddings
based on successful Matbench approaches without requiring complex dependencies.

This implements feature engineering strategies from top Matbench models:
- Crystal graph-inspired descriptors
- Enhanced elemental property features
- Structure-aware features
"""

import numpy as np
import warnings
from typing import List, Optional
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital, IonProperty
from matminer.featurizers.structure import (
    DensityFeatures, GlobalSymmetryFeatures, RadialDistributionFunction,
    SineCoulombMatrix, PartialRadialDistributionFunction
)
from matminer.featurizers.base import BaseFeaturizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Suppress warnings and fix YAML issue
warnings.filterwarnings("ignore")

# Fix matminer YAML compatibility issue
import sys
if 'matminer' not in sys.modules:
    # Set environment variable to disable problematic matminer features
    import os
    os.environ['MATMINER_DISABLE_FINGERPRINT'] = 'true'


class MatbenchInspiredTransformer:
    """
    High-quality featurizer inspired by top Matbench models.
    
    Combines multiple feature types that have shown success in Matbench:
    1. Enhanced elemental properties (inspired by ALIGNN/CGCNN success)
    2. Crystal structure features (coordination, symmetry)
    3. Graph-inspired local environment features
    4. Dimensionality reduction to create dense embeddings
    """
    
    def __init__(self, 
                 embedding_dim: int = 256,
                 use_structure_features: bool = True,
                 use_dimensionality_reduction: bool = True,
                 standardize_features: bool = True):
        """
        Args:
            embedding_dim: Target dimension for final embeddings
            use_structure_features: Include structure-based features
            use_dimensionality_reduction: Use PCA to reduce to embedding_dim
            standardize_features: Standardize features before PCA
        """
        self.embedding_dim = embedding_dim
        self.use_structure_features = use_structure_features
        self.use_dimensionality_reduction = use_dimensionality_reduction
        self.standardize_features = standardize_features
        
        # Initialize featurizers
        self.composition_featurizers = []
        self.structure_featurizers = []
        self.scaler = None
        self.pca = None
        
        self._setup_featurizers()
    
    def _setup_featurizers(self):
        """Setup all featurizers based on successful Matbench approaches."""
        print("🔧 Setting up Matbench-inspired featurizers...")
        
        # 1. Enhanced Composition Features (inspired by ALIGNN success)
        self.composition_featurizers = [
            # Core elemental properties
            ElementProperty.from_preset("magpie"),
            ElementProperty.from_preset("deml"),
            ElementProperty.from_preset("matminer"),
            
            # Chemical descriptors
            Stoichiometry(),
            ValenceOrbital(),
            IonProperty(),
        ]
        
        # 2. Structure Features (inspired by CGCNN/ALIGNN graph approaches)
        if self.use_structure_features:
            self.structure_featurizers = [
                # Basic structure properties
                DensityFeatures(),
                GlobalSymmetryFeatures(),
                
                # Graph-inspired features
                PartialRadialDistributionFunction(),
                RadialDistributionFunction(),
                
                # Matrix representations (graph-like)
                SineCoulombMatrix(flatten=True),
            ]
        
        print(f"✅ Setup {len(self.composition_featurizers)} composition + {len(self.structure_featurizers)} structure featurizers")
    
    def _extract_composition_features(self, structures: List[Structure]) -> np.ndarray:
        """Extract composition-based features."""
        print(f"🧬 Extracting composition features...")
        
        compositions = [struct.composition for struct in structures]
        all_comp_features = []
        
        for featurizer in self.composition_featurizers:
            try:
                features = featurizer.featurize_many(compositions, ignore_errors=True)
                features = np.array(features)
                
                # Handle NaN values
                if np.any(np.isnan(features)):
                    print(f"   ⚠️  Found NaN in {featurizer.__class__.__name__}, filling with median")
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='median')
                    features = imputer.fit_transform(features)
                
                all_comp_features.append(features)
                print(f"   ✅ {featurizer.__class__.__name__}: {features.shape[1]} features")
                
            except Exception as e:
                print(f"   ❌ {featurizer.__class__.__name__} failed: {e}")
        
        if all_comp_features:
            comp_features = np.hstack(all_comp_features)
            print(f"✅ Total composition features: {comp_features.shape[1]}")
            return comp_features
        else:
            print("❌ No composition features extracted")
            return np.zeros((len(structures), 64))
    
    def _extract_structure_features(self, structures: List[Structure]) -> np.ndarray:
        """Extract structure-based features."""
        if not self.use_structure_features:
            return np.array([]).reshape(len(structures), 0)
        
        print(f"🏗️  Extracting structure features...")
        
        all_struct_features = []
        
        for featurizer in self.structure_featurizers:
            try:
                features = featurizer.featurize_many(structures, ignore_errors=True)
                features = np.array(features)
                
                # Handle NaN values
                if np.any(np.isnan(features)):
                    print(f"   ⚠️  Found NaN in {featurizer.__class__.__name__}, filling with median")
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='median')
                    features = imputer.fit_transform(features)
                
                all_struct_features.append(features)
                print(f"   ✅ {featurizer.__class__.__name__}: {features.shape[1]} features")
                
            except Exception as e:
                print(f"   ❌ {featurizer.__class__.__name__} failed: {e}")
        
        if all_struct_features:
            struct_features = np.hstack(all_struct_features)
            print(f"✅ Total structure features: {struct_features.shape[1]}")
            return struct_features
        else:
            print("⚠️  No structure features extracted")
            return np.array([]).reshape(len(structures), 0)
    
    def _extract_graph_inspired_features(self, structures: List[Structure]) -> np.ndarray:
        """
        Extract graph-inspired local environment features.
        Mimics what ALIGNN/CGCNN do but with simpler approaches.
        """
        print(f"📊 Extracting graph-inspired features...")
        
        graph_features = []
        
        for i, struct in enumerate(structures):
            try:
                # Crystal coordination analysis (graph-inspired)
                cnn = CrystalNN()
                
                # Coordination numbers and local environment
                coord_features = []
                
                for site_idx in range(min(len(struct), 10)):  # Sample first 10 sites
                    try:
                        cn_dict = cnn.get_cn_dict(struct, site_idx)
                        avg_cn = np.mean(list(cn_dict.values())) if cn_dict else 0
                        max_cn = np.max(list(cn_dict.values())) if cn_dict else 0
                        coord_features.extend([avg_cn, max_cn])
                    except:
                        coord_features.extend([0, 0])
                
                # Pad or truncate to fixed size
                coord_features = coord_features[:20]  # Max 20 features
                coord_features.extend([0] * (20 - len(coord_features)))
                
                # Add basic graph metrics
                n_sites = len(struct)
                volume_per_atom = struct.volume / n_sites
                density = struct.density
                
                site_features = coord_features + [n_sites, volume_per_atom, density]
                graph_features.append(site_features)
                
            except Exception as e:
                # Fallback to basic features
                graph_features.append([0] * 23)
        
        graph_features = np.array(graph_features)
        print(f"✅ Graph-inspired features: {graph_features.shape[1]} features")
        return graph_features
    
    def _apply_dimensionality_reduction(self, features: np.ndarray) -> np.ndarray:
        """Apply PCA to reduce to target embedding dimension."""
        if not self.use_dimensionality_reduction or features.shape[1] <= self.embedding_dim:
            return features
        
        print(f"📉 Reducing dimensionality: {features.shape[1]} → {self.embedding_dim}")
        
        # Standardize features
        if self.standardize_features:
            if self.scaler is None:
                self.scaler = StandardScaler()
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        # Apply PCA
        if self.pca is None:
            self.pca = PCA(n_components=self.embedding_dim, random_state=42)
            features_reduced = self.pca.fit_transform(features_scaled)
            
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f"✅ PCA complete, explained variance: {explained_variance:.3f}")
        else:
            features_reduced = self.pca.transform(features_scaled)
        
        return features_reduced
    
    def transform(self, structures: List[Structure]) -> np.ndarray:
        """
        Transform structures to high-quality embeddings.
        
        Args:
            structures: List of pymatgen Structure objects
            
        Returns:
            Feature matrix of shape (n_structures, embedding_dim)
        """
        print(f"\n🎯 Matbench-Inspired Featurization of {len(structures)} structures")
        print(f"Target embedding dimension: {self.embedding_dim}")
        
        # Extract different types of features
        comp_features = self._extract_composition_features(structures)
        struct_features = self._extract_structure_features(structures)
        graph_features = self._extract_graph_inspired_features(structures)
        
        # Combine all features
        all_features = [comp_features]
        if struct_features.shape[1] > 0:
            all_features.append(struct_features)
        if graph_features.shape[1] > 0:
            all_features.append(graph_features)
        
        combined_features = np.hstack(all_features)
        print(f"🔗 Combined features: {combined_features.shape[1]} dimensions")
        
        # Apply dimensionality reduction if needed
        final_features = self._apply_dimensionality_reduction(combined_features)
        
        print(f"✅ Final embeddings: {final_features.shape}")
        print(f"   Feature range: [{final_features.min():.3f}, {final_features.max():.3f}]")
        
        return final_features


def make_matbench_inspired_featurizer(embedding_dim: int = 256,
                                     use_structure_features: bool = True) -> MatbenchInspiredTransformer:
    """
    Create Matbench-inspired featurizer with recommended settings.
    
    Args:
        embedding_dim: Target embedding dimension
        use_structure_features: Include computationally expensive structure features
        
    Returns:
        Configured featurizer
    """
    return MatbenchInspiredTransformer(
        embedding_dim=embedding_dim,
        use_structure_features=use_structure_features,
        use_dimensionality_reduction=True,
        standardize_features=True
    )


if __name__ == "__main__":
    print("🧪 Testing Matbench-inspired featurizer...")
    
    # Create test featurizer
    featurizer = make_matbench_inspired_featurizer(embedding_dim=256)
    print(f"Featurizer created with embedding_dim={featurizer.embedding_dim}")
    print("✅ Matbench-inspired featurizer ready!")