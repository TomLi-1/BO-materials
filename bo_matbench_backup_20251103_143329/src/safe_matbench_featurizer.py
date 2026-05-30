"""
Safe Matbench-inspired featurizer that avoids YAML issues in matminer.
Uses only stable matminer features and adds custom structure descriptors.
"""

import numpy as np
import warnings
from typing import List, Optional
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Suppress warnings
warnings.filterwarnings("ignore")

# Safe matminer imports (avoid problematic ones)
try:
    from matminer.featurizers.composition import ElementProperty, Stoichiometry
    MATMINER_AVAILABLE = True
except ImportError:
    MATMINER_AVAILABLE = False
    print("⚠️  Matminer not available")


class SafeMatbenchTransformer:
    """
    Safe featurizer inspired by Matbench approaches without YAML dependencies.
    
    Uses only stable matminer features plus custom structure descriptors.
    """
    
    def __init__(self, 
                 embedding_dim: int = 256,
                 use_custom_structure_features: bool = True):
        """
        Args:
            embedding_dim: Target dimension for final embeddings
            use_custom_structure_features: Include custom structure features
        """
        self.embedding_dim = embedding_dim
        self.use_custom_structure_features = use_custom_structure_features
        
        # Initialize components
        self.composition_featurizers = []
        self.scaler = None
        self.pca = None
        
        self._setup_safe_featurizers()
    
    def _setup_safe_featurizers(self):
        """Setup safe featurizers that don't cause YAML issues."""
        print("🔧 Setting up safe Matbench-inspired featurizers...")
        
        if MATMINER_AVAILABLE:
            # Use only the most stable matminer features
            try:
                self.composition_featurizers = [
                    ElementProperty.from_preset("magpie"),
                    Stoichiometry(),
                ]
                print("✅ Safe matminer featurizers loaded")
            except Exception as e:
                print(f"⚠️  Matminer featurizer setup failed: {e}")
                self.composition_featurizers = []
        else:
            print("⚠️  Using fallback features (no matminer)")
    
    def _extract_safe_composition_features(self, structures: List[Structure]) -> np.ndarray:
        """Extract composition features using safe featurizers."""
        print(f"🧬 Extracting safe composition features...")
        
        if not self.composition_featurizers:
            return self._extract_fallback_composition_features(structures)
        
        compositions = [struct.composition for struct in structures]
        all_features = []
        
        for featurizer in self.composition_featurizers:
            try:
                features = featurizer.featurize_many(compositions, ignore_errors=True)
                features = np.array(features)
                
                # Handle NaN values
                if np.any(np.isnan(features)):
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='median')
                    features = imputer.fit_transform(features)
                
                all_features.append(features)
                print(f"   ✅ {featurizer.__class__.__name__}: {features.shape[1]} features")
                
            except Exception as e:
                print(f"   ❌ {featurizer.__class__.__name__} failed: {e}")
        
        if all_features:
            result = np.hstack(all_features)
            print(f"✅ Safe composition features: {result.shape[1]}")
            return result
        else:
            return self._extract_fallback_composition_features(structures)
    
    def _extract_fallback_composition_features(self, structures: List[Structure]) -> np.ndarray:
        """Extract basic composition features without matminer."""
        print("🔄 Using fallback composition features...")
        
        features = []
        for struct in structures:
            comp = struct.composition
            
            # Basic composition statistics
            comp_features = [
                comp.num_atoms,
                comp.weight,
                len(comp.elements),
                comp.get_atomic_fraction_amounts().std(),
                comp.anonymized_formula != comp.formula,  # Boolean for complexity
            ]
            
            # Elemental properties (hardcoded basic ones)
            atomic_nums = [el.Z for el in comp.elements]
            atomic_weights = [el.atomic_mass for el in comp.elements]
            
            comp_features.extend([
                np.mean(atomic_nums),
                np.std(atomic_nums),
                np.mean(atomic_weights),
                np.std(atomic_weights),
                max(atomic_nums),
                min(atomic_nums),
            ])
            
            features.append(comp_features)
        
        result = np.array(features)
        print(f"✅ Fallback composition features: {result.shape[1]}")
        return result
    
    def _extract_custom_structure_features(self, structures: List[Structure]) -> np.ndarray:
        """Extract custom structure features without problematic dependencies."""
        if not self.use_custom_structure_features:
            return np.array([]).reshape(len(structures), 0)
        
        print(f"🏗️  Extracting custom structure features...")
        
        features = []
        
        for struct in structures:
            try:
                # Basic structure properties
                struct_features = [
                    struct.volume,
                    struct.volume / len(struct),  # Volume per atom
                    struct.density,
                    len(struct),  # Number of atoms
                    len(struct.species),  # Number of unique species
                ]
                
                # Lattice properties
                lattice = struct.lattice
                struct_features.extend([
                    lattice.a, lattice.b, lattice.c,
                    lattice.alpha, lattice.beta, lattice.gamma,
                    lattice.volume,
                ])
                
                # Crystal system info (encoded) - simplified
                try:
                    spg_info = struct.get_space_group_info()
                    if hasattr(spg_info[0], 'crystal_system'):
                        crystal_system_map = {
                            'cubic': 1, 'tetragonal': 2, 'orthorhombic': 3,
                            'hexagonal': 4, 'trigonal': 5, 'monoclinic': 6, 'triclinic': 7
                        }
                        cs_encoded = crystal_system_map.get(spg_info[0].crystal_system, 0)
                    else:
                        cs_encoded = 0
                except:
                    cs_encoded = 0
                struct_features.append(cs_encoded)
                
                # Simple coordination analysis (safe version)
                try:
                    # Just use basic distance statistics instead of complex coordination
                    distances = []
                    sample_sites = min(3, len(struct))  # Even smaller sample
                    
                    for i in range(sample_sites):
                        try:
                            neighbors = struct.get_all_neighbors(3.5, include_index=False)  # Smaller cutoff
                            if neighbors and len(neighbors) > i:
                                site_distances = [d[1] for d in neighbors[i][:5]]  # Max 5 neighbors
                                if site_distances:
                                    distances.extend([np.mean(site_distances), len(site_distances)])
                                else:
                                    distances.extend([0, 0])
                            else:
                                distances.extend([0, 0])
                        except:
                            distances.extend([0, 0])
                    
                    # Pad to fixed size
                    distances = distances[:6]  # Max 6 features (3 sites × 2 features)
                    distances.extend([0] * (6 - len(distances)))
                    struct_features.extend(distances)
                    
                except Exception as e:
                    # Fallback coordination features
                    struct_features.extend([0] * 6)
                
                features.append(struct_features)
                
            except Exception as e:
                print(f"   ⚠️  Structure {len(features)} failed: {e}")
                # Use zero features as fallback
                features.append([0] * 20)  # Updated expected feature count: 5+7+1+6=19, round to 20
        
        result = np.array(features)
        print(f"✅ Custom structure features: {result.shape[1]}")
        return result
    
    def _apply_dimensionality_reduction(self, features: np.ndarray) -> np.ndarray:
        """Apply PCA to reduce to target embedding dimension."""
        n_samples, n_features = features.shape
        
        # Check if we can actually do PCA
        max_components = min(n_samples, n_features)
        target_dim = min(self.embedding_dim, max_components)
        
        if n_features <= self.embedding_dim:
            print(f"📊 No reduction needed: {n_features} ≤ {self.embedding_dim}")
            return features
        
        if n_samples < self.embedding_dim:
            print(f"⚠️  Too few samples ({n_samples}) for target dimension ({self.embedding_dim})")
            print(f"📉 Reducing to maximum possible: {n_features} → {target_dim}")
        else:
            print(f"📉 Reducing dimensionality: {n_features} → {target_dim}")
        
        # Standardize features
        if self.scaler is None:
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)
        
        # Apply PCA with safe number of components
        if self.pca is None:
            self.pca = PCA(n_components=target_dim, random_state=42)
            features_reduced = self.pca.fit_transform(features_scaled)
            
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f"✅ PCA complete, explained variance: {explained_variance:.3f}")
        else:
            features_reduced = self.pca.transform(features_scaled)
        
        return features_reduced
    
    def transform(self, structures: List[Structure]) -> np.ndarray:
        """
        Transform structures to embeddings.
        
        Args:
            structures: List of pymatgen Structure objects
            
        Returns:
            Feature matrix of shape (n_structures, embedding_dim)
        """
        print(f"\n🛡️  Safe Matbench Featurization of {len(structures)} structures")
        print(f"Target embedding dimension: {self.embedding_dim}")
        
        # Extract features safely
        comp_features = self._extract_safe_composition_features(structures)
        struct_features = self._extract_custom_structure_features(structures)
        
        # Combine features
        all_features = [comp_features]
        if struct_features.shape[1] > 0:
            all_features.append(struct_features)
        
        combined_features = np.hstack(all_features)
        print(f"🔗 Combined features: {combined_features.shape[1]} dimensions")
        
        # Apply dimensionality reduction
        final_features = self._apply_dimensionality_reduction(combined_features)
        
        print(f"✅ Final embeddings: {final_features.shape}")
        return final_features


def make_safe_matbench_featurizer(embedding_dim: int = 256) -> SafeMatbenchTransformer:
    """
    Create safe Matbench-inspired featurizer.
    
    Args:
        embedding_dim: Target embedding dimension
        
    Returns:
        Configured safe featurizer
    """
    return SafeMatbenchTransformer(
        embedding_dim=embedding_dim,
        use_custom_structure_features=True
    )


if __name__ == "__main__":
    print("🧪 Testing safe Matbench-inspired featurizer...")
    featurizer = make_safe_matbench_featurizer(embedding_dim=256)
    print("✅ Safe featurizer ready!")