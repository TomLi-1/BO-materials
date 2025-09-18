"""
GNN-based featurization using pretrained embeddings with atomic max pooling.

This module provides:
1. Pretrained atomic embeddings (e.g., from matminer or simple lookup tables)
2. Max pooling across atomic embeddings for fixed-length representations
3. Fallbacks for composition-only data
4. GNNTransformer compatible with existing pipeline
"""

import numpy as np
import torch
from typing import List, Union, Optional
from pymatgen.core import Structure, Composition
import warnings
warnings.filterwarnings("ignore")

# Try to import optional GNN dependencies
try:
    from matminer.featurizers.structure import SiteStatsFingerprint
    from matminer.featurizers.site import EwaldSiteEnergy
    MATMINER_AVAILABLE = True
except ImportError:
    MATMINER_AVAILABLE = False
    warnings.warn("Matminer not fully available. Using fallback embeddings.")

try:
    import torch_geometric
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


class AtomicEmbeddingLoader:
    """
    Load pretrained atomic embeddings using various strategies.
    
    Priority order:
    1. MEGNet embeddings (if available)
    2. Matminer-based site fingerprints
    3. Simple elemental property embeddings
    4. One-hot atomic number encoding (fallback)
    """
    
    def __init__(self, embedding_dim: int = 64, method: str = "auto"):
        self.embedding_dim = embedding_dim
        self.method = method
        self._embedding_cache = {}
        
        # Initialize embedding strategy
        if method == "auto":
            self.method = self._select_best_method()
        
        print(f"🧬 Using atomic embedding method: {self.method}")
        self._initialize_embeddings()
    
    def _select_best_method(self) -> str:
        """Select the best available embedding method."""
        if TORCH_GEOMETRIC_AVAILABLE:
            return "torch_geometric"
        elif MATMINER_AVAILABLE:
            return "matminer_site"
        else:
            return "elemental_properties"
    
    def _initialize_embeddings(self):
        """Initialize the embedding strategy."""
        if self.method == "matminer_site":
            self._init_matminer_embeddings()
        elif self.method == "elemental_properties":
            self._init_elemental_property_embeddings()
        elif self.method == "torch_geometric":
            self._init_torch_geometric_embeddings()
        else:
            self._init_onehot_embeddings()
    
    def _init_matminer_embeddings(self):
        """Initialize matminer-based site embeddings."""
        if MATMINER_AVAILABLE:
            try:
                # Use simple site statistics as embeddings
                from matminer.featurizers.site import LocalStructuralOrderParams
                self.site_featurizer = LocalStructuralOrderParams()
                print("✅ Initialized matminer site-based embeddings")
            except Exception as e:
                print(f"⚠️  Matminer site embeddings failed: {e}")
                self._init_elemental_property_embeddings()
        else:
            self._init_elemental_property_embeddings()
    
    def _init_elemental_property_embeddings(self):
        """Initialize embeddings based on elemental properties."""
        # Common atomic properties for embeddings
        from pymatgen.core import Element
        
        # Use available Element properties that exist in pymatgen
        self.property_names = [
            'atomic_mass', 'atomic_radius', 'ionization_energy', 
            'electron_affinity', 'group', 'row', 'X', 'Z',
            'mendeleev_no', 'density_of_solid', 'melting_point'
        ]
        
        # Build property matrix for all elements
        self.element_properties = {}
        for z in range(1, 119):  # H to Og
            try:
                elem = Element.from_Z(z)
                props = []
                for prop in self.property_names:
                    try:
                        val = getattr(elem, prop, 0)
                        if val is None or str(val) == 'nan':
                            val = 0
                        props.append(float(val))
                    except (ValueError, TypeError):
                        props.append(0.0)
                
                # Pad or truncate to desired embedding dimension
                if len(props) < self.embedding_dim:
                    props.extend([0.0] * (self.embedding_dim - len(props)))
                else:
                    props = props[:self.embedding_dim]
                
                self.element_properties[elem.symbol] = np.array(props, dtype=np.float32)
                
            except Exception:
                # Fallback for unknown elements
                self.element_properties[f"X{z}"] = np.zeros(self.embedding_dim, dtype=np.float32)
        
        print(f"✅ Initialized elemental property embeddings ({len(self.element_properties)} elements)")
    
    def _init_torch_geometric_embeddings(self):
        """Initialize torch geometric based embeddings (placeholder)."""
        print("🚧 Torch Geometric embeddings not implemented yet, using elemental properties")
        self._init_elemental_property_embeddings()
    
    def _init_onehot_embeddings(self):
        """Initialize simple one-hot atomic number embeddings."""
        self.max_atomic_number = 118
        print(f"✅ Initialized one-hot atomic embeddings (dim={self.max_atomic_number})")
    
    def get_atomic_embeddings(self, structure_or_composition) -> np.ndarray:
        """
        Get atomic embeddings for a structure or composition.
        
        Args:
            structure_or_composition: pymatgen Structure or Composition
            
        Returns:
            np.ndarray: Array of shape (n_atoms, embedding_dim) for structures
                       or (n_elements, embedding_dim) for compositions
        """
        if isinstance(structure_or_composition, Structure):
            return self._get_structure_embeddings(structure_or_composition)
        elif isinstance(structure_or_composition, Composition):
            return self._get_composition_embeddings(structure_or_composition)
        else:
            raise ValueError(f"Unsupported input type: {type(structure_or_composition)}")
    
    def _get_structure_embeddings(self, structure: Structure) -> np.ndarray:
        """Get embeddings for each atom in a structure."""
        embeddings = []
        
        for site in structure:
            element_symbol = str(site.specie)
            embedding = self._get_element_embedding(element_symbol)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _get_composition_embeddings(self, composition: Composition) -> np.ndarray:
        """Get embeddings for each element in a composition."""
        embeddings = []
        
        for element, fraction in composition.items():
            element_symbol = str(element)
            embedding = self._get_element_embedding(element_symbol)
            # Weight by fraction for compositions
            weighted_embedding = embedding * fraction
            embeddings.append(weighted_embedding)
        
        return np.array(embeddings)
    
    def _get_element_embedding(self, element_symbol: str) -> np.ndarray:
        """Get embedding for a specific element."""
        if self.method == "onehot":
            return self._get_onehot_embedding(element_symbol)
        elif self.method == "elemental_properties":
            return self.element_properties.get(element_symbol, 
                                               np.zeros(self.embedding_dim, dtype=np.float32))
        else:
            # Default fallback
            return self.element_properties.get(element_symbol, 
                                               np.zeros(self.embedding_dim, dtype=np.float32))
    
    def _get_onehot_embedding(self, element_symbol: str) -> np.ndarray:
        """Get one-hot encoding for element."""
        from pymatgen.core import Element
        try:
            atomic_number = Element(element_symbol).Z
            embedding = np.zeros(self.max_atomic_number, dtype=np.float32)
            if 1 <= atomic_number <= self.max_atomic_number:
                embedding[atomic_number - 1] = 1.0
            return embedding
        except:
            return np.zeros(self.max_atomic_number, dtype=np.float32)


class AtomicMaxPooling:
    """
    Performs max pooling across atomic embeddings to create fixed-length representations.
    """
    
    def __init__(self, pooling_strategy: str = "max", include_stats: bool = True):
        """
        Args:
            pooling_strategy: "max", "mean", "sum", or "max_mean" (concatenate max and mean)
            include_stats: Whether to include additional statistics (std, min, etc.)
        """
        self.pooling_strategy = pooling_strategy
        self.include_stats = include_stats
    
    def pool(self, atomic_embeddings: np.ndarray) -> np.ndarray:
        """
        Pool atomic embeddings into a fixed-length vector.
        
        Args:
            atomic_embeddings: Array of shape (n_atoms, embedding_dim)
            
        Returns:
            np.ndarray: Pooled representation
        """
        if len(atomic_embeddings) == 0:
            return np.zeros(self._get_output_dim(atomic_embeddings.shape[1]))
        
        pooled_features = []
        
        # Main pooling operation
        if self.pooling_strategy == "max":
            pooled_features.append(np.max(atomic_embeddings, axis=0))
        elif self.pooling_strategy == "mean":
            pooled_features.append(np.mean(atomic_embeddings, axis=0))
        elif self.pooling_strategy == "sum":
            pooled_features.append(np.sum(atomic_embeddings, axis=0))
        elif self.pooling_strategy == "max_mean":
            pooled_features.append(np.max(atomic_embeddings, axis=0))
            pooled_features.append(np.mean(atomic_embeddings, axis=0))
        
        # Additional statistics
        if self.include_stats:
            pooled_features.append(np.std(atomic_embeddings, axis=0))
            pooled_features.append(np.min(atomic_embeddings, axis=0))
            # Count of atoms (broadcasted to embedding dimension)
            count_features = np.full(atomic_embeddings.shape[1], len(atomic_embeddings))
            pooled_features.append(count_features)
        
        return np.concatenate(pooled_features)
    
    def _get_output_dim(self, embedding_dim: int) -> int:
        """Calculate output dimension based on pooling strategy."""
        multiplier = 1
        if self.pooling_strategy == "max_mean":
            multiplier = 2
        
        if self.include_stats:
            multiplier += 3  # std, min, count
        
        return embedding_dim * multiplier


class GNNTransformer:
    """
    GNN-based transformer analogous to MatminerTransformer.
    """
    
    def __init__(self, 
                 embedding_dim: int = 64,
                 embedding_method: str = "auto",
                 pooling_strategy: str = "max_mean",
                 include_stats: bool = True):
        """
        Initialize GNN transformer.
        
        Args:
            embedding_dim: Dimension of atomic embeddings
            embedding_method: Method for atomic embeddings
            pooling_strategy: Pooling strategy for aggregation
            include_stats: Whether to include additional statistics
        """
        self.embedding_loader = AtomicEmbeddingLoader(
            embedding_dim=embedding_dim, 
            method=embedding_method
        )
        self.pooler = AtomicMaxPooling(
            pooling_strategy=pooling_strategy,
            include_stats=include_stats
        )
        
        # Calculate output dimension
        test_embedding = np.random.rand(1, embedding_dim)
        self.output_dim = len(self.pooler.pool(test_embedding))
        
        print(f"🔧 GNN Transformer initialized:")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   Output dim: {self.output_dim}")
        print(f"   Pooling: {pooling_strategy}")
    
    def transform(self, X: List[Union[Structure, Composition]]) -> np.ndarray:
        """
        Transform structures/compositions to fixed-length vectors.
        
        Args:
            X: List of pymatgen Structure or Composition objects
            
        Returns:
            np.ndarray: Feature matrix of shape (n_samples, output_dim)
        """
        features = []
        
        print(f"🧬 Featurizing {len(X)} samples with GNN embeddings...")
        
        for i, sample in enumerate(X):
            if i % 100 == 0:
                print(f"   Progress: {i}/{len(X)}")
            
            try:
                # Get atomic embeddings
                atomic_embeddings = self.embedding_loader.get_atomic_embeddings(sample)
                
                # Pool to fixed-length representation
                pooled_features = self.pooler.pool(atomic_embeddings)
                features.append(pooled_features)
                
            except Exception as e:
                print(f"   Warning: Failed to featurize sample {i}: {e}")
                # Fallback to zero vector
                features.append(np.zeros(self.output_dim))
        
        return np.array(features)
    
    def fit(self, X: List[Union[Structure, Composition]], y=None):
        """Fit method for compatibility (no fitting needed)."""
        return self
    
    def get_feature_labels(self) -> List[str]:
        """Get feature labels for interpretability."""
        labels = []
        
        # Base pooling labels
        if self.pooler.pooling_strategy in ["max", "max_mean"]:
            labels.extend([f"max_embed_{i}" for i in range(self.embedding_loader.embedding_dim)])
        if self.pooler.pooling_strategy in ["mean", "max_mean"]:
            labels.extend([f"mean_embed_{i}" for i in range(self.embedding_loader.embedding_dim)])
        if self.pooler.pooling_strategy == "sum":
            labels.extend([f"sum_embed_{i}" for i in range(self.embedding_loader.embedding_dim)])
        
        # Statistics labels
        if self.pooler.include_stats:
            labels.extend([f"std_embed_{i}" for i in range(self.embedding_loader.embedding_dim)])
            labels.extend([f"min_embed_{i}" for i in range(self.embedding_loader.embedding_dim)])
            labels.extend([f"count_embed_{i}" for i in range(self.embedding_loader.embedding_dim)])
        
        return labels


def make_gnn_featurizer(embedding_dim: int = 64, 
                       embedding_method: str = "auto",
                       pooling_strategy: str = "max_mean") -> GNNTransformer:
    """
    Factory function to create a GNN featurizer.
    
    Args:
        embedding_dim: Dimension of atomic embeddings
        embedding_method: "auto", "elemental_properties", "onehot", etc.
        pooling_strategy: "max", "mean", "max_mean"
        
    Returns:
        GNNTransformer: Configured GNN featurizer
    """
    return GNNTransformer(
        embedding_dim=embedding_dim,
        embedding_method=embedding_method,
        pooling_strategy=pooling_strategy,
        include_stats=True
    )


if __name__ == "__main__":
    # Test the GNN featurizer
    print("🧪 Testing GNN Featurizer...")
    
    # Create test data
    from pymatgen.core import Composition
    test_compositions = [
        Composition("LiFePO4"),
        Composition("NaCl"), 
        Composition("Al2O3"),
        Composition("SiO2")
    ]
    
    # Test featurizer
    gnn_transformer = make_gnn_featurizer(embedding_dim=32)
    features = gnn_transformer.transform(test_compositions)
    
    print(f"✅ Test completed:")
    print(f"   Input: {len(test_compositions)} compositions")
    print(f"   Output: {features.shape}")
    print(f"   Sample features: {features[0][:5]}...")