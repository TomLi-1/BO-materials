from matminer.featurizers.composition import ElementProperty
from sklearn.base import TransformerMixin


def make_magpie_featurizer():
    """
    Returns a Matminer ElementProperty featurizer
    using the 'magpie' preset.
    """
    # Disable multiprocessing to avoid issues with spawn/fork
    featurizer = ElementProperty.from_preset("magpie")
    featurizer.set_n_jobs(1)  # Force single-threaded execution
    return featurizer


class MatminerTransformer(TransformerMixin):
    """
    A simple sklearn‐style wrapper around any matminer featurizer
    that implements .featurize_many(list_of_Structures).
    """

    def __init__(self, featurizer):
        self.feat = featurizer

    def fit(self, X, y=None):
        # no fitting necessary for stateless featurizers
        return self

    def transform(self, X):
        """
        X: list of pymatgen.Structure OR list of pymatgen.Composition
        returns: 2D array of shape (len(X), n_features)
        """
        # Handle both Structure and Composition objects
        from pymatgen.core import Structure, Composition
        
        if len(X) == 0:
            return []
        
        # Check the type of the first element to determine how to extract compositions
        sample = X[0]
        if isinstance(sample, Structure):
            # Extract compositions from Structure objects
            comps = [s.composition for s in X]
        elif isinstance(sample, Composition):
            # Already Composition objects, use directly
            comps = X
        else:
            # Try to handle other types (e.g., if they have a .composition attribute)
            try:
                comps = [s.composition for s in X]
            except AttributeError:
                raise ValueError(f"Unsupported input type: {type(sample)}. Expected Structure or Composition objects.")
        
        # ignore_errors=True will skip any bad entries instead of crashing
        return self.feat.featurize_many(comps, ignore_errors=True)


if __name__ == "__main__":
    # Quick smoke‐test:
    from data_loader import load_formation_energy
    import numpy as np

    X_train, X_test, y_train, y_test = load_formation_energy()

    # Take first 5 structures
    sample_structs = X_train[:5]

    # Build the Magpie featurizer
    magpie = make_magpie_featurizer()
    transformer = MatminerTransformer(magpie)

    # Transform
    feats = transformer.transform(sample_structs)
    feats = np.array(feats)

    print("Feature array shape:", feats.shape)
    print("First row:", feats[0])
