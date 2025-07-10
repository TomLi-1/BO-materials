from matminer.featurizers.composition import ElementProperty
from sklearn.base import TransformerMixin


def make_magpie_featurizer():
    """
    Returns a Matminer ElementProperty featurizer
    using the 'magpie' preset.
    """
    return ElementProperty.from_preset("magpie")


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
        X: list of pymatgen.Structure
        returns: 2D array of shape (len(X), n_features)
        """
        # extract compositions, since ElementProperty operates on Composition
        comps = [s.composition for s in X]
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
