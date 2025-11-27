import os
import pickle

from sklearn.model_selection import train_test_split
from matminer.datasets import load_dataset

CACHE_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir, "data", "matbench_mp_e_form.pkl"
)

def load_formation_energy(test_size=0.2, random_state=42):
    # load and cache the matbench DFT‐formation‐energy dataset
    if os.path.exists(CACHE_PATH):
        df = pickle.load(open(CACHE_PATH, "rb"))
    else:
        df = load_dataset("matbench_mp_e_form")   # structures + e_form
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        pickle.dump(df, open(CACHE_PATH, "wb"))

    # split
    structures = df["structure"].tolist()
    energies   = df["e_form"]  # The formation energy column

    X_train, X_test, y_train, y_test = train_test_split(
        structures, energies,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True)

if __name__ == "__main__":
    Xtr, Xte, ytr, yte = load_formation_energy()
    print(f"Got {len(Xtr)} train / {len(Xte)} test structures.")
