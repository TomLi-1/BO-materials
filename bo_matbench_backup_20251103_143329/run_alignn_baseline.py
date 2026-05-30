#!/usr/bin/env python3
"""
Evaluate pretrained ALIGNN formation-energy predictions on the Matbench split.
Provides a ready-made baseline MAE for comparison with BO pipelines.
"""

import os
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader import load_formation_energy  # noqa: E402
from src.alignn_featurizer import make_alignn_featurizer, print_installation_instructions  # noqa: E402


def evaluate_alignn_baseline():
    print("🔍 Evaluating ALIGNN baseline on Matbench formation energy")
    X_train, X_test, y_train, y_test = load_formation_energy()

    transformer = make_alignn_featurizer()
    if not transformer.is_alignn_active():
        print("\n❌ ALIGNN model unavailable.")
        print_installation_instructions()
        return

    print("✅ ALIGNN model loaded. Computing predictions...")

    train_preds = transformer.predict(list(X_train))
    test_preds = transformer.predict(list(X_test))

    y_train_arr = y_train.to_numpy(dtype=np.float32)
    y_test_arr = y_test.to_numpy(dtype=np.float32)

    train_mae = mean_absolute_error(y_train_arr, train_preds)
    test_mae = mean_absolute_error(y_test_arr, test_preds)

    print("\n📊 ALIGNN Baseline Results")
    print(f"   Train MAE: {train_mae:.4f} eV/atom")
    print(f"   Test MAE : {test_mae:.4f} eV/atom")
    print(f"   Test set size: {len(test_preds)}")


if __name__ == "__main__":
    try:
        evaluate_alignn_baseline()
    except Exception as exc:  # pragma: no cover
        print(f"\n❌ ALIGNN baseline evaluation failed: {exc}")
