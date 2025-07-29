import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import warnings
from botorch.optim.fit import OptimizationWarning

from src.data_loader import load_formation_energy
from src.featurizer import MatminerTransformer, make_magpie_featurizer
from src.surrogate import OptimizerParameters
from src.bo_loop import bayes_optimize
import torch

warnings.filterwarnings("ignore", category=OptimizationWarning)

def main():
    # 1) Load Matbench formation-energy data
    X_train, X_test, y_train, y_test = load_formation_energy()

    # 2) Featurize
    transformer = MatminerTransformer(make_magpie_featurizer())
    X_train_feats = transformer.transform(X_train)
    X_test_feats  = transformer.transform(X_test)
    # to torch Tensors:
    X_train_t = torch.tensor(X_train_feats, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_feats, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test.values, dtype=torch.float32)

    # 3) Combine into a single pool
    X_pool = torch.cat([X_train_t, X_test_t], dim=0)
    y_pool = torch.cat([y_train_t,   y_test_t],   dim=0)

    # 4) BO parameters
    params = OptimizerParameters(
        sparsity_method="MI",
        acq_fun="EI",
        num_sparsity_feats=10,
        initialization_budget=10,
        total_sample_budget=50,
    )

    # 5) Pick an initial random seed set
    init_idx = torch.randperm(len(X_pool))[: params.initialization_budget]
    X_init   = X_pool[init_idx]
    y_init   = y_pool[init_idx]

    # 6) Run BO
    data_X, data_y = bayes_optimize(
        X_init, y_init,
        X_pool, y_pool,
        params
    )

    print(f"\nFOUND!! Best found formation energy = {data_y.max().item():.4f} eV/atom")


if __name__ == "__main__":
    main()