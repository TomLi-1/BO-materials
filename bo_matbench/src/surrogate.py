from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso

@dataclass
class OptimizerParameters:
    # Feature‐selection method: Mutual Information, Lasso, etc.
    sparsity_method: Literal["MI", "LASSO", "NONE"] = "MI"
    # Acquisition function: Expected Improvement, UCB, …
    acq_fun:       Literal["EI", "UCB", "Thompson"] = "EI"
    num_sparsity_feats: int   = 10
    multi_objective:    bool  = False
    constrained:        bool  = False
    total_sample_budget:int  = 50
    initialization_budget:int= 10
    seed:               int  = 42


def select_features(X: np.ndarray,
                    y: np.ndarray,
                    method: str,
                    k: int):
    """
    X: (n_samples, n_features)
    y: (n_samples,)
    method: "MI", "LASSO", or "NONE"
    k: number of features to keep
    returns: X_sel with only k columns
    """
    if method == "MI":
        mi = mutual_info_regression(X, y, random_state=0)
        idx = np.argsort(mi)[-k:]
    elif method == "LASSO":
        model = Lasso(alpha=1e-3, random_state=0).fit(X, y)
        coef = np.abs(model.coef_)
        idx  = np.argsort(coef)[-k:]
    else:  # NONE
        idx = np.arange(X.shape[1])
    return X[:, idx], idx

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound

def fit_surrogate(train_X: torch.Tensor,
                  train_y: torch.Tensor):
    # 1) instantiate
    gp  = SingleTaskGP(train_X, train_y.unsqueeze(-1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    # 2) fit in place
    fit_gpytorch_model(mll)
    # 3) return the trained GP model (not the MLL)
    return gp

def make_acquisition(gp, best_f: torch.Tensor, acq_fun: str):
    if acq_fun == "EI":
        return ExpectedImprovement(model=gp, best_f=best_f)
    elif acq_fun == "UCB":
        return UpperConfidenceBound(model=gp, beta=0.2)
    else:
        raise ValueError(f"Unknown acq_fun {acq_fun}")
