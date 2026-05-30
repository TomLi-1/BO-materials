import random

import numpy as np
import torch
from surrogate import (fit_surrogate, make_acquisition, OptimizerParameters, select_features)
# from acquisition import optimize_acqf
try:
    from botorch.optim import optimize_acqf
except ImportError:
    from botorch.optim.optimize import optimize_acqf


def bayes_optimize(
    X_init: torch.Tensor,
    y_init: torch.Tensor,
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    params: OptimizerParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ensure reproducible behavior across numpy, torch, and Python RNGs
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    # convert y_init to a FloatTensor
    y0 = torch.as_tensor(y_init, dtype=torch.float32)

    # 0) INITIALIZE the “seen” dataset
    data_X = X_init.clone()    # shape (init_budget, n_feats)
    data_y = y0.clone()        # shape (init_budget,)

    # 1) BO MAIN LOOP
    for it in range(params.total_sample_budget - params.initialization_budget):
        # 1a) (Re-)select k sparse features on the CURRENT data_X, data_y
        Xfs_np, feat_idx = select_features(
            data_X.numpy(),
            data_y.numpy(),
            method=params.sparsity_method,
            k=params.num_sparsity_feats,
            random_state=params.seed,
        )

        if params.baseline_feature_idx is not None and params.baseline_feature_idx not in feat_idx:
            # ensure the ALIGNN baseline channel is always available
            feat_idx = np.concatenate([feat_idx, [params.baseline_feature_idx]])
            Xfs_np = data_X.numpy()[:, feat_idx]
        feat_idx_list = feat_idx.tolist()
        Xfs = torch.tensor(Xfs_np, dtype=torch.float32)

        baseline_pos = None
        if params.baseline_feature_idx is not None and params.baseline_feature_idx in feat_idx_list:
            baseline_pos = feat_idx_list.index(params.baseline_feature_idx)

        if params.use_baseline_residual and baseline_pos is not None:
            baseline_vals = Xfs[:, baseline_pos]
            target_y = data_y - baseline_vals
            best_value = target_y.min()
        else:
            target_y = data_y
            best_value = data_y.min()

        # 1b) Fit the GP in that subspace
        gp = fit_surrogate(Xfs, target_y)

        # 1c) Make your acquisition function
        # For formation energy, we want to MINIMIZE (find most stable materials)
        acq = make_acquisition(gp, best_f=best_value, acq_fun=params.acq_fun, minimize=True)

        # 1d) Propose one new point in feature‐space
        bounds = torch.stack([Xfs.min(dim=0).values,
                              Xfs.max(dim=0).values])
        cand_feat, _ = optimize_acqf(
            acq,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,   # draw 20 random starting points for the optimizer
        )
        if cand_feat.dtype != X_pool.dtype:
            cand_feat = cand_feat.to(X_pool.dtype)

        # 1e) Snap back to the nearest structure in your pool
        #      only compare on the selected features
        dists    = torch.cdist(cand_feat, X_pool[:, feat_idx_list])
        idx_near = torch.argmin(dists).item()

        # 1f) Fetch its true y and remove it from the pool
        new_Xf = X_pool[idx_near].unsqueeze(0)
        new_y  = y_pool[idx_near].unsqueeze(0)
        X_pool = torch.cat([X_pool[:idx_near], X_pool[idx_near+1:]], dim=0)
        y_pool = torch.cat([y_pool[:idx_near], y_pool[idx_near+1:]], dim=0)

        # 1g) Append to your “seen” data
        data_X = torch.cat([data_X, new_Xf], dim=0)
        data_y = torch.cat([data_y, new_y],  dim=0)

        print(f"Iter {it:02d} — best e_form = {data_y.min():.4f}")

    return data_X, data_y
