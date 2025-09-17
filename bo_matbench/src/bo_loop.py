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
        )
        Xfs = torch.tensor(Xfs_np, dtype=torch.float32)

        # 1b) Fit the GP in that subspace
        gp = fit_surrogate(Xfs, data_y)

        # 1c) Make your acquisition function
        # For formation energy, we want to MINIMIZE (find most stable materials)
        acq = make_acquisition(gp, best_f=data_y.min(), acq_fun=params.acq_fun, minimize=True)

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
        dists    = torch.cdist(cand_feat, X_pool[:, feat_idx])
        idx_near = torch.argmin(dists).item()

        # 1f) Fetch its true y and remove it from the pool
        new_Xf = X_pool[idx_near].unsqueeze(0)
        new_y  = y_pool[idx_near].unsqueeze(0)
        X_pool = torch.cat([X_pool[:idx_near], X_pool[idx_near+1:]], dim=0)
        y_pool = torch.cat([y_pool[:idx_near], y_pool[idx_near+1:]], dim=0)

        # 1g) Append to your “seen” data
        data_X = torch.cat([data_X, new_Xf], dim=0)
        data_y = torch.cat([data_y, new_y],  dim=0)

        print(f"Iter {it:02d} — best e_form = {data_y.max():.4f}")

    return data_X, data_y
