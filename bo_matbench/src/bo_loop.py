import torch
from surrogate import (fit_surrogate, make_acquisition, OptimizerParameters, select_features)
# from acquisition import optimize_acqf
try:
    from botorch.optim import optimize_acqf
except ImportError:
    from botorch.optim.optimize import optimize_acqf


def bayes_optimize(X_init: torch.Tensor, y_init: torch.Tensor, X_pool: torch.Tensor, y_pool: torch.Tensor, params: OptimizerParameters,) -> tuple[torch.Tensor, torch.Tensor]:
    # 1) Feature‐selection
    Xfs, feat_idx = select_features(
        X_init.numpy(), y_init.numpy(),
        method=params.sparsity_method,
        k=params.num_sparsity_feats
    )
    Xfs = torch.tensor(Xfs, dtype=torch.float32)
    y  = torch.tensor(y_init, dtype=torch.float32)

    # 2) Initialize data
    data_X, data_y = Xfs.clone(), y.clone()

    # 3) BO main loop
    for it in range(params.total_sample_budget - params.initialization_budget):
        gp  = fit_surrogate(data_X, data_y)
        acq = make_acquisition(gp, best_f=data_y.max(), acq_fun=params.acq_fun)

        # propose in the *full* feature‐space, then select nearest in pool:
        cand_feat, _ = optimize_acqf(
            acq,
            bounds=torch.stack([
                data_X.min(dim=0).values,
                data_X.max(dim=0).values
            ]),
            q=1, num_restarts=5, raw_samples=20,
        )

        # find nearest neighbor in X_pool (pre‐featurized full‐X)
        distances = torch.cdist(
            cand_feat, X_pool[:, feat_idx], p=2
        ).squeeze(0)
        idx_near  = torch.argmin(distances).item()

        # fetch true (unseen) y value for that structure
        new_Xf = X_pool[idx_near, feat_idx].unsqueeze(0)
        new_y  = y_pool[idx_near].unsqueeze(0)

        data_X = torch.cat([data_X, new_Xf], dim=0)
        data_y = torch.cat([data_y, new_y], dim=0)

        print(f"Iter {it:02d} — best e_form = {data_y.max():.4f}")

    return data_X, data_y
