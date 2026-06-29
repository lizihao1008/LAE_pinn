"""
training/train.py
Training loop for LAEPINN.

One "epoch" = one pass over all available snapshots (3 redshifts).
For the MVP, batch_size=1 means one snapshot per gradient step.
"""

from __future__ import annotations
import os
import time
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from ..models.pinn import LAEPINN
    from ..training.loss import PINNLoss
except ImportError:
    from models.pinn import LAEPINN
    from training.loss import PINNLoss


def train(
    model: LAEPINN,
    graph_list: list,          # list of PyG Data objects (one per snapshot)
    cfg: dict,
    save_dir: str = "runs/default",
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> dict[str, list]:
    """
    Train LAEPINN on a list of snapshot graphs.

    Returns training history dict.
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    model.train()

    train_cfg = cfg.get("training", {})
    n_epochs   = train_cfg.get("n_epochs", 200)
    lr         = train_cfg.get("lr", 1e-3)
    warmup     = train_cfg.get("warmup_epochs", 10)
    wd         = train_cfg.get("weight_decay", 1e-5)
    log_every  = cfg.get("experiment", {}).get("log_every", 10)

    w = train_cfg.get("loss_weights", {})
    loss_fn = PINNLoss(
        field_mse_w=w.get("field_mse", 1.0),
        power_spec_w=w.get("power_spectrum", 0.1),
        binary_bce_w=w.get("binary_bce", 0.5),
        mcf_w=w.get("mcf_consistency", 0.0),
        global_xhii_w=w.get("global_xHII", 1.0),
        prior_w=w.get("prior", 0.1),
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    history = {"total": [], "field": [], "ps": [], "bce": [], "xhii": [], "prior": []}

    # Print HOD calibration once (it's fixed throughout training)
    if verbose:
        for graph in graph_list:
            hod_cal = getattr(graph, "hod_calibration", None)
            if hod_cal is not None:
                bias_info = "  ".join(
                    f"{hod_cal.bin_labels[b]}:b={hod_cal.hod_params['bias'][b]:.2f}"
                    f"(N={hod_cal.hod_params['N_halos'][b]})"
                    for b in range(len(hod_cal.bin_labels))
                )
                print(f"HOD calibration ({hod_cal.source}): {bias_info}")
                break   # only log once

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        epoch_losses = {k: 0.0 for k in history}

        # Warmup: linear LR increase for first warmup_epochs
        if epoch <= warmup:
            for pg in optimizer.param_groups:
                pg["lr"] = lr * epoch / warmup

        for graph in graph_list:
            graph = graph.to(device)
            optimizer.zero_grad()

            out = model(
                graph,
                hod_basis=graph.hod_basis,
                # xi_global not passed: no binary-search calibration.
                # alpha_nH_scale is trained via the global_xHII loss instead.
                return_intermediates=True,
            )

            x_true = graph.xbox_true.squeeze()  # (G, G, G)

            total, components = loss_fn(
                out, x_true, graph.xi_global, model.unresolved
            )

            # Guard: skip gradient step if loss is NaN/Inf (can happen in
            # early training before parameters settle)
            if torch.isnan(total) or torch.isinf(total):
                print(f"  [epoch {epoch}] NaN/Inf loss — skipping step "
                      f"(components: { {k: f'{v:.3g}' for k,v in components.items()} })")
                optimizer.zero_grad()
                continue

            total.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses["total"] += total.item() / len(graph_list)
            for k, v in components.items():
                if k in epoch_losses:
                    epoch_losses[k] += v / len(graph_list)

        if epoch > warmup:
            scheduler.step()

        for k, v in epoch_losses.items():
            history[k].append(v)

        if verbose and epoch % log_every == 0:
            dt = time.time() - t0
            phys = model.get_learnable_physics()

            # Diagnostics from the last graph in this epoch
            with torch.no_grad():
                last_out    = model(graph, hod_basis=graph.hod_basis,
                                    return_intermediates=True)
                x_pred      = last_out["x_hii_pred"]
                x_pred_mean = float(x_pred.mean().item())
                x_pred_std  = float(x_pred.std().item())   # spatial contrast: 0 = uniform
                j_scale_val = last_out.get("J_scale", float("nan"))
                s_obs_scale = last_out.get("S_obs_scale", float("nan"))
                a_obs_val   = last_out.get("A_obs", float("nan"))
                # J_norm spatial contrast: how much variation the kernel actually sees
                j_total     = last_out.get("J_total")
                j_norm_std  = float((j_total / j_total.mean()).std().item()) if j_total is not None else float("nan")

            xi_last   = float(graph.xi_global)
            n_bins    = model.unresolved.n_bins
            fesc_vals = [phys.get(f"fesc_bin{b}", 0.0) for b in range(n_bins)]

            # For many bins show min/mean/max; for ≤4 bins show all values
            if n_bins <= 4:
                fesc_str = " ".join(f"fesc[{b}]={v:.3f}" for b, v in enumerate(fesc_vals))
            else:
                fmin = min(fesc_vals); fmax = max(fesc_vals)
                fmean = sum(fesc_vals) / n_bins
                fesc_str = (f"fesc min={fmin:.3f} mean={fmean:.3f} max={fmax:.3f} "
                            f"[{' '.join(f'{v:.2f}' for v in fesc_vals)}]")

            print(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"Loss={epoch_losses['total']:.4f} | "
                f"field={epoch_losses['field']:.4f} | "
                f"xHII={epoch_losses['xhii']:.4f} | "
                f"<x>={x_pred_mean:.3f}±{x_pred_std:.3f} ξ={xi_last:.3f} | "
                f"Jσ={j_norm_std:.3f} Sobs={s_obs_scale:.2e} A_obs={a_obs_val:.1f} | "
                f"α={phys.get('alpha_nH_scale', 0):.3f} "
                f"R={phys.get('R_bub', 0):.2f} λ={phys.get('lambda_mfp', 0):.2f} | "
                f"{fesc_str} | {dt:.1f}s"
            )

    # Save
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Model saved to {save_dir}/")
    return history
