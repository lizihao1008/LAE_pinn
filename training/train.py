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

from ..models.pinn import LAEPINN
from ..training.loss import PINNLoss


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
                density_basis=graph.density_basis,
                xi_global=graph.xi_global,
                return_intermediates=True,
            )

            x_true = graph.xbox_true.squeeze()  # (G, G, G)

            total, components = loss_fn(
                out, x_true, graph.xi_global, model.unresolved
            )
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
            print(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"Loss={epoch_losses['total']:.4f} | "
                f"field={epoch_losses['field']:.4f} | "
                f"xHII={epoch_losses['xhii']:.4f} | "
                f"R={phys.get('R_bub', 0):.2f} λ={phys.get('lambda_mfp', 0):.2f} "
                f"F_bright={phys.get('bright', 0):.2f} | {dt:.1f}s"
            )

    # Save
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Model saved to {save_dir}/")
    return history
