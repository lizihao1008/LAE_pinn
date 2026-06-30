"""
experiments/run_patches.py
Train LAEPINN on all overlapping patches from simulation/make_train_data.py.

Usage:
    cd LAE_pinn/
    python -m experiments.run_patches \\
        --patch_dir ../simulation/patches_z7.14 \\
        --epochs 50

Each epoch iterates over every training patch (one gradient step per patch).
Evaluation averages field metrics over the held-out test patches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_snapshot
from data.patch_loader import build_graph_list_from_patches, load_manifest
from models.pinn import LAEPINN
from training.train import train


def parse_args():
    p = argparse.ArgumentParser(description="Train LAEPINN on patch dataset")
    p.add_argument("--patch_dir", required=True,
                   help="Root written by make_train_data.py (contains train/, test/)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_dir", default="runs/patches")
    p.add_argument("--muv_cut", type=float, default=-17.5)
    p.add_argument("--k_neighbors", type=int, default=16)
    p.add_argument("--r_link", type=float, default=15.0)
    p.add_argument("--subsample", type=int, default=None,
                   help="Max halos per patch graph (None = all)")
    p.add_argument("--max_train", type=int, default=None,
                   help="Limit number of training patches (debug)")
    p.add_argument("--max_test", type=int, default=None,
                   help="Limit number of test patches for eval (debug)")
    p.add_argument("--excursion", choices=["equilibrium", "bubble"], default="equilibrium")
    p.add_argument("--sim_root", default="../simulation",
                   help="Full snapshot root (LF stats for conditional unresolved)")
    p.add_argument("--unresolved", choices=["linear", "conditional"], default="linear")
    p.add_argument("--profile_source", choices=["observed_acf", "cof", "powerlaw"],
                   default="observed_acf")
    p.add_argument("--dv_max", type=float, default=1000.0)
    p.add_argument("--n_lae_mass_bins", type=int, default=4)
    return p.parse_args()


def field_metrics(x_pred: np.ndarray, x_true: np.ndarray) -> dict:
    mse = float(np.mean((x_pred - x_true) ** 2))
    bp, bt = x_pred > 0.5, x_true > 0.5
    iou = float((bp & bt).sum()) / (float((bp | bt).sum()) + 1e-8)
    corr = float(np.corrcoef(x_pred.ravel(), x_true.ravel())[0, 1])
    return {"mse": mse, "iou": iou, "pearson": corr}


@torch.no_grad()
def evaluate_graph_list(model, graphs, device) -> dict:
    """Average field metrics over a list of patch graphs."""
    model.eval()
    metrics = []
    for g in graphs:
        g = g.to(device)
        out = model(g, hod_basis=g.hod_basis, return_intermediates=False,
                    density_grid=getattr(g, "density_grid", None))
        x_pred = out["x_hii_pred"].cpu().numpy()
        x_true = g.xbox_true.squeeze().cpu().numpy()
        metrics.append(field_metrics(x_pred, x_true))
    keys = metrics[0].keys()
    return {k: float(np.mean([m[k] for m in metrics])) for k in keys}


def main():
    args = parse_args()
    device = torch.device(args.device)

    manifest = load_manifest(args.patch_dir)
    redshift = float(manifest.get("redshift", 7.14))
    patch_grid = int(manifest.get("patch_grid", 64))
    patch_box = float(manifest.get("patch_box_mpc", 40.0))

    muv_faint_limit = -15.0
    muv_bin_edges = list(np.arange(args.muv_cut, muv_faint_limit + 1e-6, 0.5))
    n_hod_bins = len(muv_bin_edges)

    print(f"\n{'='*60}")
    print(f"  LAEPINN patch training  |  z={redshift}")
    print(f"  patch {patch_grid}³ / {patch_box} cMpc/h")
    print(f"  train patches: {manifest.get('n_train', '?')}  "
          f"test: {manifest.get('n_test', '?')}")
    print(f"  unresolved: {args.unresolved}"
          + (f" ({args.profile_source})" if args.unresolved == "conditional" else ""))
    print(f"{'='*60}\n")

    snap_full = None
    if args.unresolved == "conditional":
        print(f"Loading full catalog for LF stats ({args.sim_root}) ...")
        snap_full = load_snapshot(args.sim_root, redshift)
        print(f"  {snap_full.n_halos} halos in full snapshot")

    graph_kw = dict(
        muv_cut=args.muv_cut,
        muv_bin_edges=muv_bin_edges,
        muv_det=args.muv_cut,
        unresolved=args.unresolved,
        snap_full=snap_full,
        profile_source=args.profile_source,
        dv_max_kms=args.dv_max,
        n_lae_mass_bins=args.n_lae_mass_bins,
        k=args.k_neighbors,
        r_max=args.r_link,
        subsample=args.subsample,
        device=device,
    )

    print("Building training graphs ...")
    train_graphs, feat_stats = build_graph_list_from_patches(
        args.patch_dir, "train", max_patches=args.max_train, **graph_kw,
    )

    # xi_global for alpha init: mean over training patch targets
    xi_vals = [float(g.xi_global) for g in train_graphs]
    xi_val = float(np.mean(xi_vals))
    xi_clamped = max(min(xi_val, 0.995), 0.005)
    A_target = xi_clamped ** 2 / (1.0 - xi_clamped)
    alpha_init = 1.0 / max(A_target, 1e-6)
    print(f"  mean patch xi_global = {xi_val:.3f}  →  alpha_init = {alpha_init:.2f}")

    model = LAEPINN(
        gnn_in_channels=8,
        gnn_hidden_dim=64,
        gnn_out_channels=32,
        gnn_n_layers=3,
        gnn_heads=4,
        kernel_type="mixture",
        n_hod_bins=n_hod_bins,
        grid_size=patch_grid,
        box_size=patch_box,
        excursion_type=args.excursion,
        alpha_nH_scale_init=alpha_init,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    cfg = {
        "training": {
            "n_epochs": args.epochs,
            "lr": args.lr,
            "warmup_epochs": min(10, args.epochs // 5),
            "weight_decay": 1e-5,
            "loss_weights": {
                "field_mse": 1.0,
                "power_spectrum": 0.1,
                "binary_bce": 0.5,
                "global_xHII": 10.0,
                "prior": 0.1,
            },
        },
        "experiment": {"log_every": max(1, args.epochs // 10)},
    }

    print(f"\nTraining: {len(train_graphs)} patches × {args.epochs} epochs "
          f"= {len(train_graphs) * args.epochs} steps\n")
    history = train(model, train_graphs, cfg, save_dir=args.save_dir,
                    device=device, verbose=True)

    print("\nEvaluating on test patches ...")
    test_graphs, _ = build_graph_list_from_patches(
        args.patch_dir, "test",
        stats=feat_stats,
        max_patches=args.max_test,
        verbose=True,
        **graph_kw,
    )
    test_metrics = evaluate_graph_list(model, test_graphs, device)
    print("  Test metrics (mean over patches):")
    for k, v in test_metrics.items():
        print(f"    {k:10s} = {v:.4f}")

    summary = {
        "patch_dir": args.patch_dir,
        "n_train_graphs": len(train_graphs),
        "n_test_graphs": len(test_graphs),
        "test_metrics": test_metrics,
        "patch_grid": patch_grid,
        "patch_box_mpc": patch_box,
    }
    with open(os.path.join(args.save_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone.  Results in {args.save_dir}/")


if __name__ == "__main__":
    main()
