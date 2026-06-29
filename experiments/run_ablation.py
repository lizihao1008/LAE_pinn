"""
experiments/run_ablation.py
Ablation runner: sweeps over kernel, source, GNN, and mark ablations.

Usage:
    python -m experiments.run_ablation --sim_root ../simulation --ablation kernel
    python -m experiments.run_ablation --sim_root ../simulation --ablation all
"""

from __future__ import annotations
import argparse
import sys, os, json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_snapshot, apply_source_model, compute_feature_stats, prepare_snapshot
from data.graph_builder import build_graph_from_snapshot
from models.pinn import LAEPINN
from training.train import train
from evaluation.topology import compute_all_topology

# ------------------------------------------------------------------ #
#  Ablation config library
# ------------------------------------------------------------------ #

KERNEL_ABLATIONS = {
    "A1_fixed_gaussian": dict(kernel_type="bubble",   kernel_R_init=5.0,  kernel_delta_init=2.5, learnable_kernel=False),
    "A2_mfp_only":       dict(kernel_type="mfp",      kernel_lambda_mfp_init=20.0),
    "A3_bubble_only":    dict(kernel_type="bubble",    kernel_R_init=5.0,  kernel_delta_init=1.0),
    "A4_mixture":        dict(kernel_type="mixture",   kernel_R_init=5.0,  kernel_delta_init=1.0, kernel_lambda_mfp_init=20.0),
}

SOURCE_ABLATIONS = {
    "B1_no_unresolved":  dict(n_hod_bins=0),
    "B3_constrained":    dict(n_hod_bins=3),
}

GNN_ABLATIONS = {
    "C1_no_gnn":    dict(gnn_n_layers=0),
    "C2_shallow":   dict(gnn_n_layers=1),
    "C4_default":   dict(gnn_n_layers=3),
}

MARK_ABLATIONS = {
    "D1_pos_only":       dict(node_features=["pos_x", "pos_y", "pos_z"]),
    "D2_pos_muv":        dict(node_features=["pos_x", "pos_y", "pos_z", "muv"]),
    "D3_pos_muv_tigm":   dict(node_features=["pos_x", "pos_y", "pos_z", "muv", "tigm"]),
    "D5_all_marks":      dict(node_features=["pos_x", "pos_y", "pos_z", "muv",
                                             "log_mass", "tigm", "ew_obs", "lya_obs_norm"]),
}

ALL_ABLATIONS = {**KERNEL_ABLATIONS, **SOURCE_ABLATIONS, **GNN_ABLATIONS}


def run_one(name: str, extra_cfg: dict, base_graph, snap, device, args) -> dict:
    """Run one ablation variant."""
    g = args.grid
    n_bins    = extra_cfg.pop("n_hod_bins", 3)
    n_layers  = extra_cfg.pop("gnn_n_layers", 3)
    learnable_kernel = extra_cfg.pop("learnable_kernel", True)

    model = LAEPINN(
        gnn_in_channels=8,
        gnn_hidden_dim=32,
        gnn_out_channels=16,
        gnn_n_layers=max(1, n_layers),
        gnn_heads=2,
        n_hod_bins=max(1, n_bins),
        grid_size=g,
        box_size=snap.box_size,
        **{k: v for k, v in extra_cfg.items() if k.startswith("kernel")},
    )

    cfg = {
        "training": {"n_epochs": args.epochs, "lr": 1e-3, "warmup_epochs": 5,
                     "weight_decay": 1e-5,
                     "loss_weights": {"field_mse": 1.0, "power_spectrum": 0.1,
                                      "binary_bce": 0.5, "global_xHII": 1.0, "prior": 0.1}},
        "experiment": {"log_every": 999},
    }
    save_dir = os.path.join(args.save_dir, name)
    train(model, [base_graph], cfg, save_dir=save_dir, device=device, verbose=False)

    model.eval()
    with torch.no_grad():
        out = model(base_graph.to(device), hod_basis=base_graph.hod_basis)

    x_pred = out["x_hii_pred"].cpu().numpy()
    x_true = base_graph.xbox_true.squeeze().cpu().numpy()

    mse  = float(np.mean((x_pred - x_true) ** 2))
    perc = compute_all_topology(x_pred, None, snap.box_size)["percolation_fraction"]
    phys = model.get_learnable_physics()
    return {"name": name, "mse": mse, "percolation": perc, "physics": phys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim_root", default="../simulation")
    p.add_argument("--redshift", type=float, default=7.14)
    p.add_argument("--grid",     type=int,   default=64)
    p.add_argument("--epochs",   type=int,   default=50)
    p.add_argument("--device",   default="cpu")
    p.add_argument("--save_dir", default="runs/ablations")
    p.add_argument("--ablation", default="kernel",
                   choices=["kernel", "source", "gnn", "all"])
    p.add_argument("--subsample", type=int, default=5000)
    args = p.parse_args()

    device = torch.device(args.device)
    snap   = load_snapshot(args.sim_root, args.redshift)
    snap   = apply_source_model(snap, "observed_only")

    from data.preprocessing import compute_feature_stats, prepare_snapshot
    stats    = compute_feature_stats([snap])
    snap_dict = prepare_snapshot(snap, stats, grid_size=args.grid, device=device)
    graph    = build_graph_from_snapshot(snap_dict, subsample=args.subsample)

    ablation_map = {
        "kernel": KERNEL_ABLATIONS,
        "source": SOURCE_ABLATIONS,
        "gnn":    GNN_ABLATIONS,
        "all":    ALL_ABLATIONS,
    }
    ablations = ablation_map[args.ablation]

    results = []
    for name, cfg in ablations.items():
        print(f"  Running {name}...")
        r = run_one(name, dict(cfg), graph, snap, device, args)
        results.append(r)
        print(f"    MSE={r['mse']:.4f}  perc={r['percolation']:.3f}")

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, f"ablation_{args.ablation}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.save_dir}/ablation_{args.ablation}.json")


if __name__ == "__main__":
    main()
