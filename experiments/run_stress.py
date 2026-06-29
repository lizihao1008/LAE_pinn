"""
experiments/run_stress.py
Source degeneracy stress tests (S1–S8 from ROADMAP.md).

Usage:
    python -m experiments.run_stress --sim_root ../simulation --test S1
    python -m experiments.run_stress --sim_root ../simulation --test all
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


STRESS_TESTS = {
    "S1_observed_only": {
        "source_model": "observed_only",
        "train_z": 7.14,
        "test_z":  7.14,
        "description": "Only detected LAEs (MUV < -17.5)",
    },
    "S2_oracle_all_halo": {
        "source_model": "fiducial",
        "train_z": 7.14,
        "test_z":  7.14,
        "description": "All halos in simulation (oracle upper bound)",
    },
    "S4_wrong_massive": {
        "source_model": "massive_only",
        "train_z": 7.14,
        "test_z":  7.14,
        "description": "Massive-only source model (wrong prescription)",
    },
    "S5_wrong_faint": {
        "source_model": "faint_only",
        "train_z": 7.14,
        "test_z":  7.14,
        "description": "Faint-only source model (wrong prescription)",
    },
    "S6_learned_mixture": {
        "source_model": "observed_only",
        "train_z": 7.14,
        "test_z":  7.14,
        "learn_F_b": True,
        "description": "Model infers F_b source-population fractions",
    },
    "S8_wrong_redshift": {
        "source_model": "observed_only",
        "train_z": 7.14,
        "test_z":  6.6,
        "description": "Train at z=7.14, test at z=6.6",
    },
}


def run_stress_test(name: str, cfg_st: dict, args, device: torch.device) -> dict:
    snap = load_snapshot(args.sim_root, cfg_st["train_z"])
    snap = apply_source_model(snap, cfg_st["source_model"])

    from data.preprocessing import compute_feature_stats, prepare_snapshot
    stats = compute_feature_stats([snap])
    snap_dict = prepare_snapshot(snap, stats, grid_size=args.grid, device=device)
    graph = build_graph_from_snapshot(snap_dict, subsample=args.subsample)

    model = LAEPINN(
        gnn_in_channels=8, gnn_hidden_dim=32, gnn_out_channels=16,
        gnn_n_layers=3, gnn_heads=2, n_hod_bins=3,
        grid_size=args.grid, box_size=snap.box_size,
    )

    cfg_train = {
        "training": {
            "n_epochs": args.epochs, "lr": 1e-3, "warmup_epochs": 5,
            "weight_decay": 1e-5,
            "loss_weights": {"field_mse": 1.0, "power_spectrum": 0.1,
                             "binary_bce": 0.5, "global_xHII": 1.0, "prior": 0.1},
        },
        "experiment": {"log_every": 999},
    }
    save_dir = os.path.join(args.save_dir, name)
    train(model, [graph], cfg_train, save_dir=save_dir, device=device, verbose=False)

    # Test (possibly different redshift)
    test_snap = load_snapshot(args.sim_root, cfg_st["test_z"])
    test_snap = apply_source_model(test_snap, "observed_only")  # always test on observed
    test_stats = compute_feature_stats([test_snap])
    test_dict  = prepare_snapshot(test_snap, test_stats, grid_size=args.grid, device=device)
    test_graph = build_graph_from_snapshot(test_dict, subsample=args.subsample)

    model.eval()
    with torch.no_grad():
        out = model(test_graph.to(device), hod_basis=test_graph.hod_basis)

    x_pred = out["x_hii_pred"].cpu().numpy()
    x_true = test_graph.xbox_true.squeeze().cpu().numpy()

    mse  = float(np.mean((x_pred - x_true) ** 2))
    perc = compute_all_topology(x_pred, None, snap.box_size)["percolation_fraction"]
    phys = model.get_learnable_physics()

    return {
        "name": name,
        "description": cfg_st["description"],
        "train_z": cfg_st["train_z"],
        "test_z": cfg_st["test_z"],
        "source_model": cfg_st["source_model"],
        "mse": mse,
        "percolation": perc,
        "F_b": {k: v for k, v in phys.items()
                if k in ["bright", "faint", "diffuse"]},
        "kernel_R": phys.get("R_bub", None),
        "kernel_lambda": phys.get("lambda_mfp", None),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim_root",  default="../simulation")
    p.add_argument("--grid",      type=int,   default=64)
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--device",    default="cpu")
    p.add_argument("--save_dir",  default="runs/stress")
    p.add_argument("--test",      default="S1",
                   help="Which test to run: S1, S2, ... or 'all'")
    p.add_argument("--subsample", type=int, default=5000)
    args = p.parse_args()

    device = torch.device(args.device)

    if args.test == "all":
        tests = STRESS_TESTS
    elif args.test in STRESS_TESTS:
        tests = {args.test: STRESS_TESTS[args.test]}
    else:
        raise ValueError(f"Unknown test '{args.test}'. Options: {list(STRESS_TESTS)}")

    results = []
    for name, cfg_st in tests.items():
        print(f"  [{name}] {cfg_st['description']}")
        r = run_stress_test(name, cfg_st, args, device)
        results.append(r)
        print(f"    MSE={r['mse']:.4f}  perc={r['percolation']:.3f}  F_b={r['F_b']}")

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"stress_{args.test}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
