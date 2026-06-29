"""
experiments/run_mvp.py
End-to-end MVP validation run.

Usage:
    cd LAE_pinn/
    python -m experiments.run_mvp --sim_root ../simulation --redshift 7.14

Steps:
    1. Load snapshot
    2. Filter to observed LAEs (MUV < -17.5)
    3. Preprocess features + downsample grids
    4. Build k-NN graph
    5. Initialise LAEPINN
    6. Train for N epochs
    7. Evaluate: field metrics + topology statistics
    8. Print learnable physics parameters
"""

from __future__ import annotations
import argparse
import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_snapshot, apply_source_model, compute_feature_stats, prepare_snapshot
from data.graph_builder import build_graph_from_snapshot
from models.pinn import LAEPINN
from training.train import train
from evaluation.topology import compute_all_topology


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sim_root",   default="../simulation")
    p.add_argument("--redshift",   type=float, default=7.14)
    p.add_argument("--muv_cut",    type=float, default=-19,
                   help="MUV cut for observed LAEs. Use -99 for all halos.")
    p.add_argument("--grid",       type=int, default=64)
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--save_dir",   default="runs/mvp")
    p.add_argument("--k_neighbors", type=int, default=16)
    p.add_argument("--r_link",     type=float, default=15.0)
    p.add_argument("--subsample",  type=int, default=None,
                   help="Subsample N halos for speed (None = use all)")
    return p.parse_args()


def field_metrics(x_pred: np.ndarray, x_true: np.ndarray) -> dict:
    """Compute basic field-level evaluation metrics."""
    mse  = float(np.mean((x_pred - x_true) ** 2))
    mae  = float(np.mean(np.abs(x_pred - x_true)))

    # Binary IoU at threshold 0.5
    bp = (x_pred > 0.5)
    bt = (x_true > 0.5)
    iou = float((bp & bt).sum()) / (float((bp | bt).sum()) + 1e-8)

    # Pearson correlation
    xp_flat = x_pred.flatten()
    xt_flat = x_true.flatten()
    corr = float(np.corrcoef(xp_flat, xt_flat)[0, 1])

    return {"mse": mse, "mae": mae, "iou": iou, "pearson": corr}


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  LAE_pinn MVP  |  z={args.redshift}  |  MUV<{args.muv_cut}")
    print(f"{'='*60}\n")

    # ---- 1. Load data ----
    print("Loading simulation snapshot...")
    # Load FULL snapshot first (all halos including faint) for HOD calibration.
    # HOD calibration must see the faint halo population (MUV > -17.5) that we
    # are trying to model — filtering first would make all bins empty.
    snap_full = load_snapshot(args.sim_root, args.redshift)
    # Observed LAEs: filtered catalog used for the GNN graph
    # Filter observed LAEs using the user-specified muv_cut (not hardcoded -17.5)
    if args.muv_cut > -90:
        snap = snap_full.filter_by_muv(args.muv_cut)
    else:
        snap = snap_full  # fiducial: all halos
    print(f"  {snap.n_halos} observed LAEs after MUV cut (full catalog: {snap_full.n_halos})")
    print(f"  xi_global = {snap.xi_global:.3f}  (x_HI ~ {1 - snap.xi_global:.2f})")

    # ---- 2. Preprocess ----
    print("Preprocessing features...")
    from data.preprocessing import compute_feature_stats, prepare_snapshot, build_hod_basis_from_simulation
    # HOD bin edges: from muv_cut to -15 in 0.5 mag steps.
    # Faint (unresolved) halos are those fainter than muv_cut (MUV > muv_cut).
    # Bins end at -15 because the simulation has virtually no halos fainter than that.
    # The last bin in build_hod_basis_from_simulation is always open-ended (MUV > last edge),
    # which captures any residual halos below -15 without needing an explicit upper bound.
    muv_faint_limit = -15.0
    muv_bin_step    = 0.5
    muv_bin_edges   = list(np.arange(args.muv_cut, muv_faint_limit + 1e-6, muv_bin_step))
    n_hod_bins      = len(muv_bin_edges)   # last bin open-ended: MUV > muv_faint_limit
    print(f"  HOD bins: {n_hod_bins} bins from MUV={muv_bin_edges[0]:.1f} "
          f"to MUV={muv_bin_edges[-1]:.1f} (step {muv_bin_step} mag)")

    # HOD calibration on the FULL halo catalog (faint halos must be present)
    hod_calibration = build_hod_basis_from_simulation(
        snap_full,
        muv_det       = args.muv_cut,
        muv_bin_edges = muv_bin_edges,
        grid_size     = args.grid,
    )
    bias_str = "  ".join(f"b{b}={hod_calibration.hod_params['bias'][b]:.2f}"
                         f"(N={hod_calibration.hod_params['N_halos'][b]})"
                         for b in range(len(hod_calibration.bin_labels)))
    print(f"  HOD: {bias_str}")

    stats = compute_feature_stats([snap])
    snap_dict = prepare_snapshot(snap, stats, grid_size=args.grid, device=device,
                                 hod_calibration=hod_calibration)

    # ---- 3. Build graph ----
    print(f"Building k-NN graph (k={args.k_neighbors}, r_max={args.r_link} cMpc/h)...")
    subsample = args.subsample
    graph = build_graph_from_snapshot(snap_dict, k=args.k_neighbors,
                                       r_max=args.r_link, subsample=subsample)
    print(f"  Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")

    # ---- 4. Model ----
    print("Initialising LAEPINN...")
    model = LAEPINN(
        gnn_in_channels=8,
        gnn_hidden_dim=64,
        gnn_out_channels=32,
        gnn_n_layers=3,
        gnn_heads=4,
        kernel_type="mixture",
        kernel_R_init=5.0,
        kernel_delta_init=1.0,
        kernel_lambda_mfp_init=20.0,
        n_hod_bins=n_hod_bins,
        grid_size=args.grid,
        box_size=snap.box_size,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ---- 5. Train ----
    cfg = {
        "training": {
            "n_epochs": args.epochs,
            "lr": args.lr,
            "warmup_epochs": 10,
            "weight_decay": 1e-5,
            "loss_weights": {
                "field_mse": 1.0,
                "power_spectrum": 0.1,
                "binary_bce": 0.5,
                "global_xHII": 1.0,
                "prior": 0.1,
            },
        },
        "experiment": {"log_every": max(1, args.epochs // 10)},
    }
    print(f"\nTraining for {args.epochs} epochs...\n")
    history = train(model, [graph], cfg, save_dir=args.save_dir,
                    device=device, verbose=True)

    # ---- 6. Evaluate ----
    print("\nEvaluating...")
    model.eval()
    graph = graph.to(device)

    with torch.no_grad():
        out = model(graph, hod_basis=graph.hod_basis, return_intermediates=True)

    x_pred_np = out["x_hii_pred"].cpu().numpy()
    x_true_np = graph.xbox_true.squeeze().cpu().numpy()

    # Field metrics
    fm = field_metrics(x_pred_np, x_true_np)
    print(f"\nField metrics:")
    for k, v in fm.items():
        print(f"  {k:10s} = {v:.4f}")

    # Topology
    from data.preprocessing import downsample_grid
    dbox_ds = downsample_grid(snap.dbox_512, args.grid) if snap.dbox_512 is not None else None

    pixel_size = snap.box_size / args.grid
    print(f"\nTopology statistics (threshold=0.5):")
    topo_pred = compute_all_topology(x_pred_np, dbox_ds, snap.box_size)
    topo_true = compute_all_topology(x_true_np, dbox_ds, snap.box_size)

    print(f"  {'Statistic':<30} {'Predicted':>12} {'True':>12}")
    print(f"  {'-'*54}")
    for key in ["percolation_fraction", "n_connected_components", "bsd_median", "bsd_mean"]:
        print(f"  {key:<30} {topo_pred.get(key, 0):>12.3f} {topo_true.get(key, 0):>12.3f}")

    # Learnable physics
    phys = model.get_learnable_physics()
    print(f"\nLearned physical parameters:")
    for k, v in phys.items():
        print(f"  {k:<20} = {v:.4f}")

    print(f"\nDone. Results in {args.save_dir}/")


if __name__ == "__main__":
    main()
