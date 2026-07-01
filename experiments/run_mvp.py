"""
experiments/run_mvp.py
End-to-end MVP validation run.

Usage:
    cd LAE_pinn/
    # All hyperparameters from config/default.yaml:
    python -m experiments.run_mvp --config

    # YAML + optional CLI overrides (e.g. change redshift only):
    python -m experiments.run_mvp --config --redshift 6.6

    # Classic CLI-only (no yaml):
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
from models.pinn import LAEPINN, build_pinn_from_config
from training.train import train
from evaluation.topology import compute_all_topology


def parse_args():
    p = argparse.ArgumentParser(
        description="LAEPINN MVP training run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", nargs="?", const="config/default.yaml", default=None,
        metavar="YAML",
        help="Load hyperparameters from YAML. Omit the path to use config/default.yaml. "
             "Any explicit CLI flags below override the yaml values.",
    )
    p.add_argument("--sim_root",   default=argparse.SUPPRESS)
    p.add_argument("--redshift",   type=float, default=argparse.SUPPRESS)
    p.add_argument("--muv_cut",    type=float, default=argparse.SUPPRESS,
                   help="MUV cut for observed LAEs. Use -99 for all halos.")
    p.add_argument("--grid",       type=int, default=argparse.SUPPRESS)
    p.add_argument("--epochs",     type=int, default=argparse.SUPPRESS)
    p.add_argument("--lr",         type=float, default=argparse.SUPPRESS)
    p.add_argument("--device",     default=argparse.SUPPRESS)
    p.add_argument("--save_dir",   default=argparse.SUPPRESS)
    p.add_argument("--k_neighbors", type=int, default=argparse.SUPPRESS)
    p.add_argument("--r_link",     type=float, default=argparse.SUPPRESS)
    p.add_argument("--subsample",  type=int, default=argparse.SUPPRESS,
                   help="Subsample N halos for speed (None = use all)")
    # --- unresolved-source model + ionization core ---
    p.add_argument("--unresolved", choices=["linear", "conditional"], default=argparse.SUPPRESS,
                   help="linear = (1+b*delta) bias field; "
                        "conditional = COF-ACF stack of faint excess around LAEs (voids -> 0)")
    p.add_argument("--excursion", choices=["equilibrium", "bubble", "bubble_equilibrium"],
                   default=argparse.SUPPRESS,
                   help="equilibrium = smooth x(J); bubble = excursion-set threshold (sharp 0/1); "
                        "bubble_equilibrium = hybrid x_HII = B(x)·x_eq(x) (bubble topology × "
                        "photoionization-equilibrium interior residual)")
    p.add_argument("--field_generator", choices=["continuous", "grid"], default=argparse.SUPPRESS,
                   help="observed-source field generator. continuous (default) = mesh-free "
                        "kernel/top-hat sum over EXACT source positions (removes scatter voxel "
                        "smearing, queryable off-grid); supports BOTH equilibrium and bubble "
                        "cores. grid = legacy scatter+FFT. Continuous bubble costs ~n_scales×(G³·N) "
                        "per step; use --field_generator grid to fall back if needed.")
    p.add_argument("--dv_max", type=float, default=argparse.SUPPRESS,
                   help="LOS redshift-space window [km/s] for conditional profiles")
    p.add_argument("--n_lae_mass_bins", type=int, default=argparse.SUPPRESS,
                   help="number of LAE halo-mass bins for the conditional stack")
    p.add_argument("--profile_source", choices=["observed_acf", "cof", "powerlaw"],
                   default=argparse.SUPPRESS,
                   help="conditional spatial template: observed_acf = measured from the "
                        "MUV-cut LAE auto-correlation (observable, no oracle, no HOD fit); "
                        "cof = COF_tools halo-model CCF (observation transfer); powerlaw = test")
    cli = p.parse_args()

    if cli.config is not None:
        from config.load_config import load_config, mvp_settings_from_config
        args = mvp_settings_from_config(load_config(cli.config))
        for k, v in vars(cli).items():
            if k == "config" or v is None:
                continue
            setattr(args, k, v)
        return args

    # No --config: classic CLI defaults, overridden by any explicit flags
    defaults = dict(
        sim_root="../simulation",
        redshift=7.14,
        muv_cut=-19.0,
        grid=64,
        epochs=100,
        lr=1e-3,
        device="cuda",
        save_dir="runs/mvp",
        k_neighbors=16,
        r_link=15.0,
        subsample=None,
        unresolved="linear",
        excursion="equilibrium",
        field_generator="continuous",
        dv_max=1000.0,
        n_lae_mass_bins=4,
        profile_source="observed_acf",
        muv_faint_limit=-15.0,
        muv_bin_step=0.5,
        seed=None,
        _yaml_cfg=None,
    )
    defaults.update({k: v for k, v in vars(cli).items() if k != "config"})
    return argparse.Namespace(**defaults)


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


def main(args=None):
    if args is None:
        args = parse_args()
    device = torch.device(args.device)

    if getattr(args, "seed", None) is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

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
    muv_faint_limit = getattr(args, "muv_faint_limit", -15.0)
    muv_bin_step    = getattr(args, "muv_bin_step", 0.5)
    muv_bin_edges   = list(np.arange(args.muv_cut, muv_faint_limit + 1e-6, muv_bin_step))
    n_hod_bins      = len(muv_bin_edges)   # last bin open-ended: MUV > muv_faint_limit
    print(f"  HOD bins: {n_hod_bins} bins from MUV={muv_bin_edges[0]:.1f} "
          f"to MUV={muv_bin_edges[-1]:.1f} (step {muv_bin_step} mag)")

    # HOD calibration. Two unresolved-source models:
    #   linear      — global (1+b*delta) bias field (keeps voids bright)
    #   conditional — COF-ACF stack of faint excess around observed LAEs
    #                 (voids -> 0, removes the x_HII floor)
    if args.unresolved == "conditional":
        from data.conditional_basis import build_conditional_unresolved_basis
        print(f"  Unresolved model: conditional ACF stack "
              f"(profile={args.profile_source}, dv_max={args.dv_max:.0f} km/s)")
        hod_calibration = build_conditional_unresolved_basis(
            snap,                                  # observed LAEs (graph nodes)
            muv_bin_edges   = muv_bin_edges,
            grid_size       = args.grid,
            n_lae_mass_bins = args.n_lae_mass_bins,
            dv_max_kms      = args.dv_max,
            profile_source  = args.profile_source,
            snap_full       = snap_full,
            muv_det         = args.muv_cut,
        )
        print(f"  Conditional bins: {len(hod_calibration.bin_labels)}  "
              f"(<L_b> and <N_b|M> measured per bin)")
    else:
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

    # Attach the gas-density grid n_H ∝ (1+δ) for the bubble core (optional;
    # the equilibrium core ignores it).  Downsampled from the native 512³ field.
    if args.excursion in ("bubble", "bubble_equilibrium") and snap.dbox_512 is not None:
        from data.preprocessing import downsample_grid
        dens = downsample_grid(snap.dbox_512, args.grid)         # δ (contrast)
        dens = np.clip(1.0 + dens, 0.0, None)                    # n_H ∝ (1+δ)
        graph.density_grid = torch.from_numpy(dens.astype(np.float32)).to(device)

    # ---- 4. Model ----
    print("Initialising LAEPINN...")
    # Compute alpha_nH_scale_init from xi_global.
    # Exact analytic inverse of x_HII = (sqrt(A^2+4A)-A)/2 evaluated at xi:
    #   A = xi^2 / (1 - xi),   alpha_init = 1 / A.
    # This ensures mean(x_pred) starts near xi_global when J_norm ≈ 1 (uniform),
    # preventing alpha from being 6x too small and x_pred pinned above 0.5 for
    # hundreds of epochs while the global_xHII loss slowly pulls it down.
    xi_val     = float(snap.xi_global)
    xi_clamped = max(min(xi_val, 0.995), 0.005)
    A_target   = xi_clamped ** 2 / (1.0 - xi_clamped)
    alpha_init = 1.0 / max(A_target, 1e-6)
    print(f"  alpha_nH_scale_init = {alpha_init:.2f}  (xi_global={xi_val:.3f} → A_target={A_target:.3f})")

    yaml_cfg = getattr(args, "_yaml_cfg", None)
    if yaml_cfg is not None:
        from config.load_config import sync_mvp_args_into_config
        cfg_for_model = sync_mvp_args_into_config(yaml_cfg, args)
        cfg_for_model.setdefault("data", {})["box_size_mpc"] = snap.box_size
        cfg_for_model.setdefault("unresolved_sources", {})["n_hod_bins"] = n_hod_bins
        cfg_for_model.setdefault("excursion_set", {})["alpha_nH_scale_init"] = alpha_init
        model = build_pinn_from_config(cfg_for_model)
    else:
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
            excursion_type=args.excursion,
            alpha_nH_scale_init=alpha_init,
            field_generator=args.field_generator,
        )
    print(f"  Ionization core: {args.excursion}")
    print(f"  Field generator: {model.field_generator}"
          + ("  (mesh-free off-grid queries; grid training uses scatter+FFT)"
             if model.field_generator == "continuous" else ""))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ---- 5. Train ----
    if yaml_cfg is not None:
        from config.load_config import training_cfg_from_yaml
        cfg = training_cfg_from_yaml(yaml_cfg, n_epochs=args.epochs)
    else:
        cfg = {
            "training": {
                "n_epochs": args.epochs,
                "lr": args.lr,
                "warmup_epochs": 10,
                "weight_decay": 1e-5,
                "loss_weights": {
                    "field_mse":      1.0,
                    "power_spectrum": 0.1,
                    "binary_bce":     0.5,
                    "global_xHII":   10.0,
                    "prior":          0.1,
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
        out = model(graph, hod_basis=graph.hod_basis, return_intermediates=True,
                    density_grid=getattr(graph, "density_grid", None))

    x_pred_np = out["x_hii_pred"].cpu().numpy()
    x_true_np = graph.xbox_true.squeeze().cpu().numpy()

    np.save(os.path.join(args.save_dir, "x_pred.npy"), x_pred_np)
    np.save(os.path.join(args.save_dir, "x_true.npy"), x_true_np)
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
    np.save(os.path.join(args.save_dir, "topo_pred.npy"), topo_pred)
    np.save(os.path.join(args.save_dir, "topo_true.npy"), topo_true)
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
