"""Checkpoint loading helpers for LAEPINN inference."""

from __future__ import annotations

import numpy as np
import torch

from .pinn import LAEPINN


def infer_hparams_from_state_dict(state_dict: dict) -> dict:
    """Infer LAEPINN constructor kwargs needed to load a checkpoint."""
    n_hod_bins = int(state_dict["unresolved._logit_fesc"].shape[0])
    if "excursion._zeta_raw" in state_dict:
        excursion_type = "bubble"
        th = state_dict["excursion._th_kernels"]
        grid_size = int(th.shape[1])
        n_scales = int(th.shape[0])
    elif "excursion._scale_raw" in state_dict:
        excursion_type = "equilibrium"
        grid_size = None
        n_scales = None
    else:
        excursion_type = "equilibrium"
        grid_size = None
        n_scales = None
    out = {"n_hod_bins": n_hod_bins, "excursion_type": excursion_type}
    if grid_size is not None:
        out["grid_size"] = grid_size
    if n_scales is not None:
        out["bubble_n_scales"] = n_scales
    return out


def muv_bin_edges_from_n_hod(
    n_hod_bins: int,
    muv_faint_limit: float = -15.0,
    step: float = 0.5,
) -> list[float]:
    """Reconstruct HOD bin edges from the number of bins (matches run_mvp / run_patches)."""
    muv_cut = muv_faint_limit - step * (n_hod_bins - 1)
    return list(np.arange(muv_cut, muv_faint_limit + 1e-6, step))


def load_laepinn_checkpoint(
    checkpoint_path: str,
    train_config: dict | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[LAEPINN, dict]:
    """
    Build LAEPINN, load weights, and return merged config for inference.

    ``train_config`` may come from ``train_config.json``; missing keys are
    inferred from the checkpoint (excursion type, n_hod_bins, grid size).
    """
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    inferred = infer_hparams_from_state_dict(state_dict)
    cfg = dict(train_config or {})
    cfg.setdefault("excursion", inferred["excursion_type"])
    cfg.setdefault("n_hod_bins", inferred["n_hod_bins"])
    cfg.setdefault("patch_grid", inferred.get("grid_size", 64))
    cfg.setdefault("patch_box_mpc", 40.0)
    if "muv_bin_edges" not in cfg:
        cfg["muv_bin_edges"] = muv_bin_edges_from_n_hod(int(cfg["n_hod_bins"]))
    if "muv_cut" not in cfg:
        cfg["muv_cut"] = float(cfg["muv_bin_edges"][0])

    model = LAEPINN(
        gnn_in_channels=8,
        gnn_hidden_dim=64,
        gnn_out_channels=32,
        gnn_n_layers=3,
        gnn_heads=4,
        kernel_type="mixture",
        n_hod_bins=int(cfg["n_hod_bins"]),
        grid_size=int(cfg.get("patch_grid", cfg.get("grid_size", 64))),
        box_size=float(cfg.get("patch_box_mpc", cfg.get("box_size", 40.0))),
        excursion_type=cfg.get("excursion", "equilibrium"),
    )
    model.load_state_dict(state_dict)
    model.to(map_location)
    model.eval()
    return model, cfg
