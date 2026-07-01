"""
Load LAE_pinn YAML config and flatten MVP run settings.

Usage:
    from config.load_config import load_config, mvp_settings_from_config
    cfg = load_config()                    # config/default.yaml
    args = mvp_settings_from_config(cfg)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config file. Paths relative to LAE_pinn/ root."""
    p = Path(path) if path is not None else DEFAULT_CONFIG
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p) as f:
        return yaml.safe_load(f)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return str((PROJECT_ROOT / path).resolve())


def mvp_settings_from_config(cfg: dict) -> argparse.Namespace:
    """Build an argparse.Namespace compatible with experiments.run_mvp.main()."""
    data = cfg.get("data", {})
    graph = cfg.get("graph", {})
    train = cfg.get("training", {})
    exp = cfg.get("experiment", {})
    unr = cfg.get("unresolved_sources", {})
    mvp = cfg.get("mvp", {})
    cf = cfg.get("continuous_field", {})
    exc = cfg.get("excursion_set", {})

    redshifts = data.get("redshifts", [7.14])
    redshift = float(mvp.get("redshift", redshifts[0]))

    save_root = exp.get("save_dir", "runs/")
    save_sub = mvp.get("save_subdir", "mvp")
    save_dir = _resolve_path(os.path.join(save_root.rstrip("/"), save_sub))

    field_gen = cf.get("generator", cfg.get("field_generator", "continuous"))

    subsample = mvp.get("subsample")
    if subsample is not None:
        subsample = int(subsample)

    return argparse.Namespace(
        sim_root=_resolve_path(data.get("sim_root", "../simulation")),
        redshift=redshift,
        muv_cut=float(data.get("muv_cut", -19.0)),
        grid=int(data.get("grid_mvp", 64)),
        epochs=int(train.get("n_epochs", 200)),
        lr=float(train.get("lr", 1e-3)),
        device=exp.get("device", "cuda"),
        save_dir=save_dir,
        k_neighbors=int(graph.get("k_neighbors", 16)),
        r_link=float(graph.get("r_link_mpc", 15.0)),
        subsample=subsample,
        unresolved=unr.get("model", mvp.get("unresolved", "linear")),
        excursion=exc.get("type", "equilibrium"),
        field_generator=field_gen,
        dv_max=float(unr.get("dv_max_kms", 1000.0)),
        n_lae_mass_bins=int(unr.get("n_lae_mass_bins", 4)),
        profile_source=unr.get("profile_source", "observed_acf"),
        muv_faint_limit=float(unr.get("muv_faint_limit", -15.0)),
        muv_bin_step=float(unr.get("muv_bin_step", 0.5)),
        seed=int(exp.get("seed", 42)),
        _yaml_cfg=cfg,
    )


def training_cfg_from_yaml(cfg: dict, n_epochs: int | None = None) -> dict:
    """Training loop config dict for training.train.train()."""
    train = cfg.get("training", {})
    exp = cfg.get("experiment", {})
    epochs = n_epochs if n_epochs is not None else int(train.get("n_epochs", 200))
    return {
        "training": {
            "n_epochs": epochs,
            "lr": float(train.get("lr", 1e-3)),
            "warmup_epochs": int(train.get("warmup_epochs", 10)),
            "weight_decay": float(train.get("weight_decay", 1e-5)),
            "loss_weights": dict(train.get("loss_weights", {})),
        },
        "experiment": {
            "log_every": int(exp.get("log_every", max(1, epochs // 10))),
        },
    }
