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


def patch_settings_from_config(cfg: dict) -> argparse.Namespace:
    """Build argparse.Namespace for experiments.run_patches.main()."""
    args = mvp_settings_from_config(cfg)
    patches = cfg.get("patches", {})
    exp = cfg.get("experiment", {})

    if patches.get("patch_dir") is not None:
        args.patch_dir = _resolve_path(patches["patch_dir"])
    else:
        args.patch_dir = None

    save_root = exp.get("save_dir", "runs/")
    save_sub = patches.get("save_subdir", "patches")
    args.save_dir = _resolve_path(os.path.join(save_root.rstrip("/"), save_sub))

    args.max_train = patches.get("max_train")
    args.max_test = patches.get("max_test")
    if args.max_train is not None:
        args.max_train = int(args.max_train)
    if args.max_test is not None:
        args.max_test = int(args.max_test)
    return args


def sync_mvp_args_into_config(cfg: dict, args: argparse.Namespace) -> dict:
    """
    Merge resolved MVP argparse settings back into the YAML dict so
    ``build_pinn_from_config`` sees the same values as ``args`` (including CLI overrides).
    """
    out = dict(cfg)
    out["field_generator"] = args.field_generator
    cf = dict(out.get("continuous_field", {}))
    cf["generator"] = args.field_generator
    out["continuous_field"] = cf

    exc = dict(out.get("excursion_set", {}))
    exc["type"] = args.excursion
    out["excursion_set"] = exc

    unr = dict(out.get("unresolved_sources", {}))
    unr["model"] = args.unresolved
    unr["dv_max_kms"] = args.dv_max
    unr["n_lae_mass_bins"] = args.n_lae_mass_bins
    unr["profile_source"] = args.profile_source
    out["unresolved_sources"] = unr

    data = dict(out.get("data", {}))
    data["grid_mvp"] = args.grid
    data["muv_cut"] = args.muv_cut
    out["data"] = data
    return out


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
        # LOS transmission target (tigm|lya) for the auxiliary loss in train.py.
        "los_transmission": {
            "target": cfg.get("los_transmission", {}).get("target", "tigm"),
        },
    }
