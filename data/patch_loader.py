"""
data/patch_loader.py
Load overlapping simulation patches produced by simulation/make_train_data.py
and build PyG graphs for multi-patch LAEPINN training.

Each patch folder (train/000042/ or test/001331/) contains:
    xgrid.npy     — (G, G, G) ionization field in the patch
    catalog.npz   — halo properties (pos relative to patch origin)
    meta.json     — origin, box size, redshift, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .graph_builder import build_graph_from_snapshot
from .preprocessing import (
    HODCalibration,
    build_hod_basis_from_simulation,
    compute_feature_stats,
    prepare_patch,
)


@dataclass
class PatchSnapshot:
    """One spatial patch: local xgrid + halo catalog (API-compatible with SimSnapshot)."""

    patch_id: int
    split: str
    redshift: float
    box_size: float                     # patch extent [cMpc/h]
    origin_mpc: tuple[float, float, float]

    pos: np.ndarray
    halo_mass: np.ndarray
    muv: np.ndarray
    lya_int: np.ndarray
    lya_obs: np.ndarray
    rew_int: np.ndarray
    rew_obs: np.ndarray
    pec_vel: np.ndarray
    tigm: np.ndarray
    xgrid: np.ndarray                   # (G, G, G)

    @property
    def n_halos(self) -> int:
        return len(self.halo_mass)

    @property
    def xi_global(self) -> float:
        return float(self.xgrid.mean())

    @property
    def x_hi(self) -> float:
        return 1.0 - self.xi_global

    # Aliases used by prepare_snapshot / build_hod_basis_from_simulation
    @property
    def xbox_512(self) -> np.ndarray:
        return self.xgrid

    @property
    def dbox_512(self) -> None:
        return None


def list_patch_dirs(patch_root: str | Path, split: str = "train") -> list[Path]:
    """Sorted list of patch directories for a split ('train' or 'test')."""
    root = Path(patch_root) / split
    if not root.is_dir():
        raise FileNotFoundError(f"Patch split not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_dir())


def load_patch(patch_dir: str | Path) -> PatchSnapshot:
    """Load one numbered patch folder into a PatchSnapshot."""
    p = Path(patch_dir)
    with open(p / "meta.json") as f:
        meta = json.load(f)

    cat = dict(np.load(p / "catalog.npz"))
    xgrid = np.load(p / "xgrid.npy")

    return PatchSnapshot(
        patch_id=int(meta["id"]),
        split=str(meta["split"]),
        redshift=float(meta["redshift"]),
        box_size=float(meta.get("patch_box_mpc", meta.get("box_size", 40.0))),
        origin_mpc=tuple(meta["origin_mpc"]),
        pos=cat["pos"],
        halo_mass=cat["halo_mass"],
        muv=cat["muv"],
        lya_int=cat["lya_int"],
        lya_obs=cat["lya_obs"],
        rew_int=cat["rew_int"],
        rew_obs=cat["rew_obs"],
        pec_vel=cat["pec_vel"],
        tigm=cat.get("tigm", np.zeros(len(cat["pos"]), dtype=np.float32)),
        xgrid=xgrid,
    )


def load_patches(
    patch_root: str | Path,
    split: str = "train",
    max_patches: int | None = None,
    skip_empty: bool = True,
) -> list[PatchSnapshot]:
    """Load all patches for a split."""
    dirs = list_patch_dirs(patch_root, split)
    if max_patches is not None:
        dirs = dirs[:max_patches]
    patches = []
    for d in dirs:
        patch = load_patch(d)
        if skip_empty and patch.n_halos == 0:
            continue
        patches.append(patch)
    return patches


def load_manifest(patch_root: str | Path) -> dict:
    with open(Path(patch_root) / "manifest.json") as f:
        return json.load(f)


def _hod_for_patch(
    patch: PatchSnapshot,
    muv_bin_edges: list[float],
    muv_det: float,
    hod_calibration: HODCalibration | None,
    *,
    unresolved: str = "linear",
    snap_full=None,
    profile_source: str = "observed_acf",
    dv_max_kms: float = 1000.0,
    n_lae_mass_bins: int = 4,
) -> HODCalibration:
    """Per-patch HOD basis (linear or conditional ACF), computed on the fly."""
    if hod_calibration is not None:
        return hod_calibration
    G = patch.xgrid.shape[0]
    if unresolved == "conditional":
        from .conditional_basis import build_conditional_unresolved_basis
        return build_conditional_unresolved_basis(
            patch,
            muv_bin_edges=muv_bin_edges,
            grid_size=G,
            n_lae_mass_bins=n_lae_mass_bins,
            dv_max_kms=dv_max_kms,
            profile_source=profile_source,
            snap_full=snap_full if snap_full is not None else patch,
            muv_det=muv_det,
        )
    return build_hod_basis_from_simulation(
        patch,
        muv_det=muv_det,
        muv_bin_edges=muv_bin_edges,
        grid_size=G,
    )


def build_graph_from_patch(
    patch: PatchSnapshot,
    stats: dict[str, tuple[float, float]],
    *,
    muv_cut: float | None = None,
    muv_bin_edges: list[float] | None = None,
    muv_det: float = -17.5,
    hod_calibration: HODCalibration | None = None,
    unresolved: str = "linear",
    snap_full=None,
    profile_source: str = "observed_acf",
    dv_max_kms: float = 1000.0,
    n_lae_mass_bins: int = 4,
    k: int = 16,
    r_max: float = 15.0,
    subsample: int | None = None,
    device: str | torch.device = "cpu",
):
    """Build one PyG graph from a patch.  Returns None if no halos remain after cuts."""
    p = patch
    if muv_cut is not None and muv_cut > -90:
        mask = p.muv < muv_cut
        if not np.any(mask):
            return None
        p = _slice_patch(patch, mask)

    G = p.xgrid.shape[0]
    if muv_bin_edges is None:
        muv_bin_edges = [muv_det]

    hod_cal = _hod_for_patch(
        p, muv_bin_edges, muv_det, hod_calibration,
        unresolved=unresolved, snap_full=snap_full,
        profile_source=profile_source, dv_max_kms=dv_max_kms,
        n_lae_mass_bins=n_lae_mass_bins,
    )
    snap_dict = prepare_patch(
        p, stats, grid_size=G, hod_calibration=hod_cal, device=device,
    )
    return build_graph_from_snapshot(
        snap_dict, k=k, r_max=r_max, subsample=subsample,
    )


def build_graph_list_from_patches(
    patch_root: str | Path,
    split: str = "train",
    *,
    stats: dict[str, tuple[float, float]] | None = None,
    muv_cut: float | None = None,
    muv_bin_edges: list[float] | None = None,
    muv_det: float = -17.5,
    hod_calibration: HODCalibration | None = None,
    unresolved: str = "linear",
    snap_full=None,
    profile_source: str = "observed_acf",
    dv_max_kms: float = 1000.0,
    n_lae_mass_bins: int = 4,
    k: int = 16,
    r_max: float = 15.0,
    subsample: int | None = None,
    max_patches: int | None = None,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> tuple[list, dict[str, tuple[float, float]]]:
    """
    Load all patches in `split` and return a list of PyG graphs for train().

    Feature normalisation stats are computed from the loaded patches when
    `stats` is None.
    """
    patches = load_patches(patch_root, split, max_patches=max_patches)
    if not patches:
        raise RuntimeError(f"No non-empty patches in {patch_root}/{split}")

    if stats is None:
        stats = compute_feature_stats(patches)
        if verbose:
            print(f"  Feature stats from {len(patches)} patches")

    graphs = []
    skipped = 0
    for i, patch in enumerate(patches):
        g = build_graph_from_patch(
            patch, stats,
            muv_cut=muv_cut,
            muv_bin_edges=muv_bin_edges,
            muv_det=muv_det,
            hod_calibration=hod_calibration,
            unresolved=unresolved,
            snap_full=snap_full,
            profile_source=profile_source,
            dv_max_kms=dv_max_kms,
            n_lae_mass_bins=n_lae_mass_bins,
            k=k, r_max=r_max, subsample=subsample, device=device,
        )
        if g is None:
            skipped += 1
            continue
        g.patch_id = patch.patch_id
        g.patch_origin = patch.origin_mpc
        graphs.append(g)
        if verbose and (i + 1) % 200 == 0:
            print(f"    built {i + 1}/{len(patches)} graphs ...")

    if verbose:
        print(f"  {split}: {len(graphs)} graphs built ({skipped} skipped)")
    if not graphs:
        raise RuntimeError(f"All patches in {split} were empty after MUV cut")
    return graphs, stats


def _slice_patch(patch: PatchSnapshot, mask: np.ndarray) -> PatchSnapshot:
    """Return a shallow copy with catalog rows masked (xgrid unchanged)."""
    return PatchSnapshot(
        patch_id=patch.patch_id,
        split=patch.split,
        redshift=patch.redshift,
        box_size=patch.box_size,
        origin_mpc=patch.origin_mpc,
        pos=patch.pos[mask],
        halo_mass=patch.halo_mass[mask],
        muv=patch.muv[mask],
        lya_int=patch.lya_int[mask],
        lya_obs=patch.lya_obs[mask],
        rew_int=patch.rew_int[mask],
        rew_obs=patch.rew_obs[mask],
        pec_vel=patch.pec_vel[mask] if patch.pec_vel.ndim == 1 else patch.pec_vel[mask],
        tigm=patch.tigm[mask],
        xgrid=patch.xgrid,
    )
