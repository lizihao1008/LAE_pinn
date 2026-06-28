"""
data/preprocessing.py
Grid downsampling, feature normalisation, and node-feature assembly.
"""

from __future__ import annotations
import numpy as np
import torch
from scipy.ndimage import zoom

from .loader import SimSnapshot


# ------------------------------------------------------------------ #
#  Grid operations
# ------------------------------------------------------------------ #

def downsample_grid(grid: np.ndarray, target: int = 64) -> np.ndarray:
    """
    Downsample a cubic grid from its native resolution to (target)^3.
    Uses simple averaging (zoom with order=1) to preserve mean.
    """
    factor = target / grid.shape[0]
    return zoom(grid, factor, order=1)


def grid_to_tensor(grid: np.ndarray, device: str | torch.device = "cpu") -> torch.Tensor:
    """(H,W,D) numpy → (1,1,H,W,D) float32 tensor."""
    return torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)


# ------------------------------------------------------------------ #
#  Node feature assembly
# ------------------------------------------------------------------ #

_FEATURE_STATS: dict[str, tuple[float, float]] = {}  # filled lazily


def compute_feature_stats(snaps: list[SimSnapshot]) -> dict[str, tuple[float, float]]:
    """
    Compute global mean and std for each node feature across all snapshots.
    Returns a dict: feature_name → (mean, std).
    """
    features = {
        "pos_x":        [],
        "pos_y":        [],
        "pos_z":        [],
        "muv":          [],
        "log_mass":     [],
        "tigm":         [],
        "ew_obs":       [],
        "lya_obs_norm": [],
    }
    for s in snaps:
        features["pos_x"].append(s.pos[:, 0])
        features["pos_y"].append(s.pos[:, 1])
        features["pos_z"].append(s.pos[:, 2])
        features["muv"].append(s.muv)
        features["log_mass"].append(np.log10(np.clip(s.halo_mass, 1e6, None)))
        features["tigm"].append(s.tigm)
        # EW_obs: clip negative (unphysical) and log-normalise
        ew = np.clip(s.rew_obs, 0., 300.)
        features["ew_obs"].append(ew)
        # Lya_obs normalised by box mean (per snapshot)
        lya_mean = s.lya_obs.mean() + 1e-40
        features["lya_obs_norm"].append(s.lya_obs / lya_mean)

    stats = {}
    for k, arrs in features.items():
        cat = np.concatenate(arrs)
        stats[k] = (float(cat.mean()), float(cat.std() + 1e-8))
    return stats


def build_node_features(
    snap: SimSnapshot,
    stats: dict[str, tuple[float, float]],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Assemble standardised node feature matrix for all halos in snap.
    Returns: (N, 8) float32 tensor.

    Features (in order):
        0  pos_x / box_size           (0–1)
        1  pos_y / box_size
        2  pos_z / box_size
        3  muv (standardised)
        4  log10(M_h) (standardised)
        5  T_IGM (standardised)
        6  EW_obs (standardised)
        7  Lya_obs / <Lya_obs> (standardised)
    """

    def _norm(arr, key):
        mu, sigma = stats[key]
        return (arr - mu) / sigma

    box = snap.box_size
    lya_mean = snap.lya_obs.mean() + 1e-40
    ew = np.clip(snap.rew_obs, 0., 300.)

    feats = np.stack([
        snap.pos[:, 0] / box,
        snap.pos[:, 1] / box,
        snap.pos[:, 2] / box,
        _norm(snap.muv, "muv"),
        _norm(np.log10(np.clip(snap.halo_mass, 1e6, None)), "log_mass"),
        _norm(snap.tigm, "tigm"),
        _norm(ew, "ew_obs"),
        _norm(snap.lya_obs / lya_mean, "lya_obs_norm"),
    ], axis=-1).astype(np.float32)  # (N, 8)

    return torch.from_numpy(feats).to(device)


def build_source_weights(
    snap: SimSnapshot,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Raw source weights w_i = L_i (observed Lyα, proxy for UV).
    These are multiplied by f_esc (GNN output) inside the model.
    Returns (N,) float32 tensor, normalised to unit mean.
    """
    w = snap.lya_obs.astype(np.float32)
    w = w / (w.mean() + 1e-40)
    return torch.from_numpy(w).to(device)


# ------------------------------------------------------------------ #
#  Density basis fields for unresolved sources
# ------------------------------------------------------------------ #

def build_density_basis(
    snap: SimSnapshot,
    grid_size: int = 64,
    n_populations: int = 3,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Build n_populations density-bias basis fields on a (grid_size)^3 grid.

    For simplicity in the MVP, uses the DM density field raised to powers
    {0.5, 1.0, 2.0}, representing diffuse / linear / clustered populations.

    Returns: (n_populations, grid_size, grid_size, grid_size) float32 tensor.
    """
    if snap.dbox_512 is None:
        # Fall back: uniform density
        unity = torch.ones(n_populations, grid_size, grid_size, grid_size,
                           device=device, dtype=torch.float32)
        return unity / unity.sum(dim=0, keepdim=True).clamp(min=1e-8)

    # Downsample density to working resolution
    rho = downsample_grid(snap.dbox_512, target=grid_size)
    rho = np.clip(rho, 0.0, None) + 1e-4   # ensure positive

    # Power-law bias models
    powers = np.linspace(0.5, 2.0, n_populations)
    basis = np.stack([rho ** p for p in powers], axis=0)  # (P, G, G, G)

    # Normalise each basis field to unit mean
    for b in range(n_populations):
        basis[b] /= basis[b].mean() + 1e-8

    return torch.from_numpy(basis.astype(np.float32)).to(device)


# ------------------------------------------------------------------ #
#  Convenience: pack everything for one snapshot
# ------------------------------------------------------------------ #

def prepare_snapshot(
    snap: SimSnapshot,
    stats: dict[str, tuple[float, float]],
    grid_size: int = 64,
    n_populations: int = 3,
    device: str | torch.device = "cpu",
) -> dict:
    """
    Returns a dict with all tensors needed for one forward pass:
        pos:          (N, 3) normalised to [0, 1]
        node_feats:   (N, 8) standardised features
        src_weights:  (N,) source weights (Lya_obs normalised)
        xbox_true:    (1, 1, G, G, G) downsampled ground-truth ionization
        density_basis:(P, G, G, G) bias basis fields
        xi_global:    float, global ionized fraction
        z:            float, redshift
    """
    xbox_ds = downsample_grid(snap.xbox_512, target=grid_size)

    return {
        "pos":           torch.from_numpy((snap.pos / snap.box_size).astype(np.float32)).to(device),
        "pos_raw":       torch.from_numpy(snap.pos.astype(np.float32)).to(device),  # cMpc/h
        "node_feats":    build_node_features(snap, stats, device),
        "src_weights":   build_source_weights(snap, device),
        "xbox_true":     grid_to_tensor(xbox_ds, device),
        "density_basis": build_density_basis(snap, grid_size, n_populations, device),
        "xi_global":     snap.xi_global,
        "z":             snap.redshift,
        "box_size":      snap.box_size,
        "grid_size":     grid_size,
    }
