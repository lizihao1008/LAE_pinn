"""
data/preprocessing.py
Grid downsampling, feature normalisation, node-feature assembly, and
HOD calibration for the unresolved faint-source emissivity field.

HOD calibration overview
------------------------
The unresolved source term ε_unres(x) = Σ_b f_esc_b · ε_b(x) requires
pre-computed basis fields ε_b(x) that encode the spatial clustering of
faint halos in each mass / M_UV bin.

Two calibration paths share the same HODCalibration interface:

  Simulation path   — build_hod_basis_from_simulation()
      Uses the full halo catalog (including faint halos below the detection
      threshold).  Measures linear bias directly from the cross-correlation
      of halo number density with the DM density field.  No HMF assumption.

  Observations path — build_hod_basis_from_observations()
      Takes pre-fitted HOD parameters (bias, n_bar) from ACF fitting and
      a density proxy field (e.g. smoothed LAE density or WL convergence).
      This is the interface to use when applying the trained model to real data.

Both functions return a HODCalibration dataclass whose .basis_fields tensor
has shape (n_bins, G, G, G) with unit mean per bin.  The HODUnresolvedField
model in physics/unresolved_sources.py is identical in both cases.
"""

from __future__ import annotations
from dataclasses import dataclass, field as dc_field

import numpy as np
import torch

from .loader import SimSnapshot


# ------------------------------------------------------------------ #
#  Grid operations
# ------------------------------------------------------------------ #

def downsample_grid(grid: np.ndarray, target: int = 64) -> np.ndarray:
    """
    Downsample a cubic grid from its native resolution to (target)^3.

    Uses block averaging (mean of (N/target)^3 sub-voxels) rather than
    interpolation.  Interpolation (e.g. scipy.ndimage.zoom order=1) samples
    a single point per output voxel, which preserves the high-frequency
    variance of the native grid and inflates std(delta) by up to 2.5×
    compared to the true large-scale field.

    For the DM density field this inflated variance suppresses the inferred
    HOD linear bias by the same factor (b = Cov/Var, so larger Var → smaller b).
    Block averaging is the physically correct operation: it gives the true
    mean density (or ionization fraction) per output voxel.

    Requires that the input grid size is divisible by target (enforced by assert).
    """
    src = grid.shape[0]
    assert src % target == 0, (
        f"downsample_grid: input size {src} must be divisible by target {target}"
    )
    factor = src // target
    # Reshape into (target, factor, target, factor, target, factor) and average
    # over the three 'factor' axes → (target, target, target)
    return (grid
            .reshape(target, factor, target, factor, target, factor)
            .mean(axis=(1, 3, 5)))


def grid_to_tensor(grid: np.ndarray, device: str | torch.device = "cpu") -> torch.Tensor:
    """(H,W,D) numpy → (1,1,H,W,D) float32 tensor."""
    return torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)


# ------------------------------------------------------------------ #
#  Node feature assembly
# ------------------------------------------------------------------ #

_FEATURE_STATS: dict[str, tuple[float, float]] = {}  # filled lazily


def _log10_lya(lya_obs: np.ndarray) -> np.ndarray:
    """
    Safe log10 of Lyα luminosity.

    Two issues in real catalogs:
      1. Negative values (numerical noise in the RT output) → clip to 0 first.
      2. Exact zeros (full IGM attenuation at z~7) → add a floor eps so
         log10 is finite everywhere.

    eps is set to 1e-10 × max(L_Lyα), falling back to 1.0 when max == 0.
    The resulting values are in log10 space (float64).
    """
    lya = np.clip(lya_obs, 0.0, None)          # kill negatives before log
    lya_max = lya.max()
    eps = max(float(lya_max) * 1e-10, 1.0)     # relative floor, abs fallback
    return np.log10(lya + eps)


def compute_feature_stats(snaps: list[SimSnapshot]) -> dict[str, tuple[float, float]]:
    """
    Compute global mean and std for each node feature across all snapshots.
    Returns a dict: feature_name → (mean, std).
    """
    features = {
        "pos_x":       [],
        "pos_y":       [],
        "pos_z":       [],
        "muv":         [],
        "log_mass":    [],
        "tigm":        [],
        "ew_obs":      [],
        "lya_obs_log": [],   # log10(L_Lyα)
    }
    for s in snaps:
        features["pos_x"].append(s.pos[:, 0])
        features["pos_y"].append(s.pos[:, 1])
        features["pos_z"].append(s.pos[:, 2])
        features["muv"].append(s.muv)
        features["log_mass"].append(np.log10(np.clip(s.halo_mass, 1e6, None)))
        features["tigm"].append(s.tigm)
        ew = np.clip(s.rew_obs, 0., 300.)
        features["ew_obs"].append(ew)
        features["lya_obs_log"].append(_log10_lya(s.lya_obs))

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
        7  log10(L_Lyα) (standardised)
    """

    def _norm(arr, key):
        mu, sigma = stats[key]
        return (arr - mu) / sigma

    box = snap.box_size
    ew  = np.clip(snap.rew_obs, 0., 300.)

    feats = np.stack([
        snap.pos[:, 0] / box,
        snap.pos[:, 1] / box,
        snap.pos[:, 2] / box,
        _norm(snap.muv, "muv"),
        _norm(np.log10(np.clip(snap.halo_mass, 1e6, None)), "log_mass"),
        _norm(snap.tigm, "tigm"),
        _norm(ew, "ew_obs"),
        _norm(_log10_lya(snap.lya_obs), "lya_obs_log"),
    ], axis=-1).astype(np.float32)  # (N, 8)

    return torch.from_numpy(feats).to(device)


def build_source_weights(
    snap: SimSnapshot,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Source weights w_i for the scatter step.

    Uses log10(L_Lyα) as the weight so the enormous dynamic range of
    luminosities doesn't cause float32 overflow.

    Pipeline:
        log_L  = log10(L_Lyα + eps)         [float64, avoids log(0)]
        w      = log_L − min(log_L)          [shift to ≥ 0]
        w      = w / mean(w)                 [unit mean]
        cast to float32
    """
    log_lya = _log10_lya(snap.lya_obs)
    log_lya = log_lya - log_lya.min()
    norm    = float(log_lya.mean()) + 1e-8
    w       = (log_lya / norm).astype(np.float32)
    return torch.from_numpy(w).to(device)


# ------------------------------------------------------------------ #
#  HOD calibration
# ------------------------------------------------------------------ #

@dataclass
class HODCalibration:
    """
    Pre-computed HOD basis fields for the unresolved faint-source emissivity.

    This is the shared interface between the simulation and observation paths.
    The same HODUnresolvedField model operates identically in both cases;
    only the basis_fields array changes.

    Attributes
    ----------
    basis_fields : (n_bins, G, G, G) float32
        HOD basis fields ε_b(x) = (1 + b_b δ(x)), unit mean per bin.
        Fixed during training — only f_esc_b is learned.
    bin_labels : list[str]
        Human-readable labels for each bin (e.g. "MUV(-17.5,-16.5]").
    hod_params : dict
        Measured / fitted parameters for logging:
            "bias"         : [float, ...] linear bias per bin
            "n_bar"        : [float, ...] mean halo count per voxel
            "mean_log_lum" : [float, ...] mean log10(L_Lyα) per bin
            "N_halos"      : [int, ...]   number of halos in each bin
    source : str
        "simulation" | "observations"
    """
    basis_fields : np.ndarray          # (n_bins, G, G, G)
    bin_labels   : list[str]
    hod_params   : dict
    source       : str = "simulation"


def _scatter_count_to_grid(
    pos     : np.ndarray,   # (N, 3) in cMpc/h
    box_size: float,
    G       : int,
) -> np.ndarray:
    """
    Trilinear scatter of N halos (uniform weight = 1) onto a G³ grid.
    Returns (G, G, G) float64 = halo number density per voxel.

    Vectorised: 8 np.add.at calls, each over N halos.  O(8N) time.
    """
    pos_norm = np.clip(pos / box_size, 0.0, 1.0 - 1e-6)
    pix      = pos_norm * G
    i0       = pix.astype(int)
    i1       = (i0 + 1) % G
    frac     = pix - i0.astype(float)
    f0, f1   = 1.0 - frac, frac

    flat = np.zeros(G * G * G, dtype=np.float64)
    for bx, wx in ((i0[:, 0], f0[:, 0]), (i1[:, 0], f1[:, 0])):
        for by, wy in ((i0[:, 1], f0[:, 1]), (i1[:, 1], f1[:, 1])):
            for bz, wz in ((i0[:, 2], f0[:, 2]), (i1[:, 2], f1[:, 2])):
                idx = (bx % G) * G * G + (by % G) * G + (bz % G)
                np.add.at(flat, idx, wx * wy * wz)
    return flat.reshape(G, G, G)


def build_hod_basis_from_simulation(
    snap         : SimSnapshot,
    muv_det      : float = -17.5,
    muv_bin_edges: list[float] | None = None,
    grid_size    : int = 64,
) -> HODCalibration:
    """
    Calibrate HOD basis fields directly from the simulation's halo catalog.

    Algorithm
    ---------
    For each faint halo mass bin b  (M_UV > muv_det):
      1. Scatter halos to a G³ grid (uniform weight → number density field n_b(x))
      2. Measure linear bias:  b_b = Cov(n_b, δ_dm) / Var(δ_dm)
      3. Build linear bias field:  ε_b(x) = 1 + b_b · δ(x)
      4. Clip to ≥ 0 (no negative emissivity) and normalise to unit mean

    This is model-free: no HMF assumption, no analytical bias model.
    The same ε_b(x) structure — a linear bias field — would be constructed
    from observational HOD fitting, making the model transferable.

    Parameters
    ----------
    snap          : SimSnapshot with full halo catalog (including faint halos)
    muv_det       : M_UV detection threshold; halos with M_UV > muv_det are
                    "unresolved" (not detected). Fainter = more positive M_UV.
    muv_bin_edges : M_UV bin edges for the faint population. Default:
                    [-17.5, -16.5, -15.0] → 3 bins:
                      bin 0 (-17.5, -16.5]: near-threshold, brightest unresolved
                      bin 1 (-16.5, -15.0]: intermediate
                      bin 2 (-15.0, +∞)  : faintest
    grid_size     : output grid resolution G (default 64)

    Returns
    -------
    HODCalibration with basis_fields (n_bins, G, G, G)
    """
    if muv_bin_edges is None:
        muv_bin_edges = [-17.5, -16.5, -15.0]

    G = grid_size

    # ── DM density contrast ──────────────────────────────────────────
    if snap.dbox_512 is not None:
        rho   = downsample_grid(snap.dbox_512, target=G).astype(np.float64)
        rho   = np.clip(rho, 0.0, None) + 1e-4
        delta = rho / rho.mean() - 1.0
    else:
        # No DM density grid available: use the full halo number density as a
        # proxy.  Halos trace DM with some mean bias, so Cov(n_b, δ_halo)/Var(δ_halo)
        # is a valid relative bias estimator even if the absolute normalisation is
        # off.  Much better than zeros (which give b_b = 0 for all bins).
        n_all  = _scatter_count_to_grid(snap.pos, snap.box_size, G)
        n_mean = float(n_all.mean())
        if n_mean > 1e-10:
            delta = (n_all / n_mean - 1.0).astype(np.float64)
        else:
            delta = np.zeros((G, G, G), dtype=np.float64)

    var_delta = float(np.var(delta))

    # ── Faint halos: M_UV > muv_det ──────────────────────────────────
    mask_faint = snap.muv > muv_det
    pos_faint  = snap.pos[mask_faint]           # (N_faint, 3) cMpc/h
    muv_faint  = snap.muv[mask_faint]
    lya_faint  = snap.lya_obs[mask_faint]

    # ── Per-bin processing ────────────────────────────────────────────
    n_bins      = len(muv_bin_edges)            # last bin is open-ended
    basis_list  = []
    bin_labels  = []
    hod_n_bar, hod_bias, hod_mean_lum, hod_n_halos = [], [], [], []

    for b in range(n_bins):
        lo = muv_bin_edges[b]
        hi = muv_bin_edges[b + 1] if b + 1 < n_bins else np.inf

        if np.isinf(hi):
            mask_bin = muv_faint > lo
            label    = f"MUV>{lo:.1f}"
        else:
            mask_bin = (muv_faint > lo) & (muv_faint <= hi)
            label    = f"MUV({lo:.1f},{hi:.1f}]"

        bin_labels.append(label)
        N_bin = int(mask_bin.sum())
        hod_n_halos.append(N_bin)

        if N_bin < 8:
            # Too few halos: fall back to uniform (no spatial variation)
            basis    = np.ones((G, G, G), dtype=np.float32)
            n_bar_b  = 0.0
            bias_b   = 0.0
            mean_lum = 0.0
        else:
            # Number density field (uniform halo weights)
            n_grid = _scatter_count_to_grid(pos_faint[mask_bin], snap.box_size, G)
            n_bar_b = float(n_grid.mean())

            # Mean log10(L_Lyα) for amplitude reference (logging only)
            mean_lum = float(_log10_lya(lya_faint[mask_bin]).mean())

            # Linear bias from cross-correlation:
            #   b_linear = Cov(δ_n, δ_DM) / Var(δ_DM)
            # where δ_n = (n - n̄)/n̄  is the fractional number overdensity.
            #
            # BUG FIX: the previous code computed Cov(n - n̄, δ) / Var(δ),
            # which equals n̄ × b_linear (n̄ times too large in the numerator,
            # giving b values ~10× too small for sparse halo bins).
            if var_delta > 1e-8 and n_bar_b > 1e-8:
                delta_n = (n_grid - n_bar_b) / n_bar_b   # fractional overdensity
                cov_nd  = float(np.mean(delta_n * delta))
                bias_b  = float(np.clip(cov_nd / var_delta, 0.0, 20.0))
            else:
                bias_b = 0.0

            # Linear bias field: 1 + b_b · δ(x)
            basis = (1.0 + bias_b * delta).astype(np.float32)
            basis = np.clip(basis, 0.0, None)   # no negative emissivity

        # Normalise to unit mean so f_esc absorbs the amplitude
        bmean = float(basis.mean())
        if bmean > 1e-10:
            basis = basis / bmean

        basis_list.append(basis)
        hod_n_bar.append(n_bar_b)
        hod_bias.append(bias_b)
        hod_mean_lum.append(mean_lum)

    return HODCalibration(
        basis_fields = np.stack(basis_list, axis=0).astype(np.float32),
        bin_labels   = bin_labels,
        hod_params   = {
            "bias":         hod_bias,
            "n_bar":        hod_n_bar,
            "mean_log_lum": hod_mean_lum,
            "N_halos":      hod_n_halos,
        },
        source = "simulation",
    )


def build_hod_basis_from_observations(
    hod_params_per_bin : list[dict],
    delta_proxy        : np.ndarray,
    bin_labels         : list[str] | None = None,
    grid_size          : int = 64,
) -> HODCalibration:
    """
    Build HOD basis fields from observationally-fitted HOD parameters.

    This is the real-data path, producing the same HODCalibration structure
    as the simulation path so HODUnresolvedField works without modification.

    Parameters
    ----------
    hod_params_per_bin : list of dicts, one per mass bin.
        Required key per dict:
            "bias"  : float — linear galaxy bias, from ACF fitting
                      (or Tinker bias model evaluated at the bin's mean mass)
        Optional keys (used for logging only):
            "n_bar"        : float — mean number density (cMpc/h)^-3
                             from HMF + HOD integral
            "mean_log_lum" : float — mean log10(L_Lyα) from abundance matching
            "label"        : str   — bin name

    delta_proxy : (G, G, G) or any-resolution ndarray
        Density contrast field proxy. Options for real observations:
          - Smoothed LAE density field / b_LAE  (bias-corrected galaxy density)
          - Weak-lensing convergence κ map
          - DM field from N-body (for mock tests)
        Resampled to grid_size if needed.

    bin_labels : optional list of label strings (overrides "label" keys)
    grid_size  : output grid resolution G

    Returns
    -------
    HODCalibration with source="observations"

    Example — applying model to real COSMOS-3D data
    ------------------------------------------------
    From the ACF of detected LAEs fit HOD parameters (M_min, σ, M_1, α).
    Extrapolate HOD to faint end (M_UV > -17.5) to get n_bar and bias per bin.
    Build a density proxy from the LAE density field (smoothed, bias-corrected).

        params = [
            {"bias": 4.2, "n_bar": 3e-4, "mean_log_lum": 40.1, "label": "near_thresh"},
            {"bias": 2.8, "n_bar": 8e-4, "mean_log_lum": 39.5, "label": "intermediate"},
            {"bias": 1.9, "n_bar": 2e-3, "mean_log_lum": 38.8, "label": "faint"},
        ]
        delta_proxy = (lae_density / lae_density.mean() - 1) / b_LAE
        hod = build_hod_basis_from_observations(params, delta_proxy)
    """
    G = grid_size
    n_bins = len(hod_params_per_bin)

    # Resample proxy if needed
    if delta_proxy.shape[0] != G:
        delta_proxy = downsample_grid(delta_proxy, target=G)
    delta_proxy = delta_proxy.astype(np.float64)

    basis_list  = []
    labels_out  = []
    n_bar_list, bias_list, lum_list = [], [], []

    for b, params in enumerate(hod_params_per_bin):
        bias_b   = float(params["bias"])
        n_bar_b  = float(params.get("n_bar", 0.0))
        mean_lum = float(params.get("mean_log_lum", 0.0))
        # Label priority: explicit param "label" > bin_labels arg > default
        label = params.get("label",
                           bin_labels[b] if bin_labels and b < len(bin_labels)
                           else f"bin{b}")

        # Linear bias field: 1 + b_b · δ_proxy(x)
        basis = (1.0 + bias_b * delta_proxy).astype(np.float32)
        basis = np.clip(basis, 0.0, None)

        # Normalise to unit mean
        bmean = float(basis.mean())
        if bmean > 1e-10:
            basis = basis / bmean

        basis_list.append(basis)
        labels_out.append(label)
        n_bar_list.append(n_bar_b)
        bias_list.append(bias_b)
        lum_list.append(mean_lum)

    return HODCalibration(
        basis_fields = np.stack(basis_list, axis=0).astype(np.float32),
        bin_labels   = labels_out,
        hod_params   = {
            "bias":         bias_list,
            "n_bar":        n_bar_list,
            "mean_log_lum": lum_list,
        },
        source = "observations",
    )


# ------------------------------------------------------------------ #
#  Convenience: pack everything for one snapshot
# ------------------------------------------------------------------ #

def prepare_snapshot(
    snap            : SimSnapshot,
    stats           : dict[str, tuple[float, float]],
    grid_size       : int = 64,
    n_populations   : int = 3,          # kept for API compat; used as n_hod_bins
    muv_det         : float = -17.5,
    muv_bin_edges   : list[float] | None = None,
    hod_calibration : HODCalibration | None = None,
    device          : str | torch.device = "cpu",
) -> dict:
    """
    Returns a dict with all tensors needed for one forward pass.

    If hod_calibration is None, it is computed from the simulation snapshot via
    build_hod_basis_from_simulation().  Pass a pre-computed HODCalibration to
    avoid recomputing across training epochs (the basis is fixed and identical
    for all epochs of the same snapshot).

    Dict keys
    ---------
    pos            : (N, 3) normalised positions [0, 1]
    pos_raw        : (N, 3) cMpc/h
    node_feats     : (N, 8) standardised node features
    src_weights    : (N,) log10(L_Lyα) source weights
    xbox_true      : (1, 1, G, G, G) ground-truth ionization field
    hod_basis      : (n_bins, G, G, G) HOD basis fields (fixed, pre-computed)
    hod_calibration: HODCalibration object (for logging / inspection)
    xi_global      : float, global ionized fraction
    z              : float, redshift
    box_size       : float, cMpc/h
    grid_size      : int
    """
    if hod_calibration is None:
        hod_calibration = build_hod_basis_from_simulation(
            snap,
            muv_det       = muv_det,
            muv_bin_edges = muv_bin_edges,
            grid_size     = grid_size,
        )

    xbox_ds    = downsample_grid(snap.xbox_512, target=grid_size)
    hod_basis_t = torch.from_numpy(hod_calibration.basis_fields).to(device)

    def _f32(a):
        return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)

    return {
        "pos":             torch.from_numpy((snap.pos / snap.box_size).astype(np.float32)).to(device),
        "pos_raw":         torch.from_numpy(snap.pos.astype(np.float32)).to(device),
        "node_feats":      build_node_features(snap, stats, device),
        "src_weights":     build_source_weights(snap, device),
        "xbox_true":       grid_to_tensor(xbox_ds, device),
        "hod_basis":       hod_basis_t,           # (n_bins, G, G, G)
        "hod_calibration": hod_calibration,        # for logging
        "xi_global":       snap.xi_global,
        "z":               snap.redshift,
        "box_size":        snap.box_size,
        "grid_size":       grid_size,
        # ── Raw per-LAE Lyα marks (NOT normalised) for the LOS transmission loss.
        #    These are OBSERVATION targets, not GNN inputs: tigm = L_obs / L_int.
        "tigm_obs":        _f32(snap.tigm),        # (N,) LOS transmission in [0,1]
        "lya_int":         _f32(snap.lya_int),     # (N,) intrinsic Lyα luminosity
        "lya_obs":         _f32(snap.lya_obs),     # (N,) observed Lyα luminosity
        "muv":             _f32(snap.muv),         # (N,) UV magnitude
    }


def prepare_patch(
    patch,                               # PatchSnapshot (or any duck-typed catalog + xgrid)
    stats: dict[str, tuple[float, float]],
    grid_size: int | None = None,
    muv_det: float = -17.5,
    muv_bin_edges: list[float] | None = None,
    hod_calibration: HODCalibration | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    """
    Like prepare_snapshot(), but the ionization field is taken directly from
    patch.xgrid (already cropped to the patch volume by make_train_data.py).
    """
    G = grid_size if grid_size is not None else int(patch.xgrid.shape[0])
    xbox = patch.xgrid
    if xbox.shape != (G, G, G):
        if xbox.shape[0] % G == 0:
            xbox = downsample_grid(xbox, target=G)
        else:
            raise ValueError(f"patch xgrid {xbox.shape} incompatible with grid_size={G}")

    if hod_calibration is None:
        hod_calibration = build_hod_basis_from_simulation(
            patch,
            muv_det=muv_det,
            muv_bin_edges=muv_bin_edges,
            grid_size=G,
        )

    hod_basis_t = torch.from_numpy(hod_calibration.basis_fields).to(device)

    return {
        "pos":             torch.from_numpy((patch.pos / patch.box_size).astype(np.float32)).to(device),
        "pos_raw":         torch.from_numpy(patch.pos.astype(np.float32)).to(device),
        "node_feats":      build_node_features(patch, stats, device),
        "src_weights":     build_source_weights(patch, device),
        "xbox_true":       grid_to_tensor(xbox, device),
        "hod_basis":       hod_basis_t,
        "hod_calibration": hod_calibration,
        "xi_global":       patch.xi_global,
        "z":               patch.redshift,
        "box_size":        patch.box_size,
        "grid_size":       G,
    }
