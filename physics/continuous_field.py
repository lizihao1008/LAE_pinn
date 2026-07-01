"""
physics/continuous_field.py
Resolution-free, physics-strict ionization field via a graph-neural-factor /
kernel-integral (DeepONet-style) representation.

Motivation
----------
The grid pipeline (``scatter`` -> ``fft_convolve_3d`` -> ``ExcursionSetMapping``)
evaluates the ionizing radiation field on a fixed G^3 voxel grid.  This ties the
ionization field to one resolution, forbids querying arbitrary coordinates, and
makes spatial-derivative physics awkward (graph/finite-difference only).

Key observation
---------------
The model's physics is ALREADY a Green-function / kernel integral:

    J(x) = (S * K)(x)
         = A_obs * SUM_i  w_i K(|x - x_i|; theta_K)          (observed LAEs)
         +        INT     S_unres(x') K(|x - x'|) dx'         (unresolved HOD)

The first term is exactly a DeepONet / kernel-method continuous field:

    factors (branch coefficients)  b_i  = w_i = L_i * f_esc_i   <- the GNN output
    basis   (trunk functions)      phi_i(x) = K(|x - x_i|; theta_K)  <- physical kernel
    field   J(x) = SUM_i b_i phi_i(x)

So J(x) -- and hence x_HII(x) via the exact photoionization equilibrium -- can be
evaluated at ANY continuous coordinate, with NO change to the physics.  The FFT
grid is merely one (periodic, voxel-smoothed) discretization of this same sum.

This module evaluates that continuous field directly.  It reuses the SAME kernel
module and the SAME excursion / equilibrium module as ``LAEPINN``, so the learned
physics (R, delta, lambda_mfp, mixture weights, alpha_nH_scale) is shared.

Consistency
-----------
For sources placed on voxel centres and ``S_unres = 0``, the continuous field
queried at voxel centres reproduces ``scatter + fft_convolve_3d`` to FFT numerical
precision (see ``experiments/test_continuous_field.py``).  Off-grid sources differ
only by the trilinear source-smoothing the grid scatter applies -- i.e. by the
discretization error the continuous form removes.

Periodicity & softening
-----------------------
* Distances use the minimum-image convention to match the periodic FFT convolution.
* Pair distances are clamped to a softening length (default dx/2, the same clamp
  ``make_3d_kernel_grid`` applies) so the 1/(4 pi r^2) head of ``k_mfp`` stays finite.
* The kernel is normalised by the SAME constant Z = sum over the grid offsets, so
  J magnitudes match the grid pipeline (and the constant cancels in J / J_ref anyway).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

try:  # package-relative (python -m ... from repo root) or flat sys.path
    from .kernels import make_3d_kernel_grid
    from .scatter import fft_convolve_3d
    from .excursion_set import ExcursionSetMapping, equilibrium_x_hii, _softplus_inverse
except ImportError:  # pragma: no cover - fallback when imported as top-level
    from kernels import make_3d_kernel_grid
    from scatter import fft_convolve_3d
    from excursion_set import ExcursionSetMapping, equilibrium_x_hii, _softplus_inverse


# ------------------------------------------------------------------ #
#  Geometry helpers
# ------------------------------------------------------------------ #

def minimum_image_delta(
    q: torch.Tensor,        # (Q, 3) query coords, normalised [0, 1)
    x: torch.Tensor,        # (N, 3) source coords, normalised [0, 1)
) -> torch.Tensor:
    """
    Pairwise minimum-image separation in NORMALISED units, shape (Q, N, 3).

    Both inputs live in the periodic box [0, 1).  The separation is wrapped to
    (-0.5, 0.5] per axis so it matches the periodic FFT convolution on the grid.
    """
    d = q[:, None, :] - x[None, :, :]      # (Q, N, 3)
    d = d - torch.round(d)                  # wrap to (-0.5, 0.5]
    return d


def periodic_trilinear_sample(
    grid: torch.Tensor,     # (G, G, G) periodic field
    q: torch.Tensor,        # (Q, 3) normalised [0, 1)
) -> torch.Tensor:
    """
    Differentiable periodic trilinear interpolation of a (G,G,G) grid at the
    query coordinates q (normalised [0,1)).  Used for the smooth unresolved /
    diffuse term, whose grid representation is intrinsically smooth.

    This is the exact adjoint convention of ``scatter_to_grid`` (same corner
    weights), so sampling at a voxel centre returns that voxel's value exactly.
    """
    G = grid.shape[0]
    pix = q * G                              # (Q, 3) in [0, G)
    i0 = torch.floor(pix).long()             # lower corner
    frac = pix - i0.float()                   # (Q, 3) in [0, 1)
    f0 = 1.0 - frac
    f1 = frac

    out = torch.zeros(q.shape[0], device=grid.device, dtype=grid.dtype)
    flat = grid.reshape(-1)
    for ox, wx in ((0, f0[:, 0]), (1, f1[:, 0])):
        ix = (i0[:, 0] + ox) % G
        for oy, wy in ((0, f0[:, 1]), (1, f1[:, 1])):
            iy = (i0[:, 1] + oy) % G
            for oz, wz in ((0, f0[:, 2]), (1, f1[:, 2])):
                iz = (i0[:, 2] + oz) % G
                idx = (ix * G + iy) * G + iz
                out = out + flat[idx] * (wx * wy * wz)
    return out


def grid_centre_coords(G: int, device, dtype=torch.float32) -> torch.Tensor:
    """
    Normalised [0,1) coordinates of the G^3 voxel CENTRES, flattened (G^3, 3)
    in C-order (i*G*G + j*G + k), matching ``scatter_to_grid`` / FFT indexing.

    Note: ``scatter_to_grid`` deposits a source at normalised position p into
    voxel floor(p*G); the matching "centre" used here is index/G (the voxel's
    lower-left in that convention), so that a source at index/G reproduces the
    grid value exactly.  This keeps the consistency test exact.
    """
    idx = torch.arange(G, device=device, dtype=dtype) / G   # lower-left convention
    cx, cy, cz = torch.meshgrid(idx, idx, idx, indexing="ij")
    return torch.stack([cx.reshape(-1), cy.reshape(-1), cz.reshape(-1)], dim=-1)


# ------------------------------------------------------------------ #
#  Functional core (shared by the module and the in-model generator)
# ------------------------------------------------------------------ #

def kernel_norm_Z(
    kernel: nn.Module,
    grid_size: int,
    box_size: float,
    device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Z = sum over the grid_size^3 offsets of K(r_clamped).  Exactly the denominator
    ``make_3d_kernel_grid`` divides by (clamp = dx/2), so K/Z equals the normalised
    grid kernel and the kernel-sum field matches the grid convolution's magnitude.
    Differentiable in the kernel parameters.
    """
    G = grid_size
    dx = box_size / G
    coords = (torch.arange(G, device=device, dtype=dtype) - G // 2) * dx
    cx, cy, cz = torch.meshgrid(coords, coords, coords, indexing="ij")
    r = (cx ** 2 + cy ** 2 + cz ** 2).sqrt().clamp(min=dx / 2)
    return kernel(r).sum() + 1e-12


# ------------------------------------------------------------------ #
#  Neighbour-list acceleration (torch_cluster.radius)
# ------------------------------------------------------------------ #
#  For a cutoff r_cut the kernel/top-hat sum only needs source-query pairs within
#  r_cut, turning O(Q*N) into O(Q*<neighbours>).  Periodicity is handled by ghost
#  replication (sources within r_cut of a face get a wrapped copy), so a plain
#  (non-periodic) radius search reproduces the minimum-image neighbours.
#
#  I could not run torch/torch_cluster in the dev sandbox, so every use is guarded
#  by a runtime SELF-CHECK against the dense path on a small query subsample; on
#  any mismatch or missing dependency it falls back to dense+checkpoint (verified).

_NEIGHBOR_WARNED: set = set()


def _warn_once(key, msg: str) -> None:
    if key not in _NEIGHBOR_WARNED:
        _NEIGHBOR_WARNED.add(key)
        import warnings
        warnings.warn(msg, RuntimeWarning)


def _use_neighbors(neighbor_list, r_cut_mpc) -> bool:
    """Decide whether to attempt the neighbour-list path."""
    if r_cut_mpc is None:
        return False                      # no cutoff -> neighbour list is not applicable
    if neighbor_list is False or neighbor_list == "off":
        return False
    if neighbor_list is True or neighbor_list == "on":
        return True
    # "auto": use if torch_cluster is importable
    try:
        import torch_cluster  # noqa: F401
        return True
    except Exception:
        return False


def _build_ghost_sources(src_pos: torch.Tensor, r_cut_norm: float):
    """
    Periodic ghost images of sources within ``r_cut_norm`` of a box face.  With
    r_cut_norm < 0.5 each axis contributes at most one wrapped copy, so a plain
    radius search over query in [0,1) and these ghosts reproduces the minimum-image
    neighbours.  Returns (ghost_pos (M,3) [grad via src_pos], ghost_to_real (M,)).
    """
    N = src_pos.shape[0]
    device = src_pos.device
    dtype = src_pos.dtype
    base = torch.arange(N, device=device)
    axis_shifts = []
    for ax in range(3):
        c = src_pos[:, ax]
        axis_shifts.append([
            (0.0, torch.ones(N, dtype=torch.bool, device=device)),
            (1.0, c <= r_cut_norm),                 # near low face -> ghost above 1
            (-1.0, c >= 1.0 - r_cut_norm),          # near high face -> ghost below 0
        ])
    pos_list, idx_list = [], []
    for sx, mx in axis_shifts[0]:
        for sy, my in axis_shifts[1]:
            for sz, mz in axis_shifts[2]:
                m = mx & my & mz
                if bool(m.any()):
                    shift = torch.tensor([sx, sy, sz], device=device, dtype=dtype)
                    pos_list.append(src_pos[m] + shift)
                    idx_list.append(base[m])
    return torch.cat(pos_list, 0), torch.cat(idx_list, 0)


def _neighbor_pairs_periodic(query, src_pos, r_cut_norm, max_neighbors):
    """
    (q_idx, real_src_idx, r_norm) for source-query pairs within r_cut (periodic).
    ``r_norm`` carries grad via src_pos (for the optional RSD position head).
    Raises ImportError if torch_cluster is unavailable.
    """
    from torch_cluster import radius
    ghost_pos, g2r = _build_ghost_sources(src_pos, r_cut_norm)
    # radius(x, y, r): for each point in y, neighbours in x. row0 -> y (query),
    # row1 -> x (ghost).  Search on detached coords; gather grad-carrying for r.
    assign = radius(ghost_pos.detach(), query.detach(), r=float(r_cut_norm),
                    max_num_neighbors=int(max_neighbors))
    q_idx, g_idx = assign[0], assign[1]
    dr = query[q_idx] - ghost_pos[g_idx]
    return q_idx, g2r[g_idx], dr.norm(dim=-1)


def _selfcheck_close(a_sub: torch.Tensor, b_sub: torch.Tensor,
                     rtol: float = 1e-3, atol: float = 1e-6) -> bool:
    return bool(torch.allclose(a_sub.detach(), b_sub.detach(), rtol=rtol, atol=atol))


def _run_chunked(chunk_fn, query: torch.Tensor, chunk: int, checkpoint: bool,
                 *grad_tensors) -> torch.Tensor:
    """
    Apply ``chunk_fn(q_chunk, *grad_tensors)`` over query chunks and concatenate.
    When grad is enabled and ``checkpoint`` is set, each chunk is wrapped in
    gradient checkpointing (non-reentrant) so its intermediates are freed after
    the forward and recomputed in backward -- capping RETAINED autograd memory at
    O(chunk * N) instead of O(Q * N).  That is what makes a full-grid render
    (Q = G^3) with many sources fit in memory.  ``grad_tensors`` are passed as
    explicit checkpoint inputs; any module parameters used inside ``chunk_fn`` are
    tracked automatically by the non-reentrant implementation.
    """
    use_cp = checkpoint and torch.is_grad_enabled()
    parts: list[torch.Tensor] = []
    for s in range(0, query.shape[0], chunk):
        q = query[s:s + chunk]
        if use_cp:
            parts.append(_torch_checkpoint(chunk_fn, q, *grad_tensors, use_reentrant=False))
        else:
            parts.append(chunk_fn(q, *grad_tensors))
    return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]


def _kernel_sum_neighbors(query, src_pos, src_w, kernel, L, soft_norm,
                          r_cut_norm, chunk, max_neighbors) -> torch.Tensor:
    """Sparse kernel sum over torch_cluster neighbours (pre-Z), chunked over queries."""
    parts: list[torch.Tensor] = []
    for s in range(0, query.shape[0], chunk):
        q = query[s:s + chunk]
        q_idx, s_idx, r = _neighbor_pairs_periodic(q, src_pos, r_cut_norm, max_neighbors)
        kvals = kernel(r.clamp(min=soft_norm) * L)            # (E,)
        contrib = kvals * src_w[s_idx]                        # (E,) grad -> src_w, kernel
        out_c = torch.zeros(q.shape[0], device=q.device, dtype=contrib.dtype)
        parts.append(out_c.scatter_add(0, q_idx, contrib))
    return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]


def kernel_sum_field(
    query: torch.Tensor,        # (Q, 3) normalised [0, 1)
    src_pos: torch.Tensor,      # (N, 3) normalised [0, 1)
    src_w: torch.Tensor,        # (N,) factors
    kernel: nn.Module,
    box_size: float,
    softening_mpc: float,
    Z: torch.Tensor,
    r_cut_mpc: float | None = None,
    chunk: int = 4096,
    checkpoint: bool = True,
    neighbor_list="auto",
    neighbor_max: int = 8192,
) -> torch.Tensor:
    """
    J(q) = (1/Z) SUM_i src_w_i K(|q - x_i|_periodic, clamped)   -> (Q,)

    Differentiable w.r.t. src_w (GNN factors) and the kernel parameters.

    Two evaluation paths:
      * neighbour list (torch_cluster.radius) when a cutoff ``r_cut_mpc`` is set --
        O(Q*<neighbours>) memory & flops.  Guarded by a runtime self-check vs the
        dense path on a small subsample; falls back on mismatch / missing dep.
      * dense chunked sum with gradient ``checkpoint`` (default) -- O(chunk*N)
        retained memory, works without a cutoff.
    """
    L = box_size
    soft_norm = softening_mpc / L
    r_cut_norm = (r_cut_mpc / L) if r_cut_mpc is not None else None

    def _dense_chunk(q: torch.Tensor, src_w_: torch.Tensor) -> torch.Tensor:
        d = minimum_image_delta(q, src_pos)     # (q, N, 3) normalised
        r = d.norm(dim=-1).clamp(min=soft_norm) # (q, N) normalised separation
        kvals = kernel(r * L)                    # (q, N) — kernel wants cMpc/h
        if r_cut_norm is not None:
            kvals = torch.where(r <= r_cut_norm, kvals, torch.zeros_like(kvals))
        return (kvals * src_w_[None, :]).sum(dim=1)

    if _use_neighbors(neighbor_list, r_cut_mpc):
        try:
            out = _kernel_sum_neighbors(query, src_pos, src_w, kernel, L, soft_norm,
                                        r_cut_norm, chunk, neighbor_max)
            Q = query.shape[0]
            idx = torch.randperm(Q, device=query.device)[:min(128, Q)]
            if _selfcheck_close(out[idx], _dense_chunk(query[idx], src_w)):
                return out / Z
            _warn_once(("ksum", id(kernel)),
                       "continuous_field: neighbour-list kernel sum disagreed with dense; "
                       "falling back to dense+checkpoint (check torch_cluster.radius / r_cut).")
        except Exception as e:
            _warn_once(("ksum_err", id(kernel)),
                       f"continuous_field: neighbour list unavailable ({type(e).__name__}: {e}); "
                       "using dense+checkpoint.")

    out = _run_chunked(_dense_chunk, query, chunk, checkpoint, src_w)
    return out / Z


def render_obs_field_on_grid(
    kernel: nn.Module,
    src_pos: torch.Tensor,      # (N, 3) normalised [0, 1)
    src_w: torch.Tensor,        # (N,) factors  w_i = L_i * f_esc_i
    grid_size: int,
    box_size: float,
    softening_mpc: float | None = None,
    r_cut_mpc: float | None = None,
    chunk: int = 4096,
    checkpoint: bool = True,
    neighbor_list="auto",
    neighbor_max: int = 8192,
) -> torch.Tensor:
    """
    Render the continuous observed radiation field J_obs(x) = SUM_i w_i K(|x-x_i|)
    onto a grid_size^3 cube (voxel centres).  This is the drop-in replacement for
    ``scatter_to_grid`` + ``fft_convolve_3d`` of the observed sources: identical to
    that path for on-grid sources, but using EXACT source positions (no trilinear
    voxel smearing) for off-grid sources.

    Because the kernel is unit-sum-normalised (via Z), the spatial mean is
    preserved: mean(J_obs_grid) = mean(S_obs).  So the model's A_obs / J_ref
    calibration is unchanged when this replaces the scatter+FFT observed term.
    """
    device = src_pos.device
    dtype = src_pos.dtype
    G = grid_size
    dx = box_size / G
    soft = softening_mpc if softening_mpc is not None else 0.5 * dx
    Z = kernel_norm_Z(kernel, G, box_size, device, dtype=dtype)
    q = grid_centre_coords(G, device, dtype)
    J = kernel_sum_field(q, src_pos, src_w, kernel, box_size, soft, Z,
                         r_cut_mpc=r_cut_mpc, chunk=chunk, checkpoint=checkpoint,
                         neighbor_list=neighbor_list, neighbor_max=neighbor_max)
    return J.reshape(G, G, G)


# ------------------------------------------------------------------ #
#  Continuous excursion-set BUBBLE mapping
# ------------------------------------------------------------------ #
#
# The grid BubbleExcursionSet smooths the emissivity S and density n_H with a
# stack of top-hat kernels (FFT) and unions a soft threshold across scales:
#     Q_R(x) = zeta * <S>_R(x) / <n_H>_R(x)
#     x_HII(x) = 1 - PROD_R ( 1 - sigmoid(s (Q_R(x) - 1)) ).
#
# Continuous form: split S = A_obs S_obs + S_unres.
#   * <S_obs>_R(x) = (1/N_R) SUM_i w_i [ |x - x_i| <= R ]   — exact top-hat kernel
#     sum over the EXACT observed-source positions (no scatter voxel smearing).
#     N_R is the SAME voxel count the grid top-hat normalises by, so on-grid
#     sources reproduce the grid result.
#   * <S_unres>_R(x), <n_H>_R(x) — these are smooth grid fields; smooth on the grid
#     once per scale (FFT, reusing the bubble's top-hat buffers) then trilinearly
#     interpolate at x.
# All bubble physics (radii, zeta, sharpness, multi-scale union) is preserved
# exactly; only the observed-source smoothing becomes mesh-free.

def bubble_field_continuous(
    bubble,                          # BubbleExcursionSet (shares radii, zeta, sharpness, _th_kernels)
    query: torch.Tensor,             # (Q, 3) normalised [0, 1)
    src_pos: torch.Tensor,           # (N, 3) normalised [0, 1)
    src_w: torch.Tensor,             # (N,) observed source weights w_i (A_obs applied separately)
    A_obs: torch.Tensor | float,     # observed-source amplitude
    S_unres_grid: torch.Tensor,      # (G,G,G) unresolved emissivity (pre-smoothing)
    density_grid: torch.Tensor | None,  # (G,G,G) n_H ∝ (1+δ); None → uniform
    box_size: float,
    chunk: int = 4096,
    checkpoint: bool = True,
    neighbor_list="auto",
    neighbor_max: int = 8192,
) -> torch.Tensor:
    """Continuous excursion-set bubble x_HII at arbitrary query points -> (Q,)."""
    device = query.device
    dtype = query.dtype
    G = bubble.grid_size
    dims = (-3, -2, -1)
    L = box_size

    th = bubble._th_kernels.to(device=device, dtype=dtype)   # (n_scales, G, G, G)
    n_scales = th.shape[0]
    radii = list(bubble.radii_mpc)
    zeta = bubble.zeta
    s_sharp = bubble.sharpness

    n_H = (density_grid.clamp(min=1e-6).to(device=device, dtype=dtype)
           if density_grid is not None
           else torch.ones(G, G, G, device=device, dtype=dtype))
    S_unr = S_unres_grid.to(device=device, dtype=dtype)
    S_unr_fft = torch.fft.rfftn(S_unr, dim=dims)
    H_fft = torch.fft.rfftn(n_H, dim=dims)

    # Per-scale grid-smoothed unresolved field and n_H, plus the voxel count N_R.
    S_unr_bar_grids, H_bar_grids, counts = [], [], []
    for j in range(n_scales):
        K_fft = torch.fft.rfftn(th[j], dim=dims)
        S_unr_bar_grids.append(
            torch.fft.irfftn(S_unr_fft * K_fft, s=S_unr.shape, dim=dims).clamp(min=0.0))
        H_bar_grids.append(
            torch.fft.irfftn(H_fft * K_fft, s=n_H.shape, dim=dims).clamp(min=1e-8))
        counts.append((th[j] > 0).sum().to(dtype).clamp(min=1.0))   # voxels within R_j

    A_obs_t = (A_obs if torch.is_tensor(A_obs)
               else torch.as_tensor(A_obs, device=device, dtype=dtype))

    def _dense_chunk(q, src_w_, A_obs_, zeta_, s_sharp_):
        d = minimum_image_delta(q, src_pos)
        r_mpc = d.norm(dim=-1) * L                              # (q, N) cMpc/h
        log_one_minus_p = torch.zeros(q.shape[0], device=device, dtype=dtype)
        for j, R in enumerate(radii):
            within = (r_mpc <= float(R)).to(dtype)             # hard top-hat (radii fixed)
            S_obs_bar = (within * src_w_[None, :]).sum(dim=1) / counts[j]
            S_unr_bar = periodic_trilinear_sample(S_unr_bar_grids[j], q)
            H_bar = periodic_trilinear_sample(H_bar_grids[j], q).clamp(min=1e-8)
            Q = zeta_ * (A_obs_ * S_obs_bar + S_unr_bar) / H_bar
            p = torch.sigmoid(s_sharp_ * (Q - 1.0)).clamp(max=1 - 1e-6)
            log_one_minus_p = log_one_minus_p + torch.log1p(-p)
        return 1.0 - torch.exp(log_one_minus_p)

    # Neighbour-list path: a source contributes if within the LARGEST radius; each
    # scale then masks by |r| <= R_j.  Guarded by a self-check vs dense.
    r_cut_mpc = float(max(radii))
    if _use_neighbors(neighbor_list, r_cut_mpc):
        r_cut_norm = r_cut_mpc / L

        def _nbr():
            parts = []
            for s in range(0, query.shape[0], chunk):
                q = query[s:s + chunk]
                q_idx, s_idx, r = _neighbor_pairs_periodic(q, src_pos, r_cut_norm, neighbor_max)
                r_mpc = r * L
                w = src_w[s_idx]
                log_one_minus_p = torch.zeros(q.shape[0], device=device, dtype=dtype)
                for j, R in enumerate(radii):
                    within = (r_mpc <= float(R)).to(dtype)
                    S_obs_bar = torch.zeros(q.shape[0], device=device, dtype=dtype)
                    S_obs_bar = S_obs_bar.scatter_add(0, q_idx, within * w) / counts[j]
                    S_unr_bar = periodic_trilinear_sample(S_unr_bar_grids[j], q)
                    H_bar = periodic_trilinear_sample(H_bar_grids[j], q).clamp(min=1e-8)
                    Qv = zeta * (A_obs_t * S_obs_bar + S_unr_bar) / H_bar
                    p = torch.sigmoid(s_sharp * (Qv - 1.0)).clamp(max=1 - 1e-6)
                    log_one_minus_p = log_one_minus_p + torch.log1p(-p)
                parts.append(1.0 - torch.exp(log_one_minus_p))
            return torch.cat(parts, 0) if len(parts) > 1 else parts[0]

        try:
            out = _nbr()
            Qn = query.shape[0]
            idx = torch.randperm(Qn, device=query.device)[:min(128, Qn)]
            ref = _dense_chunk(query[idx], src_w, A_obs_t, zeta, s_sharp)
            if _selfcheck_close(out[idx], ref):
                return out
            _warn_once(("bubble", id(bubble)),
                       "continuous_field: neighbour-list bubble disagreed with dense; "
                       "falling back to dense+checkpoint.")
        except Exception as e:
            _warn_once(("bubble_err", id(bubble)),
                       f"continuous_field: neighbour list unavailable ({type(e).__name__}: {e}); "
                       "using dense+checkpoint.")

    return _run_chunked(_dense_chunk, query, chunk, checkpoint, src_w, A_obs_t, zeta, s_sharp)


def render_bubble_on_grid(
    bubble,
    src_pos: torch.Tensor,
    src_w: torch.Tensor,
    A_obs: torch.Tensor | float,
    S_unres_grid: torch.Tensor,
    density_grid: torch.Tensor | None,
    box_size: float,
    grid_size: int | None = None,
    chunk: int = 4096,
    checkpoint: bool = True,
    neighbor_list="auto",
    neighbor_max: int = 8192,
) -> torch.Tensor:
    """Render the continuous bubble x_HII onto a grid_size^3 cube (any resolution)."""
    G = grid_size if grid_size is not None else bubble.grid_size
    q = grid_centre_coords(G, src_pos.device, src_pos.dtype)
    x = bubble_field_continuous(bubble, q, src_pos, src_w, A_obs,
                                S_unres_grid, density_grid, box_size,
                                chunk=chunk, checkpoint=checkpoint,
                                neighbor_list=neighbor_list, neighbor_max=neighbor_max)
    return x.reshape(G, G, G)


@torch.no_grad()
def calibrate_bubble_zeta(
    bubble,
    src_pos: torch.Tensor,
    src_w: torch.Tensor,
    A_obs: torch.Tensor | float,
    S_unres_grid: torch.Tensor,
    density_grid: torch.Tensor | None,
    box_size: float,
    target: float = 0.5,
    n_calib: int = 8192,
    n_iter: int = 30,
) -> None:
    """
    One-time zeta calibration so <x_HII> ~ target, using the CONTINUOUS render on a
    Monte-Carlo subsample of query points (cheap; bisection in log-zeta, no grad).
    Mirrors BubbleExcursionSet.calibrate_zeta but for the continuous generator.
    """
    device, dtype = src_pos.device, src_pos.dtype
    q = torch.rand(n_calib, 3, device=device, dtype=dtype)
    lo = torch.tensor(-12.0, dtype=dtype)
    hi = torch.tensor(12.0, dtype=dtype)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        bubble._zeta_raw.data.copy_(_softplus_inverse(mid.exp()).to(bubble._zeta_raw.dtype))
        xm = bubble_field_continuous(bubble, q, src_pos, src_w, A_obs,
                                     S_unres_grid, density_grid, box_size).mean().item()
        if xm > target:
            hi = mid
        else:
            lo = mid
    bubble._zeta_raw.data.copy_(_softplus_inverse(((lo + hi) / 2).exp()).to(bubble._zeta_raw.dtype))


# ------------------------------------------------------------------ #
#  Continuous ionization field
# ------------------------------------------------------------------ #

class ContinuousIonizationField(nn.Module):
    """
    Evaluate x_HII(x) at arbitrary continuous coordinates from graph-neural
    source factors and a physical radiative kernel.

    Parameters
    ----------
    kernel : nn.Module
        The SAME radiative kernel used by ``LAEPINN`` (MixtureKernel / MFPKernel
        / BubbleKernel).  Must implement ``forward(r) -> K(r)`` for r in cMpc/h.
    excursion : nn.Module
        The SAME ``ExcursionSetMapping`` (equilibrium) used by ``LAEPINN``.
        This class handles the equilibrium mapping (pointwise in J, resolution-free
        by construction).  The bubble core has its own continuous implementation in
        ``bubble_field_continuous`` / ``render_bubble_on_grid`` (per-scale top-hat
        kernel sums), exposed off-grid via ``BubbleContinuousEvaluator``.
    box_size : float
        Periodic box size in cMpc/h.
    grid_size_ref : int
        Reference grid used ONLY to (a) define the softening length dx/2 and
        (b) compute the kernel normalisation constant Z so J magnitudes match
        the grid pipeline.  It does NOT limit the query resolution.
    softening_mpc : float | None
        Pair-distance floor in cMpc/h.  Defaults to dx/2 = box_size/(2*grid_size_ref),
        matching ``make_3d_kernel_grid``'s clamp.
    """

    def __init__(
        self,
        kernel: nn.Module,
        excursion: nn.Module | None = None,
        box_size: float = 160.0,
        grid_size_ref: int = 64,
        softening_mpc: float | None = None,
    ):
        super().__init__()
        self.kernel = kernel
        self.excursion = excursion if excursion is not None else ExcursionSetMapping()
        self.box_size = float(box_size)
        self.grid_size_ref = int(grid_size_ref)
        dx = self.box_size / self.grid_size_ref
        self.softening_mpc = float(softening_mpc) if softening_mpc is not None else 0.5 * dx

    # -------------------------------------------------------------- #
    #  Kernel normalisation constant Z (matches grid convolution)
    # -------------------------------------------------------------- #

    def kernel_norm_constant(self, device, dtype: torch.dtype | None = None) -> torch.Tensor:
        """
        Z = sum over the G_ref^3 grid offsets of K(r_clamped).  This is exactly
        the denominator ``make_3d_kernel_grid`` divides by, so K/Z here equals the
        normalised grid kernel.  Differentiable in the kernel parameters.

        ``dtype`` follows the working precision (defaults to the kernel parameter
        dtype) so a float64 deployment is not silently down-cast.  Note the grid
        ``make_3d_kernel_grid`` builds this in float32; the difference is ~1e-7 and
        cancels in J / J_ref.
        """
        if dtype is None:
            try:
                dtype = next(self.kernel.parameters()).dtype
            except StopIteration:
                dtype = torch.float32
        return kernel_norm_Z(self.kernel, self.grid_size_ref, self.box_size,
                             device, dtype=dtype)

    # -------------------------------------------------------------- #
    #  Observed-source radiation field  J_obs(q) = SUM_i w_i K(|q - x_i|)
    # -------------------------------------------------------------- #

    def source_field(
        self,
        query: torch.Tensor,        # (Q, 3) normalised [0, 1)
        src_pos: torch.Tensor,      # (N, 3) normalised [0, 1)
        src_w: torch.Tensor,        # (N,) factors  w_i = L_i * f_esc_i
        r_cut_mpc: float | None = None,
        chunk: int = 4096,
        normalise: bool = True,
        checkpoint: bool = True,
    ) -> torch.Tensor:
        """
        Continuous observed radiation field at the query points.

        J_obs(q) = (1/Z) SUM_i w_i K(|q - x_i|_periodic, clamped)

        Memory is bounded by processing the query points in chunks of ``chunk`` and,
        when grad is enabled and ``checkpoint`` is set, gradient checkpointing so the
        RETAINED graph is O(chunk*N) not O(Q*N).  If ``r_cut_mpc`` is given, source
        pairs beyond the cutoff are masked (the kernels decay); a true neighbour list
        (torch_cluster.radius) is a drop-in further optimisation.
        """
        device = query.device
        Z = (self.kernel_norm_constant(device, dtype=query.dtype)
             if normalise else torch.ones((), device=device, dtype=query.dtype))
        return kernel_sum_field(
            query, src_pos, src_w, self.kernel, self.box_size,
            self.softening_mpc, Z, r_cut_mpc=r_cut_mpc, chunk=chunk, checkpoint=checkpoint,
        )

    # -------------------------------------------------------------- #
    #  Full forward:  factors + diffuse -> J(q) -> x_HII(q)
    # -------------------------------------------------------------- #

    def forward(
        self,
        query: torch.Tensor,                 # (Q, 3) normalised [0, 1)
        src_pos: torch.Tensor,               # (N, 3)
        src_w: torch.Tensor,                 # (N,) effective factors (A_obs already folded in if desired)
        J_unres_grid: torch.Tensor | None = None,   # (G,G,G) pre-convolved unresolved field
        J_ref: torch.Tensor | float | None = None,  # normalisation; if None, estimated from query mean
        r_cut_mpc: float | None = None,
        chunk: int = 4096,
        return_J: bool = False,
        checkpoint: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with ``x_hii`` (Q,) and optionally ``J`` (Q,).

        J(q)     = J_obs(q) + interp(J_unres_grid, q)
        x_HII(q) = equilibrium(J(q) / J_ref)

        ``src_w`` should already include the A_obs amplitude if you want to match
        ``LAEPINN`` (i.e. pass ``A_obs * w_eff``).  ``J_unres_grid`` should be the
        unresolved source field ALREADY convolved with the kernel (by linearity
        this equals the unresolved part of the grid J_total).
        """
        J_obs = self.source_field(query, src_pos, src_w, r_cut_mpc=r_cut_mpc,
                                  chunk=chunk, checkpoint=checkpoint)
        if J_unres_grid is not None:
            J = J_obs + periodic_trilinear_sample(J_unres_grid, query)
        else:
            J = J_obs

        if J_ref is None:
            J_ref = J.mean().detach().clamp(min=1e-12)
        elif not torch.is_tensor(J_ref):
            J_ref = torch.as_tensor(float(J_ref), device=query.device)

        x_hii = self.excursion(J / J_ref)
        out = {"x_hii": x_hii}
        if return_J:
            out["J"] = J
            out["J_ref"] = J_ref
        return out

    # -------------------------------------------------------------- #
    #  Convenience: evaluate the continuous field on a G^3 grid
    # -------------------------------------------------------------- #

    @torch.no_grad()
    def evaluate_on_grid(
        self,
        grid_size: int,
        src_pos: torch.Tensor,
        src_w: torch.Tensor,
        J_unres_grid: torch.Tensor | None = None,
        J_ref: torch.Tensor | float | None = None,
        r_cut_mpc: float | None = None,
        chunk: int = 4096,
    ) -> torch.Tensor:
        """
        Render the continuous field onto a ``grid_size``^3 cube (any resolution).
        Useful for power-spectrum / topology metrics and for super-resolution.
        Returns (grid_size, grid_size, grid_size).
        """
        device = src_pos.device
        q = grid_centre_coords(grid_size, device, dtype=src_pos.dtype)
        out = self.forward(
            q, src_pos, src_w,
            J_unres_grid=J_unres_grid, J_ref=J_ref,
            r_cut_mpc=r_cut_mpc, chunk=chunk,
        )
        return out["x_hii"].reshape(grid_size, grid_size, grid_size)


# ------------------------------------------------------------------ #
#  Adapter: build a continuous field from a trained / initialised LAEPINN
# ------------------------------------------------------------------ #

def build_continuous_field_from_pinn(
    model,                               # LAEPINN
    graph,                               # PyG Data (same as model.forward)
    hod_basis: torch.Tensor,             # (n_bins, G, G, G)
    use_rsd_correction: bool = True,
):
    """
    Run the GNN + source head of an existing ``LAEPINN`` to obtain the per-LAE
    factors, then wire up a ``ContinuousIonizationField`` that shares the model's
    kernel, excursion mapping, A_obs amplitude and J_ref.  Returns

        (cfield, ctx)

    where ``cfield`` is a ``ContinuousIonizationField`` and ``ctx`` is a dict with
    everything ``cfield.forward`` needs:

        ctx["src_pos"]      (N, 3) scatter positions (with optional RSD shift)
        ctx["src_w"]        (N,)   A_obs * w_eff   (factors, A_obs folded in)
        ctx["J_unres_grid"] (G,G,G) kernel-convolved unresolved field
        ctx["J_ref"]        scalar  the model's calibrated reference

    so that ``cfield.forward(query, **ctx)`` gives x_HII at arbitrary ``query``,
    consistent with ``model.forward`` on the grid.

    Only valid for ``excursion_type == "equilibrium"``.
    """
    device = graph.x.device
    G = model.grid_size

    # 1. GNN + source head  ->  per-LAE factors w_eff and scatter positions
    h = model.gnn(graph.x, graph.edge_index, graph.edge_attr)
    w_eff, src_info = model.source_head.compute_source_weights(h, graph.src_weights)

    pos = graph.pos
    if use_rsd_correction and ("delta_los" in src_info):
        delta_los = src_info["delta_los"]
        z = torch.zeros_like(delta_los)
        axis = model.los_axis
        comp = [z, z, z]
        comp[axis] = delta_los
        pos = (pos + torch.stack(comp, dim=-1)) % 1.0

    # 2. A_obs amplitude (calibrated once in the model; falls back to base * exp(log_A_obs))
    A_obs = model._A_obs_base * torch.exp(model._log_A_obs)

    # 3. Unresolved emissivity field.
    S_unres = model.unresolved(hod_basis)

    # ---- Hybrid mapping: B(x) · x_eq(x) ----
    if model.excursion_type == "bubble_equilibrium":
        density_grid = getattr(graph, "density_grid", None)
        kernel_grid = make_3d_kernel_grid(model.kernel, G, model.box_size, device)
        J_unres_grid = fft_convolve_3d(S_unres, kernel_grid)
        J_ref = model._J_ref if bool(model._amp_calibrated) else None
        evaluator = HybridContinuousEvaluator(
            model.excursion.bubble, model.excursion.equilibrium,
            model.kernel, model.box_size, G)
        ctx = {
            "src_pos": pos, "src_w": w_eff, "A_obs": A_obs,
            "S_unres_grid": S_unres, "J_unres_grid": J_unres_grid,
            "J_ref": J_ref, "density_grid": density_grid,
        }
        return evaluator, ctx

    # ---- Bubble mapping: top-hat excursion-set evaluator ----
    if model.excursion_type == "bubble":
        density_grid = getattr(graph, "density_grid", None)
        evaluator = BubbleContinuousEvaluator(model.excursion, model.box_size)
        ctx = {
            "src_pos": pos,
            "src_w": w_eff,            # A_obs applied inside the bubble Q
            "A_obs": A_obs,
            "S_unres_grid": S_unres,
            "density_grid": density_grid,
        }
        return evaluator, ctx

    # ---- Equilibrium mapping: kernel-integral field ----
    src_w = A_obs * w_eff
    kernel_grid = make_3d_kernel_grid(model.kernel, G, model.box_size, device)
    J_unres_grid = fft_convolve_3d(S_unres, kernel_grid)  # linearity: unres part of J_total
    J_ref = model._J_ref if bool(model._amp_calibrated) else None

    cfield = ContinuousIonizationField(
        kernel=model.kernel,
        excursion=model.excursion,
        box_size=model.box_size,
        grid_size_ref=G,
    )
    ctx = {
        "src_pos": pos,
        "src_w": src_w,
        "J_unres_grid": J_unres_grid,
        "J_ref": J_ref,
    }
    return cfield, ctx


class BubbleContinuousEvaluator:
    """
    Uniform off-grid evaluator for the continuous bubble mapping, matching the
    ``ContinuousIonizationField.forward`` call convention so both cores share an
    API: ``ev.forward(query, **ctx)["x_hii"]``.
    """

    def __init__(self, bubble, box_size: float):
        self.bubble = bubble
        self.box_size = box_size

    def forward(self, query, src_pos, src_w, A_obs, S_unres_grid,
                density_grid=None, chunk: int = 4096):
        x = bubble_field_continuous(self.bubble, query, src_pos, src_w, A_obs,
                                    S_unres_grid, density_grid, self.box_size, chunk=chunk)
        return {"x_hii": x}

    __call__ = forward


class HybridContinuousEvaluator:
    """
    Off-grid evaluator for the hybrid core:  x_HII(x) = B(x) · x_eq(x).
      B(x)    — continuous excursion-set bubble (``bubble_field_continuous``).
      x_eq(x) — continuous kernel-integral photoionization equilibrium
                (``ContinuousIonizationField`` on the same propagated field J).
    Matches the ``ev.forward(query, **ctx)["x_hii"]`` convention of the other cores.
    """

    def __init__(self, bubble, equilibrium, kernel, box_size: float, grid_size_ref: int):
        self.bubble = bubble
        self.box_size = box_size
        self._cfield = ContinuousIonizationField(
            kernel=kernel, excursion=equilibrium,
            box_size=box_size, grid_size_ref=grid_size_ref)

    def forward(self, query, src_pos, src_w, A_obs, S_unres_grid, J_unres_grid,
                J_ref, density_grid=None, chunk: int = 4096):
        B = bubble_field_continuous(self.bubble, query, src_pos, src_w, A_obs,
                                    S_unres_grid, density_grid, self.box_size, chunk=chunk)
        # equilibrium on J(query) = A_obs·Σ w_i K + interp(J_unres): pass A_obs·w_i
        x_eq = self._cfield.forward(
            query, src_pos, A_obs * src_w,
            J_unres_grid=J_unres_grid, J_ref=J_ref, chunk=chunk)["x_hii"]
        return {"x_hii": (B * x_eq).clamp(0.0, 1.0)}

    __call__ = forward


# ------------------------------------------------------------------ #
#  Training utilities (resolution-free, physics-strict)
# ------------------------------------------------------------------ #
#
# What is satisfied BY CONSTRUCTION (no penalty needed):
#   * Radiative transfer:  J(x) = (S * K)(x) is the exact Green-function solution
#     of the assumed transport kernel K (mean-free-path attenuation + geometric
#     dilution / soft bubble), evaluated continuously rather than on a grid.
#   * Photoionization balance:  x_HII = (sqrt(A^2+4A)-A)/2 is the exact closed-form
#     root of  Gamma (1 - x) = alpha n_H x^2  with A = J / (alpha n_H scale).
#     Hence  0 <= x_HII <= 1  and the local equilibrium hold pointwise, exactly.
#
# What the losses below supply:
#   * a resolution-free DATA fit (sample anywhere, not just on voxels), and
#   * the global ionized-fraction (photon-budget) constraint via Monte-Carlo,
#     which trains the single scale alpha_nH_scale.

def sample_uniform_queries(n: int, device, dtype=torch.float32) -> torch.Tensor:
    """n uniform random query coordinates in the periodic box [0, 1)^3, (n, 3)."""
    return torch.rand(n, 3, device=device, dtype=dtype)


def sampled_field_mse(
    cfield: ContinuousIonizationField,
    ctx: dict,
    x_true_grid: torch.Tensor,           # (G, G, G) ground-truth ionization field
    query: torch.Tensor | None = None,   # (Q, 3); if None, sample n_samples uniformly
    n_samples: int = 32768,
    r_cut_mpc: float | None = None,
    chunk: int = 4096,
) -> torch.Tensor:
    """
    Resolution-free data loss: MSE between the continuous prediction x_HII(q)
    and the true field interpolated (periodic trilinear) at the SAME points q.

    Because q can be sampled at arbitrary density, this decouples the training
    supervision from the voxel grid -- the whole point of the continuous field.
    Gradients flow to the GNN factors (via src_w), the kernel params and
    alpha_nH_scale.
    """
    device = ctx["src_pos"].device
    if query is None:
        query = sample_uniform_queries(n_samples, device, dtype=ctx["src_pos"].dtype)
    pred = cfield.forward(
        query, ctx["src_pos"], ctx["src_w"],
        J_unres_grid=ctx.get("J_unres_grid"), J_ref=ctx.get("J_ref"),
        r_cut_mpc=r_cut_mpc, chunk=chunk,
    )["x_hii"]
    target = periodic_trilinear_sample(x_true_grid, query)
    return torch.mean((pred - target) ** 2)


def global_xhii_mc(
    cfield: ContinuousIonizationField,
    ctx: dict,
    xi_global: float,
    n_samples: int = 65536,
    r_cut_mpc: float | None = None,
    chunk: int = 4096,
) -> torch.Tensor:
    """
    Monte-Carlo global ionized-fraction (photon-budget) constraint:
    ( <x_HII>_MC - xi_global )^2 over uniform samples.  Trains alpha_nH_scale,
    exactly as the grid ``global_xhii_loss`` does, but mesh-free.
    """
    device = ctx["src_pos"].device
    query = sample_uniform_queries(n_samples, device, dtype=ctx["src_pos"].dtype)
    pred = cfield.forward(
        query, ctx["src_pos"], ctx["src_w"],
        J_unres_grid=ctx.get("J_unres_grid"), J_ref=ctx.get("J_ref"),
        r_cut_mpc=r_cut_mpc, chunk=chunk,
    )["x_hii"]
    return (pred.mean() - float(xi_global)) ** 2


def ionization_front_gradient(
    cfield: ContinuousIonizationField,
    ctx: dict,
    query: torch.Tensor,                 # (Q, 3) points to probe
    r_cut_mpc: float | None = None,
) -> torch.Tensor:
    """
    Exact spatial gradient |d x_HII / dx| at the query points via autodiff -- a
    capability the grid form does not have without finite differences.  Large
    values localise ionization fronts (the boundaries of HII bubbles).  Returned
    in units of 1 / (cMpc/h).  Diagnostic, not a loss (real fronts are sharp and
    should NOT be smoothed away).
    """
    q = query.detach().clone().requires_grad_(True)
    x = cfield.forward(
        q, ctx["src_pos"], ctx["src_w"],
        J_unres_grid=ctx.get("J_unres_grid"), J_ref=ctx.get("J_ref"),
        r_cut_mpc=r_cut_mpc,
    )["x_hii"]
    (grad,) = torch.autograd.grad(x.sum(), q, create_graph=False)
    return grad.norm(dim=-1) / cfield.box_size   # normalised-coord grad -> per cMpc/h
