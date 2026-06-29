"""
physics/scatter.py
Trilinear scattering: deposit LAE source weights onto a 3D grid.

J_obs(x) = Σ_i w_i · δ³(x - x_i)   [scatter]
         → convolved with K(r) in kernels.py

The scatter is differentiable w.r.t. w_i (and indirectly w.r.t. the GNN
parameters that produce f_esc → w_i = L_i * f_esc_i).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def scatter_to_grid(
    pos_norm: torch.Tensor,    # (N, 3) in [0, 1]
    weights: torch.Tensor,     # (N,) source weights
    grid_size: int,
    mode: str = "trilinear",   # "trilinear" | "nearest"
) -> torch.Tensor:
    """
    Scatter point source weights onto a (G, G, G) grid using
    differentiable trilinear interpolation (reverse of grid_sample).

    pos_norm: float32 (N, 3), each coordinate in [0, 1]
    weights:  float32 (N,)
    Returns:  float32 (G, G, G) grid with deposited source weights.

    Implementation: we use a manual trilinear splat loop.
    For large N this can be replaced with a CUDA atomic-add approach.
    """
    G = grid_size
    N = pos_norm.shape[0]
    device = pos_norm.device
    dtype  = pos_norm.dtype

    # Pixel coordinates in [0, G)
    pix = pos_norm * G          # (N, 3), range [0, G)
    pix = pix.clamp(0, G - 1e-6)

    # Integer lower-left corner
    i0 = pix.floor().long()     # (N, 3)
    i1 = (i0 + 1).clamp(0, G - 1)

    # Fractional offsets
    frac = pix - i0.float()     # (N, 3), in [0, 1)
    f0 = 1.0 - frac             # (N, 3)
    f1 = frac

    # Collect all 8 corner contributions then do a single non-in-place
    # scatter_add so that gradients flow back to `weights` (and hence to
    # the GNN that produced f_esc).  In-place scatter_add_ on a leaf
    # tensor with requires_grad=False breaks the autograd graph.
    flat_list: list[torch.Tensor] = []
    w_list:    list[torch.Tensor] = []

    for bx, wx in [(i0[:, 0], f0[:, 0]), (i1[:, 0], f1[:, 0])]:
        for by, wy in [(i0[:, 1], f0[:, 1]), (i1[:, 1], f1[:, 1])]:
            for bz, wz in [(i0[:, 2], f0[:, 2]), (i1[:, 2], f1[:, 2])]:
                w_corner = weights * wx * wy * wz      # (N,)
                flat     = (bx % G) * G * G + (by % G) * G + (bz % G)  # (N,)
                flat_list.append(flat)
                w_list.append(w_corner)

    flat_cat = torch.cat(flat_list, dim=0)   # (8*N,)
    w_cat    = torch.cat(w_list,    dim=0)   # (8*N,)  requires_grad == weights.requires_grad

    # out-of-place scatter_add: differentiable w.r.t. w_cat
    grid = torch.zeros(G * G * G, device=device, dtype=dtype).scatter_add(
        0, flat_cat, w_cat
    )
    return grid.view(G, G, G).contiguous()   # (G, G, G)


def fft_convolve_3d(
    source_grid: torch.Tensor,   # (G, G, G)
    kernel_grid: torch.Tensor,   # (G, G, G), pre-normalised, centred at [0,0,0]
) -> torch.Tensor:
    """
    FFT-based 3D convolution (periodic).
    Both grids must have the same shape (G, G, G).
    Kernel must be rolled so zero-lag is at index [0,0,0] (see kernels.py).

    Returns (G, G, G) float32 convolution result.
    Differentiable w.r.t. source_grid; kernel_grid needs separate handling
    (we rebuild it from learnable parameters each step via make_3d_kernel_grid).
    """
    # rfftn requires contiguous float input (roll/strided views trigger resize warnings)
    _dims = (-3, -2, -1)
    x = source_grid.float().contiguous()
    k = kernel_grid.float().contiguous()
    S_fft = torch.fft.rfftn(x, dim=_dims)
    K_fft = torch.fft.rfftn(k, dim=_dims)
    J_fft = S_fft * K_fft
    J = torch.fft.irfftn(J_fft, s=x.shape, dim=_dims)
    return J.clamp(min=0.)   # ionizing flux must be non-negative


def scatter_and_convolve(
    pos_norm: torch.Tensor,      # (N, 3) in [0, 1]
    weights: torch.Tensor,       # (N,) effective source weights = L_i * f_esc_i
    kernel_grid: torch.Tensor,   # (G, G, G) pre-built from learnable kernel
    grid_size: int,
) -> torch.Tensor:
    """
    Full pipeline: scatter → FFT convolve.
    Returns J_obs(x) on the (G, G, G) grid.
    """
    source_grid = scatter_to_grid(pos_norm, weights, grid_size)
    J = fft_convolve_3d(source_grid, kernel_grid)
    return J
