"""
physics/kernels.py
Physical radiative transfer kernels with learnable parameters.

Design principle:
    - Functional FORM is fixed (physics-motivated)
    - Parameters (R, Δ, λ_mfp, mixture weights) are LEARNABLE
    - All parameters are constrained to physical range via reparameterisation
    - No unconstrained MLP is used to represent the kernel

Implemented kernels:
    1. K_mfp(r; λ)          — exponential decay (large-scale MFP / transmission coherence)
    2. K_bub(r; R, Δ)       — soft sigmoid bubble (geometric ionized region)
    3. K_mix(r; R, Δ, λ, A) — mixture of the two above

Usage:
    kernel = MixtureKernel(R_init=5., delta_init=1., lambda_mfp_init=20.)
    r = torch.linspace(0, 50, 100)
    k_vals = kernel(r)           # (100,) — normalised kernel values

For 3D convolution: precompute K(r) on a 3D distance grid, then FFT-convolve.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Individual kernels (functional, non-parametric)
# ------------------------------------------------------------------ #

def k_mfp(r: torch.Tensor, lambda_mfp: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Exponential decay kernel: exp(-r/λ) / (4π r² + ε)
    Represents photon propagation with mean free path λ.
    λ is constrained > 0 via softplus.
    """
    lam = F.softplus(lambda_mfp)          # ensure λ > 0
    return torch.exp(-r / lam) / (4 * math.pi * r ** 2 + eps)


def k_bubble(r: torch.Tensor, R: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Soft bubble kernel: σ((R - r) / Δ)
    Represents a geometrically sharp ionized region of radius R
    with edge width Δ.  R and Δ are constrained > 0 via softplus.
    """
    R_pos = F.softplus(R)
    D_pos = F.softplus(delta)
    return torch.sigmoid((R_pos - r) / D_pos)


def normalise_kernel(k: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalise kernel values to sum to 1 (over all r values provided)."""
    return k / (k.sum() + eps)


# ------------------------------------------------------------------ #
#  Learnable kernel modules
# ------------------------------------------------------------------ #

class MFPKernel(nn.Module):
    """Learnable exponential mean-free-path kernel."""

    def __init__(self, lambda_mfp_init: float = 20.0):
        super().__init__()
        # Parameterise via softplus inverse so grad flows naturally
        self._lambda_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(lambda_mfp_init)))
        )

    @property
    def lambda_mfp(self) -> torch.Tensor:
        return F.softplus(self._lambda_raw)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return k_mfp(r, self._lambda_raw)

    def extra_repr(self):
        return f"lambda_mfp={self.lambda_mfp.item():.2f} cMpc/h"


class BubbleKernel(nn.Module):
    """Learnable soft-bubble kernel."""

    def __init__(self, R_init: float = 5.0, delta_init: float = 1.0):
        super().__init__()
        self._R_raw     = nn.Parameter(torch.tensor(math.log(math.expm1(R_init))))
        self._delta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(delta_init))))

    @property
    def R(self) -> torch.Tensor:
        return F.softplus(self._R_raw)

    @property
    def delta(self) -> torch.Tensor:
        return F.softplus(self._delta_raw)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return k_bubble(r, self._R_raw, self._delta_raw)

    def extra_repr(self):
        return f"R={self.R.item():.2f}, delta={self.delta.item():.2f} cMpc/h"


class MixtureKernel(nn.Module):
    """
    Mixture of bubble + exponential kernel:
        K(r) = A_geom * K_bub(r; R, Δ) + A_trans * K_mfp(r; λ)
    with A_geom + A_trans = 1, A ≥ 0 (enforced via softmax).

    This is the DEFAULT kernel.
    """

    def __init__(
        self,
        R_init: float = 5.0,
        delta_init: float = 1.0,
        lambda_mfp_init: float = 20.0,
        A_geom_init: float = 0.5,
        A_trans_init: float = 0.5,
    ):
        super().__init__()
        self.bubble = BubbleKernel(R_init, delta_init)
        self.mfp    = MFPKernel(lambda_mfp_init)
        # Logits for softmax: [A_geom, A_trans]
        self._mix_logits = nn.Parameter(
            torch.tensor([math.log(A_geom_init), math.log(A_trans_init)])
        )

    @property
    def mix_weights(self) -> torch.Tensor:
        """(A_geom, A_trans), sum to 1, both > 0."""
        return F.softmax(self._mix_logits, dim=0)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        w = self.mix_weights
        k = w[0] * self.bubble(r) + w[1] * self.mfp(r)
        return k

    def extra_repr(self):
        w = self.mix_weights
        return (f"A_geom={w[0].item():.3f}, A_trans={w[1].item():.3f}, "
                f"R={self.bubble.R.item():.2f}, delta={self.bubble.delta.item():.2f}, "
                f"lambda={self.mfp.lambda_mfp.item():.2f}")

    def get_params_dict(self) -> dict:
        w = self.mix_weights
        return {
            "A_geom":      w[0].item(),
            "A_trans":     w[1].item(),
            "R_bub":       self.bubble.R.item(),
            "delta_bub":   self.bubble.delta.item(),
            "lambda_mfp":  self.mfp.lambda_mfp.item(),
        }


# ------------------------------------------------------------------ #
#  Kernel registry: pick by name (for config-driven experiments)
# ------------------------------------------------------------------ #

KERNEL_REGISTRY: dict[str, type] = {
    "mfp":     MFPKernel,
    "bubble":  BubbleKernel,
    "mixture": MixtureKernel,
}


def build_kernel(kernel_type: str, **kwargs) -> nn.Module:
    if kernel_type not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel type '{kernel_type}'. "
                         f"Available: {list(KERNEL_REGISTRY)}")
    return KERNEL_REGISTRY[kernel_type](**kwargs)


# ------------------------------------------------------------------ #
#  3D kernel grid (for FFT convolution)
# ------------------------------------------------------------------ #

def make_3d_kernel_grid(
    kernel: nn.Module,
    grid_size: int,
    box_size: float,          # cMpc/h
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Evaluate kernel K(r) on a 3D grid of shape (G, G, G).
    Returns a normalised kernel volume for FFT convolution.

    The kernel is centred at (0,0,0) and wrapped periodically.
    Pixel spacing: box_size / grid_size cMpc/h.
    """
    G = grid_size
    dx = box_size / G

    # Build coordinate grids [-G/2, ..., G/2-1] * dx
    coords = torch.arange(G, dtype=torch.float32, device=device)
    coords = coords - G // 2          # centre
    coords = coords * dx               # to cMpc/h

    # 3D meshgrid
    cx, cy, cz = torch.meshgrid(coords, coords, coords, indexing="ij")
    r3d = (cx ** 2 + cy ** 2 + cz ** 2).sqrt()  # (G, G, G)

    # Clamp r to a minimum of half the voxel size.
    #
    # The k_mfp kernel has a 1/(4πr² + ε) factor that diverges at r=0.
    # With ε=1e-4 and dx=2.5 Mpc/h, k_mfp(r=0)/k_mfp(r=dx) ≈ 890,000×.
    # After normalisation, K is effectively a delta function — convolution
    # reduces to J(x) ≈ S(x) with no photon propagation at all.
    #
    # Physical motivation for the clamp: each grid voxel represents a finite
    # volume of size dx³, so sources are not true point sources.  The effective
    # minimum propagation radius is half the voxel size (dx/2).
    # With this clamp, k_mfp(r=0)/k_mfp(r=dx) ≈ 4×, which is physically
    # reasonable and lets photons propagate meaningfully across the grid.
    r3d = r3d.clamp(min=dx / 2)

    # Evaluate kernel WITHOUT torch.no_grad() so that gradients flow back
    # through K_fft → J_fft → loss to the kernel parameters (R, Δ, λ).
    k3d = kernel(r3d)               # (G, G, G)

    # Roll so that zero-lag is at (0,0,0) — required for FFT convolution
    k3d = torch.roll(k3d, shifts=(G // 2, G // 2, G // 2), dims=(0, 1, 2))

    # Normalise to unit sum; contiguous for FFT (roll leaves non-contiguous storage)
    k3d = k3d / (k3d.sum() + 1e-12)
    return k3d.contiguous()               # (G, G, G)
