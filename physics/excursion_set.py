"""
physics/excursion_set.py
Photoionization-equilibrium mapping: J(x) → x_HII(x).

    A ≡ Γ_HI / (α n_H)  ∝  J / s        (s: single learnable α·n_H scale)
    x_HII = (√(A² + 4A) - A) / 2

    J = 0  ⟹  x_HII = 0
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _softplus_inverse(y: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus: x such that softplus(x) = y (y > 0)."""
    y = torch.as_tensor(y).clamp(min=1e-8)
    return torch.log(torch.expm1(y).clamp(min=1e-12))


def equilibrium_x_hii(A: torch.Tensor) -> torch.Tensor:
    """Ionized fraction from A = Γ_HI/(α n_H); exact solution of A(1-x)=x²."""
    A = A.clamp(min=0.0)
    return (torch.sqrt(A * A + 4.0 * A + 1e-12) - A) * 0.5


def equilibrium_x_hi(A: torch.Tensor) -> torch.Tensor:
    """Neutral fraction x_HI = 1 - x_HII (same quadratic equilibrium)."""
    A = A.clamp(min=0.0)
    return (2.0 + A - torch.sqrt(A * A + 4.0 * A + 1e-12)) * 0.5


class ExcursionSetMapping(nn.Module):
    """
    J → x_HII via photoionization equilibrium.

    One learnable scale s > 0 absorbs α(T)·⟨n_H⟩ and J unit conversion.
    Optional gain g (binary search, no grad) enforces ⟨x_HII⟩ = ξ_global.
    """

    def __init__(
        self,
        alpha_nH_scale_init: float = 1.0,
        learnable: bool = True,
        # deprecated aliases
        alpha_scale_init: float | None = None,
        J_scale_init: float | None = None,
        sharpness_init: float | None = None,
        threshold_init: float | None = None,
    ):
        super().__init__()
        init = alpha_nH_scale_init
        if alpha_scale_init is not None:
            init = alpha_scale_init
        if J_scale_init is not None:
            init = J_scale_init
        if sharpness_init is not None:
            init = sharpness_init
        _ = threshold_init

        raw = math.log(math.expm1(max(init, 1e-6)))
        if learnable:
            # s ≡ α(T)·⟨n_H⟩ (effective); n_H not observed, folded into this scalar
            self._scale_raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        else:
            self.register_buffer("_scale_raw", torch.tensor(raw, dtype=torch.float32))

    @property
    def alpha_nH_scale(self) -> torch.Tensor:
        return F.softplus(self._scale_raw) + 1e-8

    # backward-compatible aliases
    @property
    def alpha_scale(self) -> torch.Tensor:
        return self.alpha_nH_scale

    @property
    def J_scale(self) -> torch.Tensor:
        return self.alpha_nH_scale

    def forward(
        self,
        J_total: torch.Tensor,
        xi_global: float | None = None,
        n_H: torch.Tensor | None = None,  # ignored; kept for call-site compat
    ) -> torch.Tensor:
        _ = n_H  # n_H not used: α and ⟨n_H⟩ merged into alpha_nH_scale
        J = J_total.clamp(min=0.0)  # flux ≥ 0  →  x_HII(0)=0 by construction
        scale = self.alpha_nH_scale
        # A = J/s;  x_HII = (√(A²+4A)−A)/2
        x_hii = equilibrium_x_hii(J / scale)

        if xi_global is not None:
            # Global constraint: find g so ⟨x_HII⟩ matches target ξ (no gradient)
            with torch.no_grad():
                xi_t = torch.tensor(xi_global, dtype=J.dtype, device=J.device)
                gain = self._calibrate_gain(J, scale, xi_t)
            x_hii = equilibrium_x_hii(gain * J / scale)

        return x_hii

    @staticmethod
    def _calibrate_gain(
        J: torch.Tensor,
        scale: torch.Tensor,
        xi_target: torch.Tensor,
        n_iter: int = 40,
    ) -> torch.Tensor:
        # Monotone in g: g=0 → all neutral; large g → saturated ionization
        lo = torch.zeros((), device=J.device, dtype=J.dtype)
        hi = torch.ones((), device=J.device, dtype=J.dtype)

        for _ in range(24):
            if equilibrium_x_hii(hi * J / scale).mean() >= xi_target:
                break
            hi = hi * 2.0
        hi = hi.clamp(max=1e6)

        for _ in range(n_iter):
            mid = (lo + hi) / 2
            val = equilibrium_x_hii(mid * J / scale).mean()
            if val > xi_target:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    def extra_repr(self) -> str:
        return f"alpha_nH_scale={self.alpha_nH_scale.item():.4f}"


# ------------------------------------------------------------------ #
#  Excursion-set BUBBLE mapping (sharp 0/1 reionization topology)
# ------------------------------------------------------------------ #

def _tophat_kernels(radii_mpc, grid_size, box_size, device):
    """
    Build a stack of normalised top-hat smoothing kernels on the grid,
    rolled so zero-lag is at index [0,0,0] (FFT-ready).

    Returns (n_scales, G, G, G) float32.
    """
    G  = grid_size
    dx = box_size / G
    coords = (torch.arange(G, device=device, dtype=torch.float32) - G // 2) * dx
    cx, cy, cz = torch.meshgrid(coords, coords, coords, indexing="ij")
    r = (cx ** 2 + cy ** 2 + cz ** 2).sqrt()
    ks = []
    for R in radii_mpc:
        k = (r <= float(R)).float()
        k = k / (k.sum() + 1e-12)
        k = torch.roll(k, (G // 2, G // 2, G // 2), dims=(0, 1, 2))
        ks.append(k.contiguous())
    return torch.stack(ks, dim=0)


class BubbleExcursionSet(nn.Module):
    """
    Differentiable excursion-set bubble model (Furlanetto-Zaldarriaga-Hernquist
    style): a cell is ionized if, smoothed on SOME scale R, the cumulative
    ionizing photons exceed the cumulative neutral hydrogen, i.e.

        Q_R(x) = zeta * <S_emiss>_R(x) / <n_H>_R(x)  >=  1   for some R.

    Differentiable surrogate
    ------------------------
        p_R(x) = sigmoid( s * (Q_R(x) - 1) )            soft threshold per scale
        x_HII(x) = 1 - Prod_R ( 1 - p_R(x) )            ionized if ANY scale wins

    This yields sharp 0/1 topology: cells deep inside ionized regions exceed the
    threshold on large R (x -> 1); cells in voids fall below threshold on every
    R (x -> 0).  Unlike the smooth equilibrium x(J), neutral islands (x ~ 0) are
    naturally produced -- removing the x_HII floor.

    Learnable parameters
    --------------------
        zeta  : ionizing efficiency (sets the global ionized fraction; trained
                by the global_xHII loss).  zeta = softplus(_zeta_raw).
        s     : threshold sharpness.  s = softplus(_sharp_raw).
    """

    def __init__(
        self,
        grid_size: int = 64,
        box_size: float = 160.0,
        radii_mpc: list[float] | None = None,
        zeta_init: float = 1.0,
        sharpness_init: float = 6.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.box_size  = box_size
        if radii_mpc is None:
            # logarithmic scales from ~1 voxel to ~quarter box
            dx = box_size / grid_size
            radii_mpc = list(np.geomspace(dx, box_size / 4.0, 6))
        self.radii_mpc = list(radii_mpc)

        raw_zeta  = math.log(math.expm1(max(zeta_init, 1e-6)))
        raw_sharp = math.log(math.expm1(max(sharpness_init, 1e-6)))
        if learnable:
            self._zeta_raw  = nn.Parameter(torch.tensor(raw_zeta,  dtype=torch.float32))
            self._sharp_raw = nn.Parameter(torch.tensor(raw_sharp, dtype=torch.float32))
        else:
            self.register_buffer("_zeta_raw",  torch.tensor(raw_zeta,  dtype=torch.float32))
            self.register_buffer("_sharp_raw", torch.tensor(raw_sharp, dtype=torch.float32))

        # Fixed top-hat smoothing kernels (one per scale)
        self.register_buffer(
            "_th_kernels",
            _tophat_kernels(self.radii_mpc, grid_size, box_size, device="cpu"),
        )

    @property
    def zeta(self) -> torch.Tensor:
        return F.softplus(self._zeta_raw) + 1e-8

    @property
    def sharpness(self) -> torch.Tensor:
        return F.softplus(self._sharp_raw) + 1e-8

    def forward(
        self,
        S_emiss: torch.Tensor,                 # (G,G,G) ionizing source/emissivity field
        density: torch.Tensor | None = None,   # (G,G,G) gas density ~ (1+delta); default uniform
    ) -> torch.Tensor:
        G = self.grid_size
        S = S_emiss.clamp(min=0.0).float().contiguous()
        n_H = (density.clamp(min=1e-6).float().contiguous()
               if density is not None else torch.ones_like(S))

        dims = (-3, -2, -1)
        S_fft = torch.fft.rfftn(S,   dim=dims)
        H_fft = torch.fft.rfftn(n_H, dim=dims)

        zeta = self.zeta
        s    = self.sharpness
        log_one_minus_p = torch.zeros_like(S)     # accumulate log(1 - p_R)
        for j in range(self._th_kernels.shape[0]):
            K_fft = torch.fft.rfftn(self._th_kernels[j], dim=dims)
            S_bar = torch.fft.irfftn(S_fft * K_fft, s=S.shape, dim=dims).clamp(min=0.0)
            H_bar = torch.fft.irfftn(H_fft * K_fft, s=S.shape, dim=dims).clamp(min=1e-8)
            Q = zeta * S_bar / H_bar                       # photon/H ratio at scale R
            p = torch.sigmoid(s * (Q - 1.0))               # soft threshold
            log_one_minus_p = log_one_minus_p + torch.log1p(-p.clamp(max=1 - 1e-6))

        x_hii = 1.0 - torch.exp(log_one_minus_p)           # ionized if ANY scale wins
        return x_hii.clamp(0.0, 1.0)

    @torch.no_grad()
    def calibrate_zeta(self, S_emiss, density=None, target=0.5, n_iter=40):
        """
        One-time calibration: set zeta so <x_HII> ~ target on the given field.
        Avoids a saturated start (x~0 or x~1 everywhere) that would stall the
        global_xHII gradient.  Bisection in log-zeta; no gradient.
        """
        lo = torch.tensor(-12.0)   # log zeta
        hi = torch.tensor(12.0)
        t  = float(target)
        for _ in range(n_iter):
            mid = (lo + hi) / 2
            self._zeta_raw.data.copy_(_softplus_inverse(mid.exp()))
            xm = self.forward(S_emiss, density).mean().item()
            if xm > t:
                hi = mid
            else:
                lo = mid
        self._zeta_raw.data.copy_(_softplus_inverse(((lo + hi) / 2).exp()))

    def get_params_dict(self) -> dict:
        return {"zeta": float(self.zeta.item()), "sharpness": float(self.sharpness.item())}

    def extra_repr(self) -> str:
        return (f"zeta={self.zeta.item():.3f}, sharpness={self.sharpness.item():.2f}, "
                f"n_scales={len(self.radii_mpc)}")
