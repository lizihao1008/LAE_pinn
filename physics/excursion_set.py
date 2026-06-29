"""
physics/excursion_set.py
Photoionization-equilibrium mapping: J(x) → x_HII(x).

    A ≡ Γ_HI / (α n_H)  ∝  J / s        (s: single learnable α·n_H scale)
    x_HII = (√(A² + 4A) - A) / 2

    J = 0  ⟹  x_HII = 0
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
