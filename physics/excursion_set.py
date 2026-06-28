"""
physics/excursion_set.py
Soft excursion-set mapping: total ionizing flux J(x) → x_HII(x).

Physics motivation:
    The excursion-set formalism maps ionizing photon budget above a threshold
    (recombination rate) to ionized regions.  Here we implement a differentiable
    soft version: a sigmoid with learnable threshold and sharpness.

    x_HII(x) = σ( (J(x) - μ_thresh) / τ )

    with a global normalisation constraint:
        mean(x_HII) ≈ ξ_global = ⟨x_i⟩_z

The threshold μ_thresh and sharpness τ are learnable (or can be fixed for ablation).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExcursionSetMapping(nn.Module):
    """
    Differentiable photon-budget → ionization mapping.

    Parameters
    ----------
    threshold_init : float
        Initial value of the excursion-set threshold (before softplus).
    sharpness_init : float
        Initial value of τ (sharpness), constrained > 0 via softplus.
    learnable : bool
        If False, threshold and sharpness are fixed (for ablation).
    """

    def __init__(
        self,
        threshold_init: float = 0.0,
        sharpness_init: float = 1.0,
        learnable: bool = True,
    ):
        super().__init__()
        if learnable:
            self._thresh_raw   = nn.Parameter(torch.tensor(threshold_init))
            self._sharp_raw    = nn.Parameter(torch.tensor(
                torch.log(torch.expm1(torch.tensor(sharpness_init))).item()
            ))
        else:
            self.register_buffer("_thresh_raw", torch.tensor(threshold_init))
            self.register_buffer("_sharp_raw",  torch.tensor(
                torch.log(torch.expm1(torch.tensor(sharpness_init))).item()
            ))

    @property
    def threshold(self) -> torch.Tensor:
        return self._thresh_raw    # unconstrained; centre of sigmoid

    @property
    def sharpness(self) -> torch.Tensor:
        return F.softplus(self._sharp_raw)   # τ > 0

    def forward(
        self,
        J_total: torch.Tensor,              # (G, G, G) total ionizing flux
        xi_global: float | None = None,     # target global ionized fraction
    ) -> torch.Tensor:
        """
        Maps J_total → x_HII field in [0, 1].

        If xi_global is provided, the threshold is shifted so that the
        global mean of x_HII matches xi_global (global constraint).
        This is applied as a calibration step (not gradient-based),
        complemented by a loss term in training.
        """
        tau = self.sharpness
        x_hii = torch.sigmoid((J_total - self.threshold) / tau)   # (G, G, G)

        if xi_global is not None:
            # Calibrate threshold so mean(x_HII) = xi_global
            # Use binary search on the threshold shift (no-gradient step)
            with torch.no_grad():
                xi_global_t = torch.tensor(xi_global, dtype=J_total.dtype,
                                           device=J_total.device)
                # Compute current mean
                delta = self._calibrate_shift(J_total, xi_global_t, tau)
            x_hii = torch.sigmoid((J_total - self.threshold - delta) / tau)

        return x_hii    # (G, G, G), ∈ [0, 1]

    @staticmethod
    def _calibrate_shift(
        J: torch.Tensor,
        xi_target: torch.Tensor,
        tau: torch.Tensor,
        n_iter: int = 30,
    ) -> torch.Tensor:
        """
        Binary search for the threshold shift δ such that
        mean(σ((J - δ) / τ)) = xi_target.
        """
        lo = J.min() - 10 * tau
        hi = J.max() + 10 * tau
        for _ in range(n_iter):
            mid = (lo + hi) / 2
            val = torch.sigmoid((J - mid) / tau).mean()
            if val > xi_target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def extra_repr(self):
        return (f"threshold={self.threshold.item():.3f}, "
                f"sharpness={self.sharpness.item():.3f}")
