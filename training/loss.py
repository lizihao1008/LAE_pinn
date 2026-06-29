"""
training/loss.py
Composite loss function for the LAE PINN.

L = L_field + λ_ps * L_ps + λ_bce * L_bce
  + λ_mcf * L_mcf + λ_xHII * L_xHII + λ_prior * L_prior

Components:
    L_field  — voxel MSE between predicted and true x_HII
    L_ps     — power spectrum MSE (isotropic 3D P(k))
    L_bce    — binary topology BCE (threshold at 0.5)
    L_mcf    — MCF consistency (predicted field vs true MCF, optional)
    L_xHII   — global ionized fraction constraint
    L_prior  — KL to LF prior on F_b + TV smoothness on ε_unres
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Individual loss terms
# ------------------------------------------------------------------ #

def field_mse_loss(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """Voxel-wise MSE between predicted and true ionization field."""
    return F.mse_loss(x_pred, x_true)


def power_spectrum_loss(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    n_k_bins: int = 20,
) -> torch.Tensor:
    """
    Isotropic 3D power spectrum MSE.
    Computed via FFT on the (G,G,G) field.
    """
    def _ps(field):
        fk = torch.fft.rfftn(field)
        pk = (fk.abs() ** 2)
        G  = field.shape[0]
        # k-grid
        kx = torch.fft.fftfreq(G, d=1.0 / G, device=field.device)
        ky = torch.fft.fftfreq(G, d=1.0 / G, device=field.device)
        kz = torch.fft.rfftfreq(G, d=1.0 / G, device=field.device)
        kx3, ky3, kz3 = torch.meshgrid(kx, ky, kz, indexing="ij")
        k_mag = (kx3 ** 2 + ky3 ** 2 + kz3 ** 2).sqrt()
        k_flat = k_mag.flatten()
        pk_flat = pk.flatten()
        # Bin into shells
        k_max = k_flat.max()
        bins  = torch.linspace(0, k_max + 1e-6, n_k_bins + 1, device=field.device)
        ps_binned = torch.zeros(n_k_bins, device=field.device)
        for i in range(n_k_bins):
            mask = (k_flat >= bins[i]) & (k_flat < bins[i + 1])
            if mask.sum() > 0:
                ps_binned[i] = pk_flat[mask].mean()
        return ps_binned

    ps_pred = _ps(x_pred)
    ps_true = _ps(x_true)
    return F.mse_loss(
        torch.log(ps_pred + 1e-12),
        torch.log(ps_true + 1e-12)
    )


def binary_bce_loss(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Binary topology loss: BCE between predicted field and binary true field.

    nan_to_num guard: NaN in x_pred (e.g. from extreme early-training
    activations) survives clamp and causes F.binary_cross_entropy to raise
    "all elements of input should be between 0 and 1".  Map NaN/Inf safely
    before clamping.
    """
    x_true_bin = (x_true > threshold).float()
    x_pred_safe = torch.nan_to_num(x_pred, nan=0.5, posinf=1.0, neginf=0.0)
    x_pred_safe = x_pred_safe.clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(x_pred_safe, x_true_bin)


def global_xhii_loss(
    x_pred: torch.Tensor,
    xi_global: float,
) -> torch.Tensor:
    """Penalise deviation of mean(x_pred) from the true global ionized fraction."""
    xi_pred = x_pred.mean()
    return (xi_pred - xi_global) ** 2


def prior_loss(
    unresolved_module,   # HODUnresolvedField (or any module with lf_kl_loss())
    J_unres: torch.Tensor,
    lf_weight: float = 0.1,
    smooth_weight: float = 1e-3,
) -> torch.Tensor:
    """
    LF KL prior on F_b + total-variation smoothness on ε_unres.
    """
    lf_kl = unresolved_module.lf_kl_loss()

    # Total variation on ε_unres (J_unres here is ε_unres for MVP)
    tv = (
        (J_unres[1:, :, :] - J_unres[:-1, :, :]).abs().mean() +
        (J_unres[:, 1:, :] - J_unres[:, :-1, :]).abs().mean() +
        (J_unres[:, :, 1:] - J_unres[:, :, :-1]).abs().mean()
    )

    return lf_weight * lf_kl + smooth_weight * tv


# ------------------------------------------------------------------ #
#  Composite loss
# ------------------------------------------------------------------ #

class PINNLoss(torch.nn.Module):

    def __init__(
        self,
        field_mse_w:    float = 1.0,
        power_spec_w:   float = 0.1,
        binary_bce_w:   float = 0.5,
        mcf_w:          float = 0.0,    # 0 → disabled by default in MVP
        global_xhii_w:  float = 1.0,
        prior_w:        float = 0.1,
        lf_weight:      float = 0.1,
        smooth_weight:  float = 1e-3,
    ):
        super().__init__()
        self.w = {
            "field":   field_mse_w,
            "ps":      power_spec_w,
            "bce":     binary_bce_w,
            "mcf":     mcf_w,
            "xhii":    global_xhii_w,
            "prior":   prior_w,
        }
        self.lf_weight     = lf_weight
        self.smooth_weight = smooth_weight

    def forward(
        self,
        model_out: dict,           # output of LAEPINN.forward(return_intermediates=True)
        x_true: torch.Tensor,      # (G, G, G)
        xi_global: float,
        unresolved_module,         # for prior loss
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns: (total_loss, loss_components_dict)
        """
        x_pred = model_out["x_hii_pred"]
        J_unres = model_out.get("J_unres", torch.zeros_like(x_pred))

        losses = {}

        losses["field"] = field_mse_loss(x_pred, x_true) * self.w["field"]
        losses["ps"]    = power_spectrum_loss(x_pred, x_true) * self.w["ps"]
        losses["bce"]   = binary_bce_loss(x_pred, x_true) * self.w["bce"]
        losses["xhii"]  = global_xhii_loss(x_pred, xi_global) * self.w["xhii"]
        losses["prior"] = prior_loss(
            unresolved_module, J_unres,
            self.lf_weight, self.smooth_weight
        ) * self.w["prior"]

        total = sum(losses.values())
        return total, {k: float(v.item()) for k, v in losses.items()}
