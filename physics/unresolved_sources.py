"""
physics/unresolved_sources.py
HOD-based unresolved faint-source emissivity field.

Physical model
--------------
    ε_unres(x) = Σ_b  f_esc_b · ε_b(x)

where
    ε_b(x)   — HOD basis field (fixed, pre-computed before training)
                ε_b(x) = (1 + b_b · δ_dm(x)),  normalised to unit mean.
                Encodes the spatial clustering of faint halos in mass bin b.
    f_esc_b  — ONLY learnable parameter: escape fraction for mass bin b.

Why this is better than ρ^α_b power-law mixing
-----------------------------------------------
    ρ^α_b is an arbitrary polynomial in density with no physical interpretation.
    The HOD approach separates what is physically known (spatial distribution,
    set by dark-matter clustering + HOD) from what is physically unknown
    (escape fractions, learned from the ionization field).

Interface
---------
    Basis fields are calibrated once before training by
    data.preprocessing.build_hod_basis_from_simulation()   (simulation path)
    data.preprocessing.build_hod_basis_from_observations() (real-data path)
    Both return a HODCalibration object whose .basis_fields tensor is passed
    to forward().  The model is identical in both cases; only the basis changes.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class HODUnresolvedField(nn.Module):
    """
    HOD-based unresolved faint-source emissivity.

    Parameters
    ----------
    n_bins : int
        Number of halo-mass / M_UV bins for faint sources (default 3).

    Learnable parameters
    --------------------
    _logit_fesc : (n_bins,)
        Logit-space escape fractions; f_esc_b = sigmoid(_logit_fesc_b) ∈ (0, 1).
        Initialised at 0 → f_esc ≈ 0.5 (agnostic prior).

    Fixed inputs (passed at each forward call)
    ------------------------------------------
    hod_basis : (n_bins, G, G, G)
        Pre-computed HOD basis fields, one per mass bin.
        Computed from build_hod_basis_from_simulation() or
        build_hod_basis_from_observations() and attached to the graph object.
        NOT registered as a buffer; passed explicitly so the same model
        checkpoint works with different calibrations (simulation vs. observations).
    """

    def __init__(self, n_bins: int = 3):
        super().__init__()
        self.n_bins = n_bins
        # Initialise at logit(0.5) = 0 → f_esc = 0.5 at start
        self._logit_fesc = nn.Parameter(torch.zeros(n_bins))

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def f_esc_bins(self) -> torch.Tensor:
        """(n_bins,) escape fractions ∈ (0, 1) for each mass bin."""
        return torch.sigmoid(self._logit_fesc)

    @property
    def F_b(self) -> torch.Tensor:
        """Alias for f_esc_bins (backward-compat with old UnresolvedSourceField)."""
        return self.f_esc_bins

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(self, hod_basis: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hod_basis : (n_bins, G, G, G)
            Pre-computed HOD basis fields, fixed during training.

        Returns
        -------
        eps_unres : (G, G, G)
            Unresolved emissivity field: weighted sum Σ_b f_esc_b · ε_b(x).
        """
        # f_esc_bins: (B,)   hod_basis: (B, G, G, G)
        return torch.einsum("b,bxyz->xyz", self.f_esc_bins, hod_basis)

    # ------------------------------------------------------------------ #
    #  Loss helpers
    # ------------------------------------------------------------------ #

    def lf_kl_loss(self) -> torch.Tensor:
        """
        No analytical prior on escape fractions — returns zero.
        Kept for API compatibility with loss.py (which calls lf_kl_loss()).
        A physics-motivated prior on f_esc(M) can be added here later
        (e.g., f_esc decreasing with halo mass from simulations).
        """
        return torch.tensor(0.0, device=self._logit_fesc.device)

    # ------------------------------------------------------------------ #
    #  Logging helpers
    # ------------------------------------------------------------------ #

    def get_fesc_dict(self, labels: list[str] | None = None) -> dict[str, float]:
        """Return {label: f_esc_val} dict for logging."""
        fesc = self.f_esc_bins.detach().cpu()
        default_labels = [f"fesc_bin{b}" for b in range(self.n_bins)]
        labs = labels if labels is not None else default_labels
        return {labs[b]: float(fesc[b]) for b in range(self.n_bins)}

    def get_fractions(self) -> dict[str, float]:
        """Alias for get_fesc_dict() (backward-compat with old UnresolvedSourceField)."""
        return self.get_fesc_dict()

    def extra_repr(self) -> str:
        fesc = self.f_esc_bins.detach().cpu().tolist()
        vals = ", ".join(f"{v:.3f}" for v in fesc)
        return f"n_bins={self.n_bins}, f_esc=[{vals}]"
