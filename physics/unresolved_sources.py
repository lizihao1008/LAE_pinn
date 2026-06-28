"""
physics/unresolved_sources.py
Constrained unresolved faint-source emissivity field.

ε_unres(x) = Σ_b F_b · ε_b(x)

Constraints:
    F_b ≥ 0,  Σ_b F_b = 1   (enforced via softmax)
    ε_b(x) ≥ 0              (enforced via basis field construction)
    ε_b(x) smooth           (prior on density power-law basis)

This is NOT a free neural field. The basis fields ε_b(x) are computed from
the dark matter density field with fixed functional form (power-law bias)
and optional luminosity-function / halo-mass-function priors on F_b.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnresolvedSourceField(nn.Module):
    """
    Learnable constrained unresolved emissivity field.

    Parameters
    ----------
    n_populations : int
        Number of source populations (e.g. bright / faint / diffuse = 3).
    lf_prior : torch.Tensor or None
        (n_populations,) prior fractions from luminosity function integral.
        If provided, KL divergence to this prior is added to the loss.
    """

    def __init__(
        self,
        n_populations: int = 3,
        lf_prior: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_populations = n_populations

        # Learnable logits for F_b (softmax → constrained fractions)
        # Initialised to uniform
        self._F_logits = nn.Parameter(torch.zeros(n_populations))

        # Optional prior fractions (not learned)
        if lf_prior is not None:
            self.register_buffer("lf_prior", lf_prior.float())
        else:
            self.register_buffer("lf_prior", None)

    @property
    def F_b(self) -> torch.Tensor:
        """(n_populations,) source-budget fractions, sum=1, ≥0."""
        return F.softmax(self._F_logits, dim=0)

    def forward(self, density_basis: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        density_basis : (n_populations, G, G, G)
            Pre-built density-bias basis fields (from preprocessing.py).
            Each is ≥ 0 and normalised to unit mean.

        Returns
        -------
        eps_unres : (G, G, G)
            Unresolved emissivity field, weighted sum of basis fields.
        """
        # F_b: (P,), density_basis: (P, G, G, G)
        fb = self.F_b  # (P,)
        # Weighted sum: Σ_b F_b · ε_b(x)
        eps_unres = torch.einsum("p,pghw->ghw", fb, density_basis)
        return eps_unres   # (G, G, G), ≥0 by construction

    def lf_kl_loss(self) -> torch.Tensor:
        """
        KL divergence from learned F_b to luminosity-function prior.
        Returns scalar tensor (0 if no prior provided).
        """
        if self.lf_prior is None:
            return torch.tensor(0.0, device=self._F_logits.device)
        # KL(F_b || lf_prior)
        fb = self.F_b.clamp(min=1e-8)
        prior = self.lf_prior.clamp(min=1e-8)
        return (fb * (fb.log() - prior.log())).sum()

    def get_fractions(self) -> dict[str, float]:
        fb = self.F_b.detach().cpu()
        labels = ["bright", "faint", "diffuse", "pop4"]
        return {labels[i]: float(fb[i]) for i in range(self.n_populations)}
