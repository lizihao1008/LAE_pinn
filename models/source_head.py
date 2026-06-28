"""
models/source_head.py
Source inference head: environment embedding h_i → physical source quantities.

Outputs (per LAE):
    f_esc,i  ∈ (0, 1)     escape fraction
    ξ_ion,i  ∈ R+         ionizing photon production efficiency (optional)

The effective ionizing luminosity weight is:
    w_i = L_i · ξ_ion · f_esc,i

where L_i is the observed Lyα luminosity (passed separately from the catalog).

f_esc is physically constrained to (0, 1) via sigmoid.
ξ_ion is constrained > 0 via softplus (or held fixed).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SourceHead(nn.Module):
    """
    MLP head mapping environment embeddings to physical source parameters.

    Parameters
    ----------
    in_dim : int
        Dimension of GNN node embedding.
    hidden_dims : list of int
        Hidden layer sizes.
    xi_ion_learnable : bool
        If False, ξ_ion is fixed at 10^xi_ion_log for all galaxies.
    xi_ion_log : float
        log10(ξ_ion) when xi_ion_learnable=False (default: 25.2).
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden_dims: list[int] | None = None,
        xi_ion_learnable: bool = False,
        xi_ion_log: float = 25.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        # Build MLP for f_esc
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))   # logit → sigmoid → f_esc ∈ (0,1)
        self.f_esc_mlp = nn.Sequential(*layers)

        # ξ_ion head (optional)
        self.xi_ion_learnable = xi_ion_learnable
        if xi_ion_learnable:
            xi_layers = []
            prev = in_dim
            for h in hidden_dims:
                xi_layers += [nn.Linear(prev, h), nn.ELU()]
                prev = h
            xi_layers.append(nn.Linear(prev, 1))   # → softplus → ξ_ion > 0
            self.xi_ion_mlp = nn.Sequential(*xi_layers)
        else:
            self.register_buffer("xi_ion_fixed",
                                 torch.tensor(10 ** xi_ion_log, dtype=torch.float32))

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        h : (N, in_dim) node embeddings

        Returns
        -------
        dict with:
            f_esc : (N,) escape fractions ∈ (0, 1)
            xi_ion: (N,) or scalar ionizing photon efficiency
        """
        f_esc_logit = self.f_esc_mlp(h).squeeze(-1)   # (N,)
        f_esc = torch.sigmoid(f_esc_logit)              # (N,) ∈ (0, 1)

        if self.xi_ion_learnable:
            xi_logit = self.xi_ion_mlp(h).squeeze(-1)
            xi_ion = F.softplus(xi_logit)               # (N,) > 0
        else:
            xi_ion = self.xi_ion_fixed.expand(f_esc.shape[0])

        return {"f_esc": f_esc, "xi_ion": xi_ion}

    def compute_source_weights(
        self,
        h: torch.Tensor,
        src_weights_raw: torch.Tensor,  # (N,) = L_i normalised
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute effective source weights w_i = L_i_norm · f_esc_i
        (ξ_ion is absorbed into the overall normalisation of J).

        Returns
        -------
        w_eff : (N,) effective weights
        info  : dict with f_esc and xi_ion for logging
        """
        out = self.forward(h)
        f_esc  = out["f_esc"]
        xi_ion = out["xi_ion"]

        # Effective weight: luminosity × escape fraction
        # ξ_ion is a global scale so it only shifts the threshold, not topology
        w_eff = src_weights_raw * f_esc   # (N,)

        return w_eff, out
