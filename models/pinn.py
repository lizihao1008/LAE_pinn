"""
models/pinn.py
Full PINN: LAE-conditioned photon-budget and ionization-topology inference.

Forward pass:
    LAE graph  →  GNN encoder  →  source head  →  f_esc_i, w_i
               →  scatter w_i onto 3D grid
               →  FFT convolve with learnable physical kernel K(r; θ_K)
               →  J_obs(x)
               →  add ε_unres(x; F_b, density_basis)
               →  J_total(x)
               →  soft excursion-set mapping
               →  x̂_HII(x)

Outputs:
    x_hii_pred :  (G, G, G) predicted ionization field
    F_b        :  (n_populations,) source-budget fractions
    theta_K    :  dict of learnable kernel parameters
    f_esc      :  (N,) per-LAE escape fractions
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.data import Data

from ..physics.kernels import MixtureKernel, build_kernel, make_3d_kernel_grid
from ..physics.scatter import scatter_and_convolve
from ..physics.unresolved_sources import UnresolvedSourceField
from ..physics.excursion_set import ExcursionSetMapping
from .gnn_encoder import build_gnn_encoder
from .source_head import SourceHead


class LAEPINN(nn.Module):
    """
    LAE-conditioned PINN for reionization topology inference.

    Parameters: see __init__ docstring.
    """

    def __init__(
        self,
        # GNN
        gnn_architecture: str = "GATv2Conv",
        gnn_in_channels: int = 8,
        gnn_hidden_dim: int = 64,
        gnn_out_channels: int = 32,
        gnn_n_layers: int = 3,
        gnn_heads: int = 4,
        gnn_dropout: float = 0.1,
        # Source head
        source_hidden_dims: list[int] | None = None,
        xi_ion_learnable: bool = False,
        xi_ion_log: float = 25.2,
        # Kernel
        kernel_type: str = "mixture",
        kernel_R_init: float = 5.0,
        kernel_delta_init: float = 1.0,
        kernel_lambda_mfp_init: float = 20.0,
        # Unresolved sources
        n_populations: int = 3,
        lf_prior: torch.Tensor | None = None,
        # Excursion set
        threshold_init: float = 0.0,
        sharpness_init: float = 1.0,
        excursion_learnable: bool = True,
        # Grid
        grid_size: int = 64,
        box_size: float = 160.0,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.box_size  = box_size

        # --- GNN encoder ---
        self.gnn = build_gnn_encoder(
            architecture=gnn_architecture,
            in_channels=gnn_in_channels,
            hidden_dim=gnn_hidden_dim,
            out_channels=gnn_out_channels,
            n_layers=gnn_n_layers,
            heads=gnn_heads,
            dropout=gnn_dropout,
        )

        # --- Source head ---
        self.source_head = SourceHead(
            in_dim=gnn_out_channels,
            hidden_dims=source_hidden_dims,
            xi_ion_learnable=xi_ion_learnable,
            xi_ion_log=xi_ion_log,
        )

        # --- Physical radiative kernel ---
        self.kernel = build_kernel(
            kernel_type,
            R_init=kernel_R_init,
            delta_init=kernel_delta_init,
            lambda_mfp_init=kernel_lambda_mfp_init,
        ) if kernel_type == "mixture" else build_kernel(
            kernel_type,
            **({} if kernel_type == "bubble" else {})
        )
        # Rebuild for any type cleanly
        from ..physics.kernels import KERNEL_REGISTRY
        k_kwargs = {}
        if kernel_type == "mixture":
            k_kwargs = dict(
                R_init=kernel_R_init,
                delta_init=kernel_delta_init,
                lambda_mfp_init=kernel_lambda_mfp_init,
            )
        elif kernel_type == "bubble":
            k_kwargs = dict(R_init=kernel_R_init, delta_init=kernel_delta_init)
        elif kernel_type == "mfp":
            k_kwargs = dict(lambda_mfp_init=kernel_lambda_mfp_init)
        self.kernel = KERNEL_REGISTRY[kernel_type](**k_kwargs)

        # --- Unresolved source field ---
        self.unresolved = UnresolvedSourceField(
            n_populations=n_populations,
            lf_prior=lf_prior,
        )

        # --- Excursion-set mapping ---
        self.excursion = ExcursionSetMapping(
            threshold_init=threshold_init,
            sharpness_init=sharpness_init,
            learnable=excursion_learnable,
        )

    def forward(
        self,
        graph: Data,
        density_basis: torch.Tensor,    # (P, G, G, G)
        xi_global: float | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        graph : PyG Data with fields:
            x           (N, F_in) node features
            edge_index  (2, E)
            edge_attr   (E, 4)
            pos         (N, 3) positions in [0, 1]
            src_weights (N,) raw source weights L_i normalised
        density_basis : (P, G, G, G) density bias basis fields
        xi_global     : float, target global ionized fraction for calibration
        return_intermediates : bool, whether to return J_obs, J_unres, J_total

        Returns
        -------
        dict with:
            x_hii_pred    (G, G, G)
            F_b           (P,)
            f_esc         (N,)
            theta_K       dict of kernel parameter values
            [optionally] J_obs, J_unres, J_total
        """
        device = graph.x.device
        G = self.grid_size

        # 1. GNN: node features → environment embeddings
        h = self.gnn(graph.x, graph.edge_index, graph.edge_attr)   # (N, d)

        # 2. Source head: h → f_esc, w_eff
        w_eff, src_info = self.source_head.compute_source_weights(
            h, graph.src_weights
        )   # (N,)

        # 3. Build 3D kernel on grid (uses current learnable parameters)
        kernel_grid = make_3d_kernel_grid(self.kernel, G, self.box_size, device)

        # 4. Scatter + convolve: LAE sources → J_obs(x)
        J_obs = scatter_and_convolve(
            pos_norm=graph.pos,        # (N, 3) in [0, 1]
            weights=w_eff,
            kernel_grid=kernel_grid,
            grid_size=G,
        )   # (G, G, G)

        # 5. Unresolved source emissivity: ε_unres(x; F_b)
        J_unres = self.unresolved(density_basis)   # (G, G, G)

        # 6. Total ionizing flux
        J_total = J_obs + J_unres   # (G, G, G)

        # 7. Soft excursion-set → x_HII
        x_hii_pred = self.excursion(J_total, xi_global=xi_global)   # (G, G, G)

        out = {
            "x_hii_pred": x_hii_pred,
            "F_b":        self.unresolved.F_b,
            "f_esc":      src_info["f_esc"],
            "theta_K":    self.kernel.get_params_dict() if hasattr(self.kernel, "get_params_dict")
                          else {},
        }

        if return_intermediates:
            out.update({
                "J_obs":        J_obs,
                "J_unres":      J_unres,
                "J_total":      J_total,
                "kernel_grid":  kernel_grid,
            })

        return out

    def get_learnable_physics(self) -> dict:
        """Return current values of all learnable physical parameters."""
        params = {}
        if hasattr(self.kernel, "get_params_dict"):
            params.update(self.kernel.get_params_dict())
        params.update(self.unresolved.get_fractions())
        params["threshold"] = float(self.excursion.threshold.item())
        params["sharpness"] = float(self.excursion.sharpness.item())
        return params


# ------------------------------------------------------------------ #
#  Ablation variants
# ------------------------------------------------------------------ #

def build_pinn_from_config(cfg: dict) -> LAEPINN:
    """Build LAEPINN from a config dict (from default.yaml)."""
    gnn_cfg    = cfg.get("gnn", {})
    src_cfg    = cfg.get("source_head", {})
    ker_cfg    = cfg.get("kernel", {})
    unr_cfg    = cfg.get("unresolved_sources", {})
    exc_cfg    = cfg.get("excursion_set", {})
    data_cfg   = cfg.get("data", {})

    return LAEPINN(
        gnn_architecture=gnn_cfg.get("architecture", "GATv2Conv"),
        gnn_in_channels=len(cfg.get("graph", {}).get("node_features", range(8))),
        gnn_hidden_dim=gnn_cfg.get("hidden_dim", 64),
        gnn_out_channels=gnn_cfg.get("output_dim", 32),
        gnn_n_layers=gnn_cfg.get("n_layers", 3),
        gnn_heads=gnn_cfg.get("heads", 4),
        gnn_dropout=gnn_cfg.get("dropout", 0.1),
        source_hidden_dims=src_cfg.get("f_esc_hidden", [32, 16]),
        xi_ion_learnable=src_cfg.get("xi_ion_learnable", False),
        xi_ion_log=src_cfg.get("xi_ion_log", 25.2),
        kernel_type=ker_cfg.get("type", "mixture"),
        kernel_R_init=ker_cfg.get("R_init_mpc", 5.0),
        kernel_delta_init=ker_cfg.get("delta_init_mpc", 1.0),
        kernel_lambda_mfp_init=ker_cfg.get("lambda_mfp_init_mpc", 20.0),
        n_populations=unr_cfg.get("n_populations", 3),
        threshold_init=exc_cfg.get("threshold_init", 0.0),
        sharpness_init=exc_cfg.get("sharpness_init", 1.0),
        grid_size=data_cfg.get("grid_mvp", 64),
        box_size=data_cfg.get("box_size_mpc", 160.0),
    )
