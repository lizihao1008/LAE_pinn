"""
models/pinn.py
Full PINN: LAE-conditioned photon-budget and ionization-topology inference.

Forward pass:
    LAE graph  →  GNN encoder  →  source head  →  f_esc_i, w_i
               →  scatter w_i onto 3D grid  →  S_obs(x)
               →  S_unres(x) = Σ_b f_esc_b · ε_b(x)   [HOD basis, fixed]
               →  J_total(x) = (S_obs + S_unres) * K(r; θ_K)   [one FFT]
               →  ionization equilibrium  A = J/s,  x_HII = (√(A²+4A)-A)/2
               →  x̂_HII(x)

Outputs:
    x_hii_pred :  (G, G, G) predicted ionization field
    f_esc_bins :  (n_hod_bins,) HOD escape fractions per mass bin
    f_esc      :  (N,) per-LAE escape fractions
    theta_K    :  dict of learnable kernel parameters
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.data import Data

try:
    from ..physics.kernels import MixtureKernel, build_kernel, make_3d_kernel_grid
    from ..physics.scatter import scatter_to_grid, fft_convolve_3d
    from ..physics.unresolved_sources import HODUnresolvedField
    from ..physics.excursion_set import ExcursionSetMapping
except ImportError:  # python -m experiments.run_mvp with project root on sys.path
    from physics.kernels import MixtureKernel, build_kernel, make_3d_kernel_grid
    from physics.scatter import scatter_to_grid, fft_convolve_3d
    from physics.unresolved_sources import HODUnresolvedField
    from physics.excursion_set import ExcursionSetMapping
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
        # RSD correction
        learn_rsd_correction: bool = False,
        rsd_max_shift_mpc: float = 2.0,    # cMpc/h, ~1 voxel at 64³
        los_axis: int = 2,                  # which axis is line-of-sight (0/1/2)
        # Kernel
        kernel_type: str = "mixture",
        kernel_R_init: float = 5.0,
        kernel_delta_init: float = 1.0,
        kernel_lambda_mfp_init: float = 20.0,
        # Unresolved sources (HOD-based)
        n_hod_bins: int = 3,
        # Ionization equilibrium  A = J / alpha_nH_scale
        alpha_nH_scale_init: float = 1.0,
        excursion_learnable: bool = True,
        # Grid
        grid_size: int = 64,
        box_size: float = 160.0,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.box_size  = box_size
        self.los_axis  = los_axis

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
            learn_rsd_correction=learn_rsd_correction,
            rsd_max_shift_mpc=rsd_max_shift_mpc,
            box_size=box_size,
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
        try:
            from ..physics.kernels import KERNEL_REGISTRY
        except ImportError:
            from physics.kernels import KERNEL_REGISTRY
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

        # --- Unresolved source field (HOD-based) ---
        # Basis fields are fixed from HOD calibration (pre-computed before training).
        # Only f_esc per mass bin is learned.
        self.unresolved = HODUnresolvedField(n_bins=n_hod_bins)

        # --- Ionization equilibrium ---
        # Photoionization balance Γ(1−x)n_H = α n_e n_p  →  x_HII(J) via A=J/s.
        # Single learnable s absorbs recombination coeff α and mean density n_H
        # (neither is directly observable at z~7).
        self.excursion = ExcursionSetMapping(
            alpha_nH_scale_init=alpha_nH_scale_init,
            learnable=excursion_learnable,
        )

    def forward(
        self,
        graph: Data,
        hod_basis: torch.Tensor,        # (n_bins, G, G, G)
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
            src_weights (N,) raw source weights (log10 L_Lyα normalised)
        hod_basis         : (n_bins, G, G, G) HOD basis fields, fixed (pre-computed)
        xi_global         : float, not used in forward (kept for API compat)
        return_intermediates : bool, whether to return J_obs, J_unres, J_total

        Returns
        -------
        dict with:
            x_hii_pred    (G, G, G)
            f_esc_bins    (n_bins,) learned escape fractions
            f_esc         (N,) per-LAE escape fractions
            theta_K       dict of kernel parameter values
            J_scale       float, mean(J_total) used for normalisation
            [optionally] J_obs, J_unres, J_total, kernel_grid
        """
        device = graph.x.device
        G = self.grid_size

        # 1. GNN: node features → environment embeddings
        h = self.gnn(graph.x, graph.edge_index, graph.edge_attr)   # (N, d)

        # 2. Source head: h → f_esc, w_eff (and optionally Δr∥)
        w_eff, src_info = self.source_head.compute_source_weights(
            h, graph.src_weights
        )   # (N,)

        # 2b. Learned RSD correction: shift scatter positions along LOS
        #     pos is in [0,1]; delta_los is also in [0,1] (normalised by box_size).
        #     The trilinear scatter is differentiable w.r.t. the fractional part of
        #     the position, so gradient flows back to delta_los_mlp for sub-voxel
        #     shifts (typical RSD amplitude ~0.5–1 voxel at 64³, z~7).
        if "delta_los" in src_info:
            delta_los = src_info["delta_los"]   # (N,) in normalised [0,1] space
            z = torch.zeros_like(delta_los)
            if self.los_axis == 0:
                delta_3d = torch.stack([delta_los, z, z], dim=-1)
            elif self.los_axis == 1:
                delta_3d = torch.stack([z, delta_los, z], dim=-1)
            else:  # los_axis == 2 (default)
                delta_3d = torch.stack([z, z, delta_los], dim=-1)
            # graph.pos has no grad; delta_3d carries grad through delta_los
            pos_scatter = graph.pos + delta_3d   # (N, 3), grad flows via delta_3d
            pos_scatter = pos_scatter % 1.0       # periodic wrap (differentiable away from boundaries)
        else:
            pos_scatter = graph.pos

        # 3. Build 3D kernel on grid (uses current learnable parameters)
        kernel_grid = make_3d_kernel_grid(self.kernel, G, self.box_size, device)

        # 4. Scatter LAE sources onto grid — source density, NOT radiation field yet.
        #    S_obs(x) = Σ_i w_i δ³(x - x̃_i)
        S_obs = scatter_to_grid(pos_scatter, w_eff, G)   # (G, G, G)

        # 5. Unresolved source density: S_unres(x) = Σ_b f_esc_b · ε_b(x)
        #    ε_b(x) = 1 + b_b δ_dm(x) is the fixed HOD basis; only f_esc_b learned.
        S_unres = self.unresolved(hod_basis)              # (G, G, G)

        # 6. Amplitude-balance S_obs before combining with S_unres.
        #
        #    HOD basis fields ε_b are normalised to unit mean in preprocessing,
        #    so  mean(S_unres) = n_bins × <f_esc> ≈ n_bins × 0.5 = O(1–5).
        #    S_obs amplitude = N_obs × <w_eff> / G³:  with N_obs ~ few thousand and
        #    G=64,  mean(S_obs) ~ O(1e-3).  The ~1000× imbalance lets S_unres
        #    dominate J_total completely, washing out all spatial structure from
        #    the observed LAE scatter regardless of what the GNN learns.
        #
        #    Fix: normalise S_obs to unit mean (same convention as ε_b) before
        #    the convolution.  This is safe because J_total is re-scaled by its
        #    own spatial mean in step 7 anyway — only the spatial *contrast* of
        #    J matters for x_pred, not its absolute level.
        #
        #    After this normalisation, S_obs contributes mean=1 and S_unres
        #    contributes mean = n_bins × f_esc  (~0–9).  As training proceeds,
        #    f_esc drives to small values, balancing the two terms.
        S_obs_scale = S_obs.mean().detach().clamp(min=1e-12)
        S_obs_normed = S_obs / S_obs_scale   # unit mean; spatial structure preserved

        # 6b. Convolve TOTAL source density with radiative kernel K(r; θ_K).
        #     J_total(x) = (S_obs_normed + S_unres) * K
        J_total = fft_convolve_3d(S_obs_normed + S_unres, kernel_grid)   # (G, G, G)

        # 7. J → x_HII
        # Normalise J by its spatial mean before the excursion mapping.
        # Without this, J values (~1e-3) are orders of magnitude below
        # alpha_nH_scale (~1.0), giving A = J/s << 1 and x_pred ≈ 0 everywhere
        # regardless of the spatial structure.  After normalisation:
        #   mean(J_norm) = 1  →  A ~ 1/s  ~  1 at init  →  x_pred ~ 0.6
        # which is already near xi_global, so alpha_nH_scale only needs to
        # fine-tune the transition sharpness via the global_xHII loss.
        # J_scale is detached so the normalisation constant doesn't zero out
        # the spatial gradients of J_total.
        J_scale    = J_total.mean().detach().clamp(min=1e-12)
        x_hii_pred = self.excursion(J_total / J_scale)

        out = {
            "x_hii_pred":  x_hii_pred,
            "f_esc_bins":  self.unresolved.f_esc_bins,   # (n_bins,) HOD escape fractions
            "f_esc":       src_info["f_esc"],             # (N,) per-LAE escape fractions
            "theta_K":     self.kernel.get_params_dict() if hasattr(self.kernel, "get_params_dict")
                           else {},
        }

        # Expose RSD correction stats for logging/diagnostics
        if "delta_los" in src_info:
            dl = src_info["delta_los"].detach()
            box = self.box_size
            out["delta_los_mean_mpc"] = float(dl.mean().item() * box)
            out["delta_los_std_mpc"]  = float(dl.std().item()  * box)

        # Always expose scales so the training loop can log amplitude diagnostics
        out["J_scale"]     = float(J_scale.item())
        out["S_obs_scale"] = float(S_obs_scale.item())   # raw S_obs mean (pre-normalisation)

        if return_intermediates:
            out.update({
                "S_obs":       S_obs,         # raw scatter grid (pre-normalisation)
                "S_obs_normed":S_obs_normed,  # unit-mean scatter grid (fed into convolution)
                "S_unres":     S_unres,       # unresolved source density (pre-kernel)
                "J_total":     J_total,       # (S_obs_normed + S_unres) * K
                "kernel_grid": kernel_grid,
                # backward-compat aliases used by loss.py (TV prior on J_unres)
                "J_obs":       S_obs_normed,
                "J_unres":     S_unres,
            })

        return out

    def get_learnable_physics(self) -> dict:
        """Return current values of all learnable physical parameters."""
        params = {}
        if hasattr(self.kernel, "get_params_dict"):
            params.update(self.kernel.get_params_dict())
        # f_esc per HOD mass bin (keys: "fesc_bin0", "fesc_bin1", ...)
        params.update(self.unresolved.get_fesc_dict())
        params["alpha_nH_scale"] = float(self.excursion.alpha_nH_scale.item())
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

    rsd_cfg = cfg.get("rsd_correction", {})
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
        learn_rsd_correction=rsd_cfg.get("enabled", False),
        rsd_max_shift_mpc=rsd_cfg.get("max_shift_mpc", 2.0),
        los_axis=rsd_cfg.get("los_axis", 2),
        kernel_type=ker_cfg.get("type", "mixture"),
        kernel_R_init=ker_cfg.get("R_init_mpc", 5.0),
        kernel_delta_init=ker_cfg.get("delta_init_mpc", 1.0),
        kernel_lambda_mfp_init=ker_cfg.get("lambda_mfp_init_mpc", 20.0),
        n_hod_bins=unr_cfg.get("n_hod_bins", unr_cfg.get("n_populations", 3)),
        alpha_nH_scale_init=exc_cfg.get(
            "alpha_nH_scale_init",
            exc_cfg.get("alpha_scale_init", exc_cfg.get("sharpness_init", 1.0)),
        ),
        grid_size=data_cfg.get("grid_mvp", 64),
        box_size=data_cfg.get("box_size_mpc", 160.0),
    )
