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
    from ..physics.excursion_set import (
        ExcursionSetMapping, BubbleExcursionSet, HybridBubbleEquilibrium,
    )
    from ..physics.continuous_field import (
        render_obs_field_on_grid, render_bubble_on_grid, calibrate_bubble_zeta,
    )
except ImportError:  # python -m experiments.run_mvp with project root on sys.path
    from physics.kernels import MixtureKernel, build_kernel, make_3d_kernel_grid
    from physics.scatter import scatter_to_grid, fft_convolve_3d
    from physics.unresolved_sources import HODUnresolvedField
    from physics.excursion_set import (
        ExcursionSetMapping, BubbleExcursionSet, HybridBubbleEquilibrium,
    )
    from physics.continuous_field import (
        render_obs_field_on_grid, render_bubble_on_grid, calibrate_bubble_zeta,
    )
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
        # Ionization mapping  J -> x_HII
        #   "equilibrium" : smooth photoionization equilibrium x(A), A=J/alpha_nH_scale
        #   "bubble"      : excursion-set bubble/threshold (sharp 0/1 topology,
        #                   x_HII -> 0 in voids; needed to reach near-zero x_HII)
        excursion_type: str = "equilibrium",
        alpha_nH_scale_init: float = 1.0,
        excursion_learnable: bool = True,
        bubble_zeta_init: float = 1.0,
        bubble_sharpness_init: float = 6.0,
        bubble_radii_mpc: list[float] | None = None,
        # Hybrid (bubble_equilibrium): residual photoionization-equilibrium scale s.
        # Small so x_eq ~ 1 inside bubbles (hybrid starts ~ pure bubble, then refines).
        hybrid_residual_scale_init: float = 0.1,
        # Grid
        grid_size: int = 64,
        box_size: float = 160.0,
        # Field generator: how the observed-source radiation field J_obs is built.
        #   "continuous" : kernel-sum  J_obs(x)=Σ_i w_i K(|x-x_i|) rendered on the
        #                  grid with EXACT source positions (no trilinear voxel
        #                  smearing); also queryable at arbitrary coordinates.
        #   "grid"       : legacy trilinear scatter + FFT convolution.
        # Identical on-grid for on-grid sources; A_obs/J_ref calibration unchanged
        # (the unit-sum kernel preserves spatial means).  Supports BOTH cores: the
        # bubble core uses per-scale top-hat kernel sums for the observed sources
        # (mesh-free) while S_unres / n_H are grid-smoothed and interpolated.
        field_generator: str = "continuous",
        continuous_r_cut_mpc: float | None = None,
        continuous_chunk: int = 4096,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.box_size  = box_size
        self.los_axis  = los_axis
        self.field_generator = field_generator
        self.cf_r_cut = continuous_r_cut_mpc
        self.cf_chunk = continuous_chunk

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

        # --- Learnable S_obs amplitude factor ---
        # S_obs (sparse observed LAEs) has mean ≈ N_obs/G³ ≈ 0.03, while
        # S_unres (HOD background) has mean ≈ n_bins × f_esc ≈ 4.4.
        # Without rebalancing, S_obs contributes only ~0.7% to J_total and
        # LAE clustering is completely swamped by the HOD background.
        # A_obs is a learnable amplitude that scales S_obs so it contributes
        # meaningfully to the ionization structure.  Initialised to 0 in log
        # space, which gives equi-amplitude (mean(A_obs × S_obs) = mean(S_unres))
        # via the dynamic rescaling in forward().  The model then learns
        # to up- or down-weight the LAE term depending on its correlation with
        # the target x_HII field.
        self._log_A_obs = nn.Parameter(torch.tensor(0.0))

        # --- Amplitude reference constants (calibrated ONCE, then fixed) ---
        # The previous implementation recomputed two DETACHED means every step
        #   A_obs_eq = mean(S_unres)/mean(S_obs)   and   J_scale = mean(J_total)
        # which made the forward output exactly invariant to the overall
        # amplitude of the source field.  As a result the photon budget
        # (f_esc, source weights) could not change <x_HII>, the field was
        # locked in the low-A regime (no spatial contrast), and the detach
        # created a spurious common-mode gradient that dragged every f_esc bin
        # to the same value.  We instead calibrate these constants on the FIRST
        # forward pass and hold them fixed, so amplitude changes propagate to
        # x_HII while the sane init (mean(J/J_ref)=1) is preserved.
        self.register_buffer("_amp_calibrated", torch.tensor(False))
        self.register_buffer("_A_obs_base", torch.tensor(1.0))
        self.register_buffer("_J_ref", torch.tensor(1.0))

        # --- Ionization mapping  J -> x_HII ---
        self.excursion_type = excursion_type
        if excursion_type == "bubble":
            # Excursion-set bubble model: a cell is ionized if the smoothed
            # photon/H ratio exceeds 1 on ANY scale.  Produces sharp 0/1 topology
            # and genuine neutral islands (x_HII -> 0), unlike the smooth
            # equilibrium which floors x_HII well above 0.
            self.excursion = BubbleExcursionSet(
                grid_size=grid_size,
                box_size=box_size,
                radii_mpc=bubble_radii_mpc,
                zeta_init=bubble_zeta_init,
                sharpness_init=bubble_sharpness_init,
                learnable=excursion_learnable,
            )
        elif excursion_type == "bubble_equilibrium":
            # Hybrid: x_HII = B(x) · x_eq(x).  Bubble sets the topology (sharp 0/1,
            # real voids); photoionization equilibrium sets the interior residual.
            self.excursion = HybridBubbleEquilibrium(
                grid_size=grid_size,
                box_size=box_size,
                radii_mpc=bubble_radii_mpc,
                zeta_init=bubble_zeta_init,
                sharpness_init=bubble_sharpness_init,
                residual_scale_init=hybrid_residual_scale_init,
                learnable=excursion_learnable,
            )
        else:
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
        density_grid: torch.Tensor | None = None,   # (G,G,G) gas density ~ (1+δ), for bubble core
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

        # 3. Build 3D kernel on grid (uses current learnable parameters).
        #    Still needed for the unresolved (diffuse) FFT convolution, and for the
        #    legacy grid generator.
        kernel_grid = make_3d_kernel_grid(self.kernel, G, self.box_size, device)

        # 4. Unresolved source density: S_unres(x) = Σ_b f_esc_b · ε_b(x)
        #    ε_b(x) = 1 + b_b δ_dm(x) is the fixed HOD basis; only f_esc_b learned.
        S_unres = self.unresolved(hod_basis)              # (G, G, G)

        # A_obs balances the sparse LAE term against the HOD background.  Its base
        # is calibrated ONCE (mean(S_unres)/mean(S_obs) on the first batch) then
        # held fixed; _log_A_obs is the learnable deviation.  The base is FIXED,
        # not a per-step detached ratio (which would cancel the photon-budget
        # signal and collapse all f_esc bins via a spurious common-mode gradient).

        if self.excursion_type == "bubble_equilibrium":
            # ===== Hybrid core: x_HII = B(x) · x_eq(x) =====
            #   B    : excursion-set bubble membership (topology, sharp 0/1, voids)
            #   x_eq : photoionization-equilibrium ionized fraction on the propagated
            #          field J (interior residual, x_HI ~ s/J).
            # Built with the active field generator; A_obs calibrated on emissivity
            # means, J_ref on mean(J), zeta on <B·x_eq> = xi (x_eq fixed).
            if self.field_generator == "continuous":
                J_obs_grid   = render_obs_field_on_grid(
                    self.kernel, pos_scatter, w_eff, G, self.box_size,
                    r_cut_mpc=self.cf_r_cut, chunk=self.cf_chunk)
                J_unres_grid = fft_convolve_3d(S_unres, kernel_grid)
                if not bool(self._amp_calibrated):
                    with torch.no_grad():
                        s_obs_m = (w_eff.sum() / float(G ** 3)).clamp(min=1e-12)
                        s_unr_m = S_unres.mean().clamp(min=1e-12)
                        self._A_obs_base.copy_(s_unr_m / s_obs_m)
                A_obs   = self._A_obs_base * torch.exp(self._log_A_obs)
                J_total = A_obs * J_obs_grid + J_unres_grid       # propagated field
                if not bool(self._amp_calibrated):
                    with torch.no_grad():
                        self._J_ref.copy_(J_total.mean().clamp(min=1e-12))
                        # zeta calib uses the (cheap) grid bubble on scattered emissivity
                        S_emiss_c = A_obs * scatter_to_grid(pos_scatter, w_eff, G) + S_unres
                        tgt = float(getattr(graph, "xi_global", 0.5))
                        self.excursion.calibrate_zeta(
                            S_emiss_c, J_total / self._J_ref,
                            density=density_grid, target=tgt)
                        self._amp_calibrated.fill_(True)
                J_ref = self._J_ref
                B     = render_bubble_on_grid(
                    self.excursion.bubble, pos_scatter, w_eff, A_obs, S_unres,
                    density_grid, self.box_size, grid_size=G, chunk=self.cf_chunk)
                x_eq  = self.excursion.equilibrium(J_total / J_ref)
                x_hii_pred  = (B * x_eq).clamp(0.0, 1.0)
                S_obs       = J_obs_grid                          # diagnostics
                S_obs_scale = (w_eff.sum() / float(G ** 3)).detach()
            else:
                S_obs   = scatter_to_grid(pos_scatter, w_eff, G)
                if not bool(self._amp_calibrated):
                    with torch.no_grad():
                        s_obs_m = S_obs.mean().clamp(min=1e-12)
                        s_unr_m = S_unres.mean().clamp(min=1e-12)
                        self._A_obs_base.copy_(s_unr_m / s_obs_m)
                A_obs   = self._A_obs_base * torch.exp(self._log_A_obs)
                S_emiss = A_obs * S_obs + S_unres
                J_total = fft_convolve_3d(S_emiss, kernel_grid)
                if not bool(self._amp_calibrated):
                    with torch.no_grad():
                        self._J_ref.copy_(J_total.mean().clamp(min=1e-12))
                        tgt = float(getattr(graph, "xi_global", 0.5))
                        self.excursion.calibrate_zeta(
                            S_emiss, J_total / self._J_ref,
                            density=density_grid, target=tgt)
                        self._amp_calibrated.fill_(True)
                J_ref = self._J_ref
                x_hii_pred  = self.excursion(S_emiss, J_total / J_ref, density=density_grid)
                S_obs_scale = S_obs.mean().detach()
        elif self.field_generator == "continuous" and self.excursion_type == "bubble":
            # ---- Continuous bubble generator (top-hat excursion-set, mesh-free obs) ----
            #   Observed sources enter each scale's top-hat smoothing as an EXACT
            #   kernel sum (no scatter voxel smearing); S_unres and n_H are smoothed
            #   on the grid (reusing the bubble's top-hat buffers) and interpolated.
            #   A_obs uses emissivity means; the observed mean is analytic
            #   (Σ w_eff / G³ == scatter S_obs.mean()), so calibration matches the
            #   grid bubble path without scattering.
            if not bool(self._amp_calibrated):
                with torch.no_grad():
                    s_obs_m = (w_eff.sum() / float(G ** 3)).clamp(min=1e-12)
                    s_unr_m = S_unres.mean().clamp(min=1e-12)
                    self._A_obs_base.copy_(s_unr_m / s_obs_m)
            A_obs = self._A_obs_base * torch.exp(self._log_A_obs)
            if not bool(self._amp_calibrated):
                with torch.no_grad():
                    tgt = float(getattr(graph, "xi_global", 0.5))
                    calibrate_bubble_zeta(
                        self.excursion, pos_scatter, w_eff, A_obs, S_unres,
                        density_grid, self.box_size, target=tgt,
                    )
                    # diagnostic emissivity scale (bubble x_HII does not use J_ref)
                    self._J_ref.copy_((w_eff.sum() / float(G ** 3)).clamp(min=1e-12))
                    self._amp_calibrated.fill_(True)
            x_hii_pred = render_bubble_on_grid(
                self.excursion, pos_scatter, w_eff, A_obs, S_unres,
                density_grid, self.box_size, grid_size=G, chunk=self.cf_chunk,
            )
            J_ref = self._J_ref
            # diagnostics only (NOT used for x_HII): emissivity grid via cheap scatter
            S_obs = scatter_to_grid(pos_scatter, w_eff, G)
            J_total = A_obs * S_obs + S_unres
            S_obs_scale = (w_eff.sum() / float(G ** 3)).detach()
        elif self.field_generator == "continuous":
            # ---- Continuous generator (equilibrium): J_obs(x)=Σ_i w_i K(|x-x_i|) ----
            #   Drop-in for scatter_to_grid + fft_convolve_3d of the observed term,
            #   but with EXACT source positions (no trilinear voxel smearing).  The
            #   unit-sum kernel preserves means, so mean(J_obs_grid)=mean(S_obs)
            #   and mean(J_unres_grid)=mean(S_unres): the A_obs / J_ref calibration
            #   is bit-for-bit the same as the grid path.  By linearity
            #   J_total = A_obs·(S_obs*K) + (S_unres*K) = A_obs·J_obs_grid + J_unres_grid.
            J_obs_grid   = render_obs_field_on_grid(
                self.kernel, pos_scatter, w_eff, G, self.box_size,
                r_cut_mpc=self.cf_r_cut, chunk=self.cf_chunk,
            )                                              # (G,G,G) observed radiation field
            J_unres_grid = fft_convolve_3d(S_unres, kernel_grid)
            if not bool(self._amp_calibrated):
                with torch.no_grad():
                    self._A_obs_base.copy_(
                        J_unres_grid.mean().clamp(min=1e-12)
                        / J_obs_grid.mean().clamp(min=1e-12)
                    )
            A_obs   = self._A_obs_base * torch.exp(self._log_A_obs)
            J_total = A_obs * J_obs_grid + J_unres_grid
            if not bool(self._amp_calibrated):
                with torch.no_grad():
                    self._J_ref.copy_(J_total.mean().clamp(min=1e-12))
                    self._amp_calibrated.fill_(True)
            J_ref       = self._J_ref
            x_hii_pred  = self.excursion(J_total / J_ref)
            S_obs       = J_obs_grid                       # obs radiation field (diagnostics)
            S_obs_scale = J_obs_grid.mean().detach()
        else:
            # ---- Legacy grid generator (trilinear scatter + FFT) / bubble path ----
            S_obs = scatter_to_grid(pos_scatter, w_eff, G)   # (G, G, G) source density
            if not bool(self._amp_calibrated):
                with torch.no_grad():
                    s_obs_m = S_obs.mean().clamp(min=1e-12)
                    s_unr_m = S_unres.mean().clamp(min=1e-12)
                    self._A_obs_base.copy_(s_unr_m / s_obs_m)
            A_obs   = self._A_obs_base * torch.exp(self._log_A_obs)
            S_emiss = A_obs * S_obs + S_unres        # total emissivity (pre-propagation)

            if self.excursion_type == "bubble":
                # Bubble model does its OWN multi-scale smoothing; the mixture
                # kernel is bypassed.  Sharp 0/1 topology with genuine voids.
                J_total = S_emiss
                if not bool(self._amp_calibrated):
                    with torch.no_grad():
                        self._J_ref.copy_(S_emiss.mean().clamp(min=1e-12))
                        tgt = float(getattr(graph, "xi_global", 0.5))
                        self.excursion.calibrate_zeta(S_emiss, density_grid, target=tgt)
                        self._amp_calibrated.fill_(True)
                x_hii_pred = self.excursion(S_emiss, density=density_grid)
                J_ref = self._J_ref
            else:
                J_total = fft_convolve_3d(S_emiss, kernel_grid)   # (G, G, G)
                if not bool(self._amp_calibrated):
                    with torch.no_grad():
                        self._J_ref.copy_(J_total.mean().clamp(min=1e-12))
                        self._amp_calibrated.fill_(True)
                J_ref      = self._J_ref
                x_hii_pred = self.excursion(J_total / J_ref)

            S_obs_scale = S_obs.mean().detach()   # for diagnostics only

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
        out["J_scale"]     = float(J_ref.item())          # fixed reference (calibrated once)
        out["S_obs_scale"] = float(S_obs_scale.item())    # raw S_obs mean (per-batch)
        out["A_obs"]       = float(A_obs.detach().item())  # current S_obs amplitude factor

        if return_intermediates:
            out.update({
                "S_obs":       S_obs,       # scatter grid of LAEs  (pre-kernel)
                "S_unres":     S_unres,     # unresolved source density (pre-kernel)
                "J_total":     J_total,     # (S_obs + S_unres) * K  (radiation field)
                "kernel_grid": kernel_grid,
                # backward-compat aliases used by loss.py (TV prior on J_unres)
                "J_obs":       S_obs,
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
        # ionization-mapping parameters
        #   equilibrium        : alpha_nH_scale
        #   bubble             : zeta, sharpness
        #   bubble_equilibrium : zeta, sharpness, alpha_nH_scale (residual)
        if self.excursion_type in ("bubble", "bubble_equilibrium"):
            params.update(self.excursion.get_params_dict())
        else:
            params["alpha_nH_scale"] = float(self.excursion.alpha_nH_scale.item())
        # log_A_obs: learnable log-deviation from equi-amplitude for S_obs vs S_unres
        params["log_A_obs"] = float(self._log_A_obs.item())
        return params

    def continuous_field(self, graph: Data, hod_basis: torch.Tensor,
                         use_rsd_correction: bool = True):
        """
        Off-grid interface: returns ``(evaluator, ctx)`` so that
        ``evaluator.forward(query, **ctx)["x_hii"]`` evaluates x_HII at ARBITRARY
        coordinates (e.g. exact LAE positions for MCF marks, sightlines, or a
        higher-resolution grid), consistent with ``forward`` on the grid.

        Supports BOTH cores: equilibrium (kernel-integral field) and bubble
        (top-hat excursion-set).  See
        ``physics.continuous_field.build_continuous_field_from_pinn``.
        """
        try:
            from ..physics.continuous_field import build_continuous_field_from_pinn
        except ImportError:
            from physics.continuous_field import build_continuous_field_from_pinn
        return build_continuous_field_from_pinn(self, graph, hod_basis,
                                                use_rsd_correction=use_rsd_correction)


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
    cf_cfg  = cfg.get("continuous_field", {})
    field_generator = cf_cfg.get("generator", cfg.get("field_generator", "continuous"))
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
        excursion_type=exc_cfg.get("type", "equilibrium"),
        alpha_nH_scale_init=exc_cfg.get(
            "alpha_nH_scale_init",
            exc_cfg.get("alpha_scale_init", exc_cfg.get("sharpness_init", 1.0)),
        ),
        bubble_zeta_init=exc_cfg.get("zeta_init", 1.0),
        bubble_sharpness_init=exc_cfg.get("bubble_sharpness_init", 6.0),
        bubble_radii_mpc=exc_cfg.get("bubble_radii_mpc", None),
        hybrid_residual_scale_init=exc_cfg.get("hybrid_residual_scale_init", 0.1),
        grid_size=data_cfg.get("grid_mvp", 64),
        box_size=data_cfg.get("box_size_mpc", 160.0),
        field_generator=field_generator,
        continuous_r_cut_mpc=cf_cfg.get("r_cut_mpc", None),
        continuous_chunk=cf_cfg.get("chunk", 4096),
    )
