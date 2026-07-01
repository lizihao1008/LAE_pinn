"""
physics/los_transmission.py
Line-of-sight Lyα transmission module.

Physical motivation
-------------------
Lyα visibility is NOT the local ionization state of the LAE's voxel.  Observed
Lyα flux / EW / T_IGM is a LINE-OF-SIGHT effect: photons emitted near line centre
are attenuated by neutral hydrogen encountered along the path to the observer,

    T_IGM(nu) = exp[-tau_IGM(nu)],   tau_IGM = INT n_HI(s) sigma_alpha[nu(s)] ds,

with the dominant contribution from the near-side bubble wall, neutral islands,
self-shielded absorbers, and the damping wing.  An LAE deep inside an ionized
bubble can still be invisible if dense neutral gas sits in front of it; a large
LOS bubble radius lets Hubble expansion redshift photons off resonance before
they meet neutral gas, making the source visible.

Role separation in the model
----------------------------
1. Spatial clustering / galaxy positions  -> 3D ionization topology (field loss).
2. Lyα marks (T_IGM, EW, L_Lyα)           -> LOS transmission toward the observer
                                             (this module + its auxiliary loss).

So T_IGM / EW / L_Lyα are NOT used as local x_HII labels at the source voxel.
Instead this module casts a ray from each LAE along the observer direction through
the PREDICTED ionization field and integrates a simplified damping-wing optical
depth.

Effective optical depth
-----------------------
    x_HI_hat(s) = 1 - x_HII_hat(x_i + s e_LOS)          (sampled along the ray)
    tau_i       = INT_0^Rmax x_HI_hat(s) W(s; dv_i) ds
    W(s)        = amp / (dv_i + H(z) s + v_floor)^2      (damping-wing detuning)
    T_IGM_hat,i = exp[-(tau_i + tau_CGM)]

`dv_i` is the intrinsic Lyα velocity offset (km/s); if unavailable a learnable
population value `dv_Lya` is used.  `tau_CGM` is an optional learnable local
(circum-galactic) attenuation.  `H(z)` is the Hubble velocity per comoving Mpc/h.

The ray is sampled by periodic trilinear interpolation of the predicted grid
field, so gradients flow to the whole ionization model AND to (dv_Lya, tau_CGM,
amp).  Everything is differentiable and mesh-consistent with the periodic box.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .continuous_field import periodic_trilinear_sample
except ImportError:  # pragma: no cover
    from continuous_field import periodic_trilinear_sample


def _softplus_inverse(v: float) -> float:
    """Numerically stable inverse softplus: raw s.t. softplus(raw) = v (v > 0)."""
    v = float(v)
    if v > 20.0:
        return v                                   # softplus(v) ~ v to < 1e-8
    return v + math.log(-math.expm1(-v))            # v + log(1 - e^{-v})


def hubble_kms_per_cmpc(redshift: float, omega_m: float = 0.3089) -> float:
    """
    Hubble velocity per comoving Mpc/h at ``redshift`` (km/s per cMpc/h).

        dv = H(z) * dr_phys,   dr_phys = dr_comoving / ((1+z) h)
           = 100 * sqrt(Om(1+z)^3 + OL) / (1+z)  [km/s]  per cMpc/h      (h cancels)
    """
    E = math.sqrt(omega_m * (1.0 + redshift) ** 3 + (1.0 - omega_m))
    return 100.0 * E / (1.0 + redshift)


class LOSTransmission(nn.Module):
    """
    Line-of-sight Lyα transmission T_IGM_hat for each LAE from the predicted field.

    Parameters
    ----------
    box_size : float           periodic box in cMpc/h
    los_axis : int             observer line-of-sight axis (0/1/2)
    los_sign : +1 | -1         direction along the axis toward the observer
    r_max_mpc : float          ray length (cMpc/h), default 80
    n_steps : int              ray samples (midpoint rule)
    redshift : float           default z for H(z) if not supplied at forward()
    omega_m : float            matter density for H(z)
    hubble_kms_per_cmpc : float | None   override H(z); None -> from (z, omega_m)
    v_floor : float            km/s floor in the detuning kernel (stability)
    dv_lya_init, tau_cgm_init, amp_init : float   initial physical values
    learn_dv, learn_tau_cgm, learn_amp : bool     which are trained
    """

    def __init__(
        self,
        box_size: float = 160.0,
        los_axis: int = 2,
        los_sign: int = 1,
        r_max_mpc: float = 80.0,
        n_steps: int = 64,
        redshift: float = 7.0,
        omega_m: float = 0.3089,
        hubble_kms_per_cmpc: float | None = None,
        v_floor: float = 50.0,
        v_ref: float = 600.0,        # reference velocity (km/s) non-dimensionalising W
        dv_lya_init: float = 200.0,
        tau_cgm_init: float = 0.05,
        amp_init: float = 1.0,
        learn_dv: bool = True,
        learn_tau_cgm: bool = True,
        learn_amp: bool = True,
    ):
        super().__init__()
        self.box_size = float(box_size)
        self.los_axis = int(los_axis)
        self.los_sign = 1 if los_sign >= 0 else -1
        self.r_max_mpc = float(r_max_mpc)
        self.n_steps = int(n_steps)
        self.v_floor = float(v_floor)
        self.v_ref = float(v_ref)
        self.omega_m = float(omega_m)
        self._fixed_hubble = hubble_kms_per_cmpc
        self._default_z = float(redshift)

        # Ray sample distances (cMpc/h), midpoint rule on [0, r_max].
        edges = torch.linspace(0.0, self.r_max_mpc, self.n_steps + 1)
        s_mid = 0.5 * (edges[:-1] + edges[1:])          # (n_steps,)
        self.register_buffer("s_mid", s_mid)
        self.register_buffer("ds", torch.tensor(self.r_max_mpc / self.n_steps))

        # Learnable physical parameters (softplus -> positive).
        self._dv_raw = nn.Parameter(torch.tensor(_softplus_inverse(dv_lya_init)),
                                    requires_grad=learn_dv)
        self._tau_cgm_raw = nn.Parameter(torch.tensor(_softplus_inverse(tau_cgm_init)),
                                         requires_grad=learn_tau_cgm)
        self._amp_raw = nn.Parameter(torch.tensor(_softplus_inverse(amp_init)),
                                     requires_grad=learn_amp)

    # ---- positive physical parameters ----
    @property
    def dv_lya(self) -> torch.Tensor:
        return F.softplus(self._dv_raw) + 1e-3

    @property
    def tau_cgm(self) -> torch.Tensor:
        return F.softplus(self._tau_cgm_raw)

    @property
    def amp(self) -> torch.Tensor:
        return F.softplus(self._amp_raw)

    def _hubble(self, z: float | None) -> float:
        if self._fixed_hubble is not None:
            return float(self._fixed_hubble)
        return hubble_kms_per_cmpc(self._default_z if z is None else float(z), self.omega_m)

    def forward(
        self,
        x_hii_grid: torch.Tensor,          # (G, G, G) predicted ionization field
        pos: torch.Tensor,                 # (N, 3) source positions, normalised [0,1)
        dv_i: torch.Tensor | None = None,  # (N,) intrinsic Lyα offset [km/s], optional
        z: float | None = None,
        return_tau: bool = True,
    ):
        """
        Returns dict with:
            T_IGM_hat : (N,)  predicted LOS transmission in (0, 1]
            tau_los   : (N,)  effective optical depth (excl. tau_CGM)   [if return_tau]
        """
        device = pos.device
        dtype = pos.dtype
        N = pos.shape[0]
        L = self.box_size
        s_mid = self.s_mid.to(device=device, dtype=dtype)          # (n_steps,)
        H = self._hubble(z)                                        # km/s per cMpc/h

        # Ray sample coordinates: pos + sign * s * e_LOS  (normalised, periodic wrap).
        offset = torch.zeros(self.n_steps, 3, device=device, dtype=dtype)
        offset[:, self.los_axis] = self.los_sign * s_mid / L
        pts = (pos[:, None, :] + offset[None, :, :]) % 1.0         # (N, n_steps, 3)

        # Neutral fraction sampled along each ray (differentiable in the field).
        x_hii_ray = periodic_trilinear_sample(
            x_hii_grid, pts.reshape(-1, 3)).reshape(N, self.n_steps)
        x_hi_ray = (1.0 - x_hii_ray).clamp(0.0, 1.0)              # (N, n_steps)

        # Damping-wing / frequency-detuning weight W(s) = amp / (dv + H s + v_floor)^2.
        if dv_i is None:
            dv = self.dv_lya                                       # scalar (population)
        else:
            dv = dv_i.to(device=device, dtype=dtype)[:, None]      # (N, 1) per-source
        denom = dv + H * s_mid[None, :] + self.v_floor            # (1,n_steps) or (N,n_steps)
        # non-dimensionalise by v_ref so amp ~ O(1) gives tau ~ O(1)
        W = self.amp * (self.v_ref * self.v_ref) / (denom * denom)

        ds = self.ds.to(device=device, dtype=dtype)
        tau = (x_hi_ray * W).sum(dim=1) * ds                      # (N,)
        T_hat = torch.exp(-(tau + self.tau_cgm)).clamp(1e-8, 1.0)

        out = {"T_IGM_hat": T_hat}
        if return_tau:
            out["tau_los"] = tau
        return out

    def get_params_dict(self) -> dict:
        return {
            "dv_lya_kms": float(self.dv_lya.item()),
            "tau_cgm": float(self.tau_cgm.item()),
            "los_amp": float(self.amp.item()),
        }

    def extra_repr(self) -> str:
        return (f"los_axis={self.los_axis}, sign={self.los_sign}, "
                f"r_max={self.r_max_mpc} cMpc/h, n_steps={self.n_steps}, "
                f"dv_lya={self.dv_lya.item():.1f} km/s, tau_cgm={self.tau_cgm.item():.3f}")


# ------------------------------------------------------------------ #
#  Auxiliary loss
# ------------------------------------------------------------------ #

def los_transmission_loss(
    model_out: dict,
    graph,
    target: str = "tigm",
    t_floor: float = 1e-3,
) -> torch.Tensor:
    """
    Auxiliary LOS-transmission constraint (returns a scalar; 0 if unavailable).

    target = "tigm":  MSE( log T_IGM_hat , log T_IGM_obs )   using graph.tigm_obs
    target = "lya" :  MSE( log L_Lya_obs , log L_Lya_int + log T_IGM_hat )
                      using graph.lya_obs and graph.lya_int (intrinsic proxy)

    Only sources with a valid (positive) measurement contribute.
    """
    T_hat = model_out.get("T_IGM_hat", None)
    if T_hat is None:
        return T_hat.new_zeros(()) if torch.is_tensor(T_hat) else torch.zeros(())

    if target == "lya":
        L_obs = getattr(graph, "lya_obs", None)
        L_int = getattr(graph, "lya_int", None)
        if L_obs is None or L_int is None:
            return T_hat.new_zeros(())
        mask = (L_obs > 0) & (L_int > 0)
        if mask.sum() == 0:
            return T_hat.new_zeros(())
        log_obs = torch.log(L_obs[mask])
        log_pred = torch.log(L_int[mask]) + torch.log(T_hat[mask].clamp(min=t_floor))
        return F.mse_loss(log_pred, log_obs)

    # default: direct T_IGM target
    T_obs = getattr(graph, "tigm_obs", None)
    if T_obs is None:
        return T_hat.new_zeros(())
    mask = T_obs > t_floor                      # tigm=0 marks "no measurement"
    if mask.sum() == 0:
        return T_hat.new_zeros(())
    log_hat = torch.log(T_hat[mask].clamp(min=t_floor))
    log_obs = torch.log(T_obs[mask].clamp(min=t_floor, max=1.0))
    return F.mse_loss(log_hat, log_obs)
