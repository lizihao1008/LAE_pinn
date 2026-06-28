"""
data/loader.py
Load Sherwood-Relics + ATON simulation data for a given redshift snapshot.

Snapshot directory layout (relative to sim_root):
    Fiducial_z={z}/
        Halo_Position.npy       (N, 3) cMpc/h
        Halo_mass.npy           (N,)   M_sun
        Muv.npy                 (N,)
        Intrinsic_Llya.npy      (N,)
        Observed_Llya.npy       (N,)
        Intrinsic_REW.npy       (N,)
        Observed_REW.npy        (N,)
        Peculiar_velocity_halo.npy (N, 3) km/s
        Xbox_grid_017_512.npy   (512,512,512) ionization fraction x_i
        Dbox_grid_017_512.npy   (512,512,512) matter density contrast

Returns a SimSnapshot dataclass with everything needed downstream.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

# Map human-readable redshift labels to directory names
SNAPSHOT_DIRS = {
    7.14:  "Fiducial_z_7.14",
    6.6:   "Fiducial_z_6.6",
    5.756: "Fiducial_z_5.756",
}

# True mean ionized fractions from METHODS.md
X_HI_TRUE = {7.14: 0.69, 6.6: 0.52, 5.756: 0.13}


@dataclass
class SimSnapshot:
    """Container for one redshift snapshot."""

    redshift: float
    box_size: float                     # cMpc/h

    # Halo / LAE catalog (N halos)
    pos: np.ndarray                     # (N, 3) cMpc/h
    halo_mass: np.ndarray               # (N,) M_sun
    muv: np.ndarray                     # (N,)
    lya_int: np.ndarray                 # (N,) intrinsic Lyα luminosity
    lya_obs: np.ndarray                 # (N,) observed Lyα luminosity
    rew_int: np.ndarray                 # (N,) intrinsic REW
    rew_obs: np.ndarray                 # (N,) observed REW
    pec_vel: np.ndarray                 # (N, 3) km/s

    # Derived
    tigm: np.ndarray                    # (N,) = lya_obs / lya_int

    # Simulation grids (512³)
    xbox_512: np.ndarray                # (512,512,512) ionization fraction
    dbox_512: Optional[np.ndarray]      # (512,512,512) density contrast

    # Global quantities
    x_hi_true: float = field(init=False)

    def __post_init__(self):
        self.x_hi_true = X_HI_TRUE.get(self.redshift, 1.0 - float(self.xbox_512.mean()))

    @property
    def n_halos(self) -> int:
        return len(self.halo_mass)

    @property
    def xi_global(self) -> float:
        """Global ionized fraction from grid."""
        return float(self.xbox_512.mean())

    def filter_by_muv(self, muv_cut: float) -> "SimSnapshot":
        """Return a new snapshot with halos brighter than muv_cut."""
        mask = self.muv < muv_cut   # MUV is negative; brighter = more negative
        return SimSnapshot(
            redshift=self.redshift,
            box_size=self.box_size,
            pos=self.pos[mask],
            halo_mass=self.halo_mass[mask],
            muv=self.muv[mask],
            lya_int=self.lya_int[mask],
            lya_obs=self.lya_obs[mask],
            rew_int=self.rew_int[mask],
            rew_obs=self.rew_obs[mask],
            pec_vel=self.pec_vel[mask],
            tigm=self.tigm[mask],
            xbox_512=self.xbox_512,
            dbox_512=self.dbox_512,
        )

    def filter_by_mass(self, m_min: float = 0., m_max: float = np.inf) -> "SimSnapshot":
        """Return snapshot filtered to halos in [m_min, m_max] M_sun."""
        mask = (self.halo_mass >= m_min) & (self.halo_mass <= m_max)
        return SimSnapshot(
            redshift=self.redshift,
            box_size=self.box_size,
            pos=self.pos[mask],
            halo_mass=self.halo_mass[mask],
            muv=self.muv[mask],
            lya_int=self.lya_int[mask],
            lya_obs=self.lya_obs[mask],
            rew_int=self.rew_int[mask],
            rew_obs=self.rew_obs[mask],
            pec_vel=self.pec_vel[mask],
            tigm=self.tigm[mask],
            xbox_512=self.xbox_512,
            dbox_512=self.dbox_512,
        )


def load_snapshot(
    sim_root: str,
    redshift: float,
    box_size: float = 160.0,
    load_density: bool = True,
) -> SimSnapshot:
    """Load a single redshift snapshot from sim_root."""
    snap_name = SNAPSHOT_DIRS.get(redshift)
    if snap_name is None:
        raise ValueError(f"Unknown redshift {redshift}. Available: {list(SNAPSHOT_DIRS)}")
    snap_dir = os.path.join(sim_root, snap_name)
    if not os.path.isdir(snap_dir):
        raise FileNotFoundError(f"Snapshot directory not found: {snap_dir}")

    def _load(fname):
        return np.load(os.path.join(snap_dir, fname))

    pos      = _load("Halo_Position.npy")          # (N, 3)
    mass     = _load("Halo_mass.npy")              # (N,)
    muv      = _load("Muv.npy")                    # (N,)
    lya_int  = _load("Intrinsic_Llya.npy")         # (N,)
    lya_obs  = _load("Observed_Llya.npy")          # (N,)
    rew_int  = _load("Intrinsic_REW.npy")          # (N,)
    rew_obs  = _load("Observed_REW.npy")           # (N,)
    pec_vel  = _load("Peculiar_velocity_halo.npy") # (N, 3)
    xbox     = _load("Xbox_grid_017_512.npy")      # (512,512,512)

    # TIGM: avoid divide-by-zero
    tigm = np.where(lya_int > 0, lya_obs / lya_int, 0.0)
    tigm = np.clip(tigm, 0.0, 1.0)

    dbox = None
    if load_density:
        dbox_path = os.path.join(snap_dir, "Dbox_grid_017_512.npy")
        if os.path.exists(dbox_path):
            dbox = _load("Dbox_grid_017_512.npy")

    return SimSnapshot(
        redshift=redshift,
        box_size=box_size,
        pos=pos,
        halo_mass=mass,
        muv=muv,
        lya_int=lya_int,
        lya_obs=lya_obs,
        rew_int=rew_int,
        rew_obs=rew_obs,
        pec_vel=pec_vel,
        tigm=tigm,
        xbox_512=xbox,
        dbox_512=dbox,
    )


def load_all_snapshots(
    sim_root: str,
    box_size: float = 160.0,
    load_density: bool = True,
) -> dict[float, SimSnapshot]:
    """Load all available snapshots."""
    snaps = {}
    for z in SNAPSHOT_DIRS:
        snap_dir = os.path.join(sim_root, SNAPSHOT_DIRS[z])
        if os.path.isdir(snap_dir):
            snaps[z] = load_snapshot(sim_root, z, box_size, load_density)
            print(f"  z={z}: {snaps[z].n_halos} halos, xi_global={snaps[z].xi_global:.3f}")
    return snaps


# ------------------------------------------------------------------ #
#  Source-model factories (for stress tests)
# ------------------------------------------------------------------ #

def apply_source_model(snap: SimSnapshot, model: str) -> SimSnapshot:
    """
    Return a version of the snapshot with halos filtered to a specific
    source prescription for stress-test experiments.

    model:
        "fiducial"      — all halos (default)
        "observed_only" — MUV < -17.5 (LAE detection threshold)
        "massive_only"  — M_h > 1e11 M_sun
        "faint_only"    — M_h < 1e10 M_sun
    """
    if model == "fiducial":
        return snap
    elif model == "observed_only":
        return snap.filter_by_muv(-17.5)
    elif model == "massive_only":
        return snap.filter_by_mass(m_min=1e11)
    elif model == "faint_only":
        return snap.filter_by_mass(m_max=1e10)
    else:
        raise ValueError(f"Unknown source model: {model}")


# ------------------------------------------------------------------ #
#  Quick sanity check
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    sim_root = sys.argv[1] if len(sys.argv) > 1 else "../simulation"
    snaps = load_all_snapshots(sim_root)
    for z, s in snaps.items():
        print(f"z={z}: {s.n_halos} halos, xi_global={s.xi_global:.3f}, "
              f"x_HI_true={s.x_hi_true:.2f}")
        lae = s.filter_by_muv(-17.5)
        print(f"  observed LAEs (MUV<-17.5): {lae.n_halos}")
