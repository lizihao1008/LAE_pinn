"""
data/conditional_basis.py
Conditional-ACF unresolved faint-source emissivity basis.

Physical model (replaces the global linear-bias field 1 + b_b*delta)
--------------------------------------------------------------------
Instead of painting the unresolved photon budget with a global linear-bias
field (which carries a uniform DC pedestal and therefore keeps voids bright),
we build the unresolved faint-source field as a CONDITIONAL stack around the
observed LAEs -- used ONLY as a spatial prior, with the photon budget fixed
separately so overlapping profiles do not double-count the faint population.

    1. unnormalised stack (spatial pattern):
           eps_tilde_b(x) = Sum_i  w_{i,b}^HOD  u_b^s(x - x_i | M_i)
    2. per-bin UNIT-MEAN normalisation (pure spatial basis):
           eps_b(x) = eps_tilde_b(x) / <eps_tilde_b>
    3. impose the HMF/LF/HOD photon budget and learn only the escape fraction:
           S_unres(x) = Sum_b  f_esc,b  ebar_b  eps_b(x)

where
    i            runs over observed LAEs at redshift-space positions x_i,
    b            is a faint-source luminosity / halo-mass bin,
    w_{i,b}^HOD  = <N_b|M_i>, the expected number of bin-b faint companions of
                 an LAE of mass M_i (HOD conditional count) -- a RELATIVE weight;
                 its absolute scale is removed by step 2,
    u_b^s        is the 3D redshift-space conditional distribution of the faint
                 excess around an LAE, from the LAE x faint-bin cross-correlation
                 xi_{LAE,b}(r) broadened along the LOS by RSD / redshift errors /
                 selection (the 's' = redshift space),
    ebar_b       = n_bar_b <L_b>, the TOTAL unresolved emissivity of bin b from
                 HMF/LF/HOD (the photon budget; independent of the LAE stack),
    f_esc,b      the ONLY learned parameter (escape fraction).

Why the unit-mean step matters
------------------------------
Stacking a profile around every observed LAE and summing means a faint source
near several LAEs is counted several times; without normalisation the global
budget <eps_tilde_b> grows with the LAE number and with profile overlap.
Step 2 fixes <eps_b> = 1, so the stack sets only the relative spatial pattern,
and step 3 sets the absolute budget from HMF/LF/HOD.  Overlap then raises only
the LOCAL relative density, never the global photon budget.

Only the per-bin escape fraction f_esc,b is learned downstream
(physics/unresolved_sources.py); this module pre-computes the fixed fields
basis_b(x) = ebar_b eps_b(x) (so <basis_b> = ebar_b) and returns them in the
same HODCalibration container used by the linear-bias path, so models.pinn /
HODUnresolvedField are unchanged.

Why this fixes the x_HII ~0.18 floor
------------------------------------
S_unres,b is a sum of localized profiles centred on observed LAEs.  Far from
any LAE (in voids) it -> 0, so the ionizing field J -> 0 and the equilibrium /
bubble mapping returns x_HII -> 0 there.  There is no uniform pedestal.

Key implementation insight
--------------------------
Sum_i W_i * u(x - x_i)  ==  scatter(LAEs, weights W_i)  convolved-with  u.
So each (LAE-mass-bin, faint-bin) contribution is one trilinear scatter plus
one FFT convolution with the (fixed) redshift-space profile kernel.

Observation-consistent, NOT oracle
----------------------------------
The spatial template comes from the auto-correlation of the MUV-cut (detected)
LAEs -- an OBSERVABLE -- measured directly from the catalog
(profile_source="observed_acf", the default).  Faint-halo POSITIONS are never
used: in a real survey the faint sources are exactly what cannot be seen, so the
model must INFER their distribution from the observed clustering.  Only the
faint LF quantities (n_bar_b, <L_b>) enter, as the photon budget ebar_b.

The halo-model CCF (profile_source="cof", via COF_tools.HOD_FRAMEWORK_CCF) is
kept for the real-observation transfer case (HOD anchored to the measured
observed ACF).  A power-law xi(r) ("powerlaw") is a dependency-free test stub.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    from .preprocessing import HODCalibration, _scatter_count_to_grid
except ImportError:  # allow standalone import in tests
    HODCalibration = None
    _scatter_count_to_grid = None


# ------------------------------------------------------------------ #
#  Real-space cross-correlation suppliers
# ------------------------------------------------------------------ #

def xi_cross_powerlaw(r0: float = 4.0, gamma: float = 1.8, r_soft: float = 0.5):
    """
    Power-law cross-correlation xi(r) = (r/r0)^(-gamma), softened at r<r_soft.

    A lightweight stand-in for the HOD CCF used for testing the stacking
    geometry without cosmology dependencies.  r0, r in cMpc/h.
    """
    def _xi(r):
        r = np.asarray(r, dtype=np.float64)
        rr = np.clip(r, r_soft, None)
        return (rr / r0) ** (-gamma)
    return _xi


def make_xi_cross_from_cof(
    redshift: float,
    logM_lae: float,
    logM_faint: float,
    cosmo: tuple[float, float, float, float, float] = (0.3089, 0.6911, 0.6774, 0.8159, 0.9667),
    sigma: float = 0.2,
    ignore_1h: bool = False,
):
    """
    Build a real-space xi_{LAE,faint}(r) callable from ``HOD.HOD_FRAMEWORK_CCF``.

    Each population is modelled as a narrow central-only HOD anchored at its
    mean halo mass (logMmin = logM_*, satellites switched off via a very large
    logMsat), so the cross-correlation is set by the two populations' biases
    plus the 1-halo term.  Plug your ACF-fitted HOD parameters here for the
    production run.

    Parameters
    ----------
    redshift   : snapshot redshift
    logM_lae   : log10(M_halo/[Msun/h]) of the observed-LAE (mass) bin
    logM_faint : log10(M_halo/[Msun/h]) of the faint-source bin
    cosmo      : (Om0, Ov0, h0, sig8, ns)
    sigma      : log-mass scatter of each HOD central kernel
    ignore_1h  : drop the 1-halo term (keep only large-scale 2-halo clustering)

    Returns
    -------
    xi_cross(r) callable, r in cMpc/h.
    """
    # Imported lazily: HOD needs scipy/astropy only on the production machine.
    try:
        from ..HOD import HOD_FRAMEWORK_CCF
    except ImportError:
        from HOD import HOD_FRAMEWORK_CCF

    Om0, Ov0, h0, sig8, ns = cosmo
    ccf = HOD_FRAMEWORK_CCF(Om0, Ov0, h0, sig8, ns, redshift)

    logMsat_off = 18.0   # satellites effectively off (Msat >> any halo)
    logMcut     = 10.0
    alpha_s     = 1.0
    ccf.define_model(
        logM_lae,   sigma, logMsat_off, logMcut, alpha_s,   # population 1 (LAE)
        logM_faint, sigma, logMsat_off, logMcut, alpha_s,   # population 2 (faint)
    )

    def _xi(r):
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        return ccf.galaxy_cross_correlation_function(
            r,
            10 ** logM_lae,   sigma, 10 ** logMcut, 10 ** logMsat_off, alpha_s,
            10 ** logM_faint, sigma, 10 ** logMcut, 10 ** logMsat_off, alpha_s,
            ignore_1h=ignore_1h,
        )
    return _xi


# ------------------------------------------------------------------ #
#  Redshift-space conditional profile kernel u_b^s(r_perp, r_par)
# ------------------------------------------------------------------ #

def sigma_los_from_dv(dv_max_kms: float, redshift: float, h0: float = 0.6774,
                      Om0: float = 0.3089, Ov0: float = 0.6911) -> float:
    """
    Line-of-sight comoving smoothing length [cMpc/h] from a velocity window.

    sigma_los = dv * (1+z) / H(z) * h ,  H(z) = 100 h E(z) km/s/(Mpc/h).
    Captures RSD (Fingers-of-God), redshift errors and selection-window
    broadening along the LOS.  This mirrors the r_los_max used in
    HOD/galaxy_angular_cross_correlation_function.
    """
    Ez = np.sqrt(Om0 * (1.0 + redshift) ** 3 + Ov0)
    Hz = 100.0 * h0 * Ez                      # km/s per (Mpc/h)
    return float(dv_max_kms * (1.0 + redshift) / Hz * h0)


def build_redshift_space_profile(
    xi_cross,
    grid_size: int,
    box_size: float,
    sigma_los: float,
    los_axis: int = 2,
    r_max: float | None = None,
) -> np.ndarray:
    """
    Build the 3D redshift-space conditional excess kernel u^s(r_perp, r_par)
    on a (G,G,G) grid, centred at (0,0,0) and FFT-ready (zero-lag at origin).

    Steps
    -----
    1. real-space excess  w(r) = max(xi_cross(r), 0)   [clustered excess only;
       no DC pedestal so the field -> 0 far from LAEs]
    2. RSD: convolve along the LOS axis with a Gaussian of width sigma_los
       (redshift-space broadening).  This makes the kernel anisotropic:
       elongated along the line of sight.
    3. clip >= 0, normalise to unit sum (so a scatter+convolve conserves the
       deposited companion weight).

    Returns (G,G,G) float64 kernel, rolled so index [0,0,0] is zero-lag.
    """
    G = grid_size
    dx = box_size / G

    coords = (np.arange(G) - G // 2) * dx
    cx, cy, cz = np.meshgrid(coords, coords, coords, indexing="ij")
    r = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
    r = np.clip(r, dx / 2, None)

    w = np.asarray(xi_cross(r.ravel()), dtype=np.float64).reshape(r.shape)
    w = np.clip(w, 0.0, None)
    if r_max is not None:
        w[r > r_max] = 0.0

    # RSD: Gaussian smear along the LOS axis (separable 1D convolution)
    if sigma_los > 0:
        n_sig = max(1, int(np.ceil(3 * sigma_los / dx)))
        t = np.arange(-n_sig, n_sig + 1) * dx
        g = np.exp(-0.5 * (t / sigma_los) ** 2)
        g /= g.sum()
        w = _convolve_1d_axis(w, g, axis=los_axis)

    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total > 1e-30:
        w = w / total

    # roll so zero-lag is at [0,0,0] for FFT convolution
    return np.roll(w, (G // 2, G // 2, G // 2), axis=(0, 1, 2))


def _convolve_1d_axis(field: np.ndarray, kern1d: np.ndarray, axis: int) -> np.ndarray:
    """Convolve a 3D field with a 1D kernel along one axis (reflect padding)."""
    field = np.moveaxis(field, axis, -1)
    pad = len(kern1d) // 2
    fp = np.pad(field, [(0, 0), (0, 0), (pad, pad)], mode="reflect")
    out = np.zeros_like(field)
    for j, kj in enumerate(kern1d):
        out += kj * fp[:, :, j:j + field.shape[-1]]
    return np.moveaxis(out, -1, axis)


# ------------------------------------------------------------------ #
#  Scatter + convolve helper (pure numpy; mirrors physics/scatter.py)
# ------------------------------------------------------------------ #

def _scatter_weights_to_grid(pos_norm: np.ndarray, weights: np.ndarray, G: int) -> np.ndarray:
    """Trilinear scatter of weighted points onto a (G,G,G) grid (numpy)."""
    grid = np.zeros((G, G, G), dtype=np.float64)
    pix = np.clip(pos_norm * G, 0, G - 1e-6)
    i0 = np.floor(pix).astype(int)
    frac = pix - i0
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                bx = (i0[:, 0] + cx) % G
                by = (i0[:, 1] + cy) % G
                bz = (i0[:, 2] + cz) % G
                wx = frac[:, 0] if cx else 1 - frac[:, 0]
                wy = frac[:, 1] if cy else 1 - frac[:, 1]
                wz = frac[:, 2] if cz else 1 - frac[:, 2]
                np.add.at(grid, (bx, by, bz), weights * wx * wy * wz)
    return grid


def _fft_convolve(S: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Periodic FFT convolution; K already rolled to zero-lag at origin."""
    return np.fft.irfftn(np.fft.rfftn(S) * np.fft.rfftn(K), s=S.shape)


# ------------------------------------------------------------------ #
#  Main builder
# ------------------------------------------------------------------ #

def build_conditional_unresolved_basis(
    snap,                                   # SimSnapshot of OBSERVED LAEs (graph nodes)
    muv_bin_edges: list[float],             # faint-source M_UV bin edges
    grid_size: int = 64,
    n_lae_mass_bins: int = 4,               # split LAEs by halo mass
    dv_max_kms: float = 1000.0,             # LOS redshift-space window
    los_axis: int = 2,
    r_max: float | None = 40.0,             # truncate profile [cMpc/h]
    profile_source: str = "observed_acf",    # "observed_acf" | "cof" | "powerlaw"
    cosmo: tuple = (0.3089, 0.6911, 0.6774, 0.8159, 0.9667),
    snap_full=None,                         # full catalog: faint LF (n_bar, <L>) only
    muv_det: float | None = None,
    use_cof: bool | None = None,            # deprecated alias for profile_source
):
    """
    Build the conditional-ACF unresolved emissivity basis S_unres,b(x).

    Returns an object exposing `.basis_fields` (n_bins, G, G, G), `.bin_labels`,
    `.hod_params`, `.source` -- API-compatible with preprocessing.HODCalibration
    so it is a drop-in for the linear-bias basis.

    profile_source
    --------------
    "observed_acf" (default, observation-consistent):
        The spatial template is derived ONLY from observables.  We measure the
        auto-correlation xi^s(r) of the MUV-cut (detected) LAEs from the catalog
        (redshift space, so RSD is included) and paint that clustering template
        around each detected LAE.  In linear bias the faint x LAE cross-
        correlation has the same radial shape, so one template serves all faint
        bins; the LF budget ebar_b sets the per-bin amplitude.  Faint-halo
        POSITIONS are never used -- the model must INFER the faint distribution
        from the observed clustering, exactly as on real data.
    "cof":
        Analytic halo-model CCF from COF_tools.HOD_FRAMEWORK_CCF, for the
        observation transfer case.  The LAE HOD should be anchored to the
        measured observed-LAE ACF; faint-halo positions are still never used.
    "powerlaw":
        xi(r) = (r/r0)^-gamma fallback (no cosmology deps; for quick tests).

    Why not cross-correlate with the faint-halo catalog?
    ----------------------------------------------------
    That would be an ORACLE: in a real survey the faint sources are exactly what
    you cannot see, so their true positions are unavailable.  Training on them
    teaches nothing transferable.  Hence only the detected-LAE ACF is used.

    Notes
    -----
    * Observed LAEs come from `snap` (the same catalog used to build the graph).
    * Only the faint LF quantities n_bar_b and <L_b> are taken from `snap_full`
      (global statistics that the UV LF measures/extrapolates) -- NOT positions.
    * Per-bin UNIT-MEAN normalisation + the LF/HMF budget ebar_b keep the global
      photon budget independent of LAE number / profile overlap.
    """
    if use_cof is not None:                 # back-compat
        profile_source = "cof" if use_cof else "powerlaw"
    G = grid_size
    box = snap.box_size
    z = float(snap.redshift)
    Om0, Ov0, h0, sig8, ns = cosmo
    full = snap_full if snap_full is not None else snap
    if muv_det is None:
        muv_det = float(np.max(snap.muv))   # faint = fainter than detected LAEs

    sigma_los = sigma_los_from_dv(dv_max_kms, z, h0, Om0, Ov0)
    dV = (box / G) ** 3

    # ---- LAE mass bins (population 1) ----
    # Positions are taken to REDSHIFT space (peculiar-velocity LOS shift) so the
    # measured conditional profile is the redshift-space u_b^s(r_perp,r_par).
    lae_logM = np.log10(np.clip(snap.halo_mass, 1e6, None))
    edges = np.quantile(lae_logM, np.linspace(0, 1, n_lae_mass_bins + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    lae_pos_norm = _redshift_space_pos_norm(
        snap.pos, getattr(snap, "pec_vel", None), box, z, los_axis, cosmo)
    lae_mass_bin = np.clip(np.digitize(lae_logM, edges) - 1, 0, n_lae_mass_bins - 1)
    lae_logM_mean = np.array([
        lae_logM[lae_mass_bin == m].mean() if np.any(lae_mass_bin == m) else lae_logM.mean()
        for m in range(n_lae_mass_bins)
    ])

    # ---- faint-source bins (population 2): positions, <L>, n_bar, mean mass ----
    n_bins = len(muv_bin_edges)            # last bin open-ended
    faint_mask = full.muv > muv_det
    f_muv = full.muv[faint_mask]
    f_lya = full.lya_obs[faint_mask]
    f_logM = np.log10(np.clip(full.halo_mass[faint_mask], 1e6, None))
    box_vol = box ** 3   # faint POSITIONS are intentionally NOT used (no oracle)

    bin_labels, Lbar_b, nbar_b, faint_logM = [], [], [], []
    for b in range(n_bins):
        lo = muv_bin_edges[b]
        hi = muv_bin_edges[b + 1] if b + 1 < n_bins else np.inf
        if np.isinf(hi):
            m_b = f_muv > lo;  label = f"MUV>{lo:.1f}"
        else:
            m_b = (f_muv > lo) & (f_muv <= hi);  label = f"MUV({lo:.1f},{hi:.1f}]"
        bin_labels.append(label)
        N_b = int(m_b.sum())
        Lbar_b.append(float(np.mean(np.clip(f_lya[m_b], 0, None))) if N_b > 0 else 0.0)
        nbar_b.append(N_b / box_vol)
        faint_logM.append(float(f_logM[m_b].mean()) if N_b > 0 else 8.0)

    # Relative mean luminosity per bin.  Absolute L (~1e41 erg/s) overflows
    # float32 and its overall scale is absorbed downstream by f_esc and the
    # model's amplitude calibration; only the RELATIVE amplitude across bins is
    # physical here, so normalise by the brightest bin.
    L_ref = max(Lbar_b) if max(Lbar_b) > 0 else 1.0
    Lbar_rel = [L / L_ref for L in Lbar_b]

    # Absolute (relative-across-bins) unresolved EMISSIVITY budget per bin, set
    # by HMF/LF/HOD:  ebar_b = n_bar_b * <L_b>.  This is the photon budget; the
    # LAE stack only supplies the spatial PATTERN.  Keeping it separate is what
    # prevents profile overlap / LAE count from inflating the global budget.
    ebar_b = [nbar_b[b] * Lbar_rel[b] for b in range(n_bins)]

    # ---- assemble per bin: spatial prior (unit-mean) x photon budget ----
    #   eps_b(x)   = unit-mean spatial template from the OBSERVED-LAE clustering
    #   basis_b(x) = ebar_b * eps_b(x)   =>   <basis_b> = ebar_b
    # The SHAPE comes only from observables (the detected-LAE ACF); the photon
    # budget ebar_b comes from the LF/HMF.  Faint-halo POSITIONS are never used
    # -- using them would be an oracle the real observation does not have.
    basis = np.zeros((n_bins, G, G, G), dtype=np.float64)
    Ncomp = np.zeros((n_lae_mass_bins, n_bins))

    if profile_source == "observed_acf":
        # Measure the auto-correlation of the MUV-cut LAEs (per LAE mass bin, in
        # redshift space) and paint that clustering template around each detected
        # LAE.  In linear bias the faint x LAE cross-correlation has the SAME
        # radial shape as the LAE auto-correlation, so one spatial template
        # serves all faint bins; only the LF budget ebar_b differs between bins.
        # The profile is already redshift-space (positions are in z-space), so no
        # extra RSD smearing is applied (sigma_los=0 here).
        eps_template = np.zeros((G, G, G), dtype=np.float64)
        acf_table = []
        for m in range(n_lae_mass_bins):
            sel = (lae_mass_bin == m)
            if int(sel.sum()) < 8:
                continue
            rc, xi_r = measure_lae_acf_radial(lae_pos_norm[sel], G, box, r_max)
            xi_obs = _xi_interp_from_acf(rc, xi_r)
            u_m = build_redshift_space_profile(xi_obs, G, box, sigma_los=0.0,
                                               los_axis=los_axis, r_max=r_max)
            S_pts = _scatter_weights_to_grid(lae_pos_norm[sel], np.ones(int(sel.sum())), G)
            eps_template += _fft_convolve(S_pts, u_m)
            acf_table.append([float(lae_logM_mean[m]), rc.tolist(), xi_r.tolist()])
        eps_template = np.clip(eps_template, 0.0, None)
        mt = float(eps_template.mean())
        if mt > 1e-30:
            eps_template /= mt                        # unit-mean shared template
            for b in range(n_bins):
                if ebar_b[b] > 0:
                    basis[b] = ebar_b[b] * eps_template
    else:
        # halo-model (cof) / power-law: profile depends on (LAE mass m, faint b).
        # For the OBSERVATION transfer case the LAE HOD should be anchored to the
        # measured observed-LAE ACF; faint-halo positions are still never used.
        acf_table = []
        for b in range(n_bins):
            if ebar_b[b] <= 0:
                continue
            eps_tilde = np.zeros((G, G, G), dtype=np.float64)
            for m in range(n_lae_mass_bins):
                sel = (lae_mass_bin == m)
                if not np.any(sel):
                    continue
                if profile_source == "cof":
                    xi = make_xi_cross_from_cof(z, lae_logM_mean[m], faint_logM[b], cosmo)
                else:  # "powerlaw"
                    r0 = 3.0 * (10 ** ((lae_logM_mean[m] + faint_logM[b]) / 2 - 10)) ** 0.05
                    xi = xi_cross_powerlaw(r0=max(r0, 1.0), gamma=1.8)
                u = build_redshift_space_profile(xi, G, box, sigma_los, los_axis, r_max)
                xi_excess_integral = float(np.clip(
                    np.asarray(xi(_radial_grid(G, box).ravel())), 0, None).sum()) * dV
                Ncomp[m, b] = nbar_b[b] * xi_excess_integral
                w_hod = Ncomp[m, b]
                if w_hod <= 0:
                    continue
                S_pts = _scatter_weights_to_grid(lae_pos_norm[sel], np.full(int(sel.sum()), w_hod), G)
                eps_tilde += _fft_convolve(S_pts, u)
            eps_tilde = np.clip(eps_tilde, 0.0, None)
            mean_tilde = float(eps_tilde.mean())
            if mean_tilde <= 1e-30:
                continue
            basis[b] = ebar_b[b] * (eps_tilde / mean_tilde)

    hod_params = {
        "bias":         [0.0] * n_bins,            # not used by this path
        "n_bar":        nbar_b,
        "mean_log_lum": [np.log10(L + 1e-30) for L in Lbar_b],
        "N_halos":      [int((f_muv > (muv_bin_edges[b])).sum()) for b in range(n_bins)],
        "N_companions": Ncomp.tolist(),
        "lae_logM_bins": lae_logM_mean.tolist(),
        # emissivity budget per bin (n_bar*<L>, relative across bins). After
        # unit-mean normalisation of the spatial basis, <basis_b> = ebar_b.
        "ebar_emissivity": ebar_b,
        # measured observed-LAE auto-correlation [(logM_mean, r[cMpc/h], xi(r)), ...]
        "acf_measured": acf_table,
    }

    cal = _make_calibration(
        basis_fields=basis.astype(np.float32),
        bin_labels=bin_labels,
        hod_params=hod_params,
        source=f"conditional_acf:{profile_source}",
    )
    return cal


def _radial_grid(G: int, box: float) -> np.ndarray:
    """Distance from the grid CENTRE [cMpc/h] (for centred profiles)."""
    dx = box / G
    coords = (np.arange(G) - G // 2) * dx
    cx, cy, cz = np.meshgrid(coords, coords, coords, indexing="ij")
    r = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
    return np.clip(r, dx / 2, None)


def _radial_grid_fft(G: int, box: float) -> np.ndarray:
    """Periodic (minimum-image) distance from index [0,0,0], for FFT-lag fields."""
    dx = box / G
    c = np.minimum(np.arange(G), G - np.arange(G)) * dx
    cx, cy, cz = np.meshgrid(c, c, c, indexing="ij")
    return np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)


def _redshift_space_pos_norm(pos, pec_vel, box, z, los_axis, cosmo):
    """
    Map comoving positions [cMpc/h] to REDSHIFT-SPACE normalised coords [0,1).

    s_los = r_los + v_pec_los (1+z) / H(z),  H(z)=100 E(z) [(km/s)/(cMpc/h)].
    If pec_vel is None, returns the real-space normalised positions.
    """
    Om0, Ov0, h0, sig8, ns = cosmo
    p = (np.asarray(pos, dtype=np.float64) / box)
    if pec_vel is not None:
        Ez = np.sqrt(Om0 * (1.0 + z) ** 3 + Ov0)
        d_los = np.asarray(pec_vel)[:, los_axis] * (1.0 + z) / (100.0 * Ez)   # cMpc/h
        p = p.copy()
        p[:, los_axis] = p[:, los_axis] + d_los / box
    return p % 1.0


def measure_lae_acf_radial(lae_pos_norm, G, box, r_max=None, n_rbins=24):
    """
    Measure the auto-correlation xi(r) of (redshift-space) LAE positions from the
    catalog via the FFT estimator  xi(r) = <DD(r)>/<RR(r)> - 1, where DD is the
    LAE pair count at lag r and RR the random expectation.  Returns
    (r_centres [cMpc/h], xi(r)).  Uses ONLY observed positions (no oracle).
    """
    N = len(lae_pos_norm)
    dx = box / G
    if N < 2:
        return np.array([dx]), np.array([0.0])
    A = _scatter_weights_to_grid(lae_pos_norm, np.ones(N), G)
    pair = np.fft.irfftn(np.abs(np.fft.rfftn(A)) ** 2, s=(G, G, G)).real  # lag-0 at origin
    rgrid = _radial_grid_fft(G, box)
    rmax = r_max if r_max is not None else box / 2
    edges = np.linspace(dx, rmax, n_rbins + 1)
    Vvox, Vbox = dx ** 3, box ** 3
    rc, xi = [], []
    for i in range(n_rbins):
        sh = (rgrid >= edges[i]) & (rgrid < edges[i + 1])
        nv = int(sh.sum())
        if nv == 0:
            continue
        dd = float(pair[sh].sum())                     # data-data pairs in shell
        rr = N * N * (nv * Vvox) / Vbox                # random expectation
        rc.append(0.5 * (edges[i] + edges[i + 1]))
        xi.append(dd / (rr + 1e-30) - 1.0)
    return np.array(rc), np.array(xi)


def _xi_interp_from_acf(rc, xi):
    """
    Callable xi(r) from a measured (r, xi) table; clips to the clustered
    (positive) excess and tapers to 0 beyond the measured range.
    """
    xc = np.clip(np.asarray(xi, dtype=np.float64), 0.0, None)
    rc = np.asarray(rc, dtype=np.float64)
    def f(r):
        r = np.asarray(r, dtype=np.float64)
        return np.interp(r, rc, xc, left=(xc[0] if len(xc) else 0.0), right=0.0)
    return f


class _CalibrationStandIn:
    """Duck-typed HODCalibration used when preprocessing (torch) is unavailable."""
    def __init__(self, basis_fields, bin_labels, hod_params, source):
        self.basis_fields = basis_fields
        self.bin_labels   = bin_labels
        self.hod_params   = hod_params
        self.source       = source


def _make_calibration(basis_fields, bin_labels, hod_params, source):
    """Return a HODCalibration (if importable) else a duck-typed stand-in."""
    if HODCalibration is not None:
        return HODCalibration(basis_fields=basis_fields, bin_labels=bin_labels,
                              hod_params=hod_params, source=source)
    return _CalibrationStandIn(basis_fields, bin_labels, hod_params, source)
