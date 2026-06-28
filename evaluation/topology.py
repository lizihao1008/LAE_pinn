"""
evaluation/topology.py
Topology statistics on 3D ionization fields.

All functions take numpy arrays (G, G, G) and return scalar or 1D numpy arrays.

Statistics implemented:
    granulometry(field, threshold, r_max, n_steps)
        → (n_steps,) ionized volume fraction vs erosion radius
    bubble_size_distribution(field, threshold)
        → (n_bubbles,) radii estimated via SDF / watershed
    percolation_fraction(field, threshold)
        → float: largest connected component / total ionized volume
    connected_components(field, threshold)
        → int: number of distinct ionized regions
    ion_density_cross_correlation(xbox, dbox, box_size, r_bins)
        → (n_bins,) ξ_xρ(r) cross-correlation
"""

from __future__ import annotations
import numpy as np
from scipy import ndimage


# ------------------------------------------------------------------ #
#  Granulometry
# ------------------------------------------------------------------ #

def granulometry(
    field: np.ndarray,
    threshold: float = 0.5,
    r_max: float = 30.0,
    n_steps: int = 20,
    pixel_size: float = 1.0,    # cMpc/h per voxel
) -> tuple[np.ndarray, np.ndarray]:
    """
    Morphological granulometry: erode the binary field with increasing spheres.

    Returns
    -------
    radii    : (n_steps,) erosion sphere radii in cMpc/h
    fractions: (n_steps,) ionized volume fraction after erosion
    """
    binary = (field > threshold).astype(np.uint8)
    vol0   = binary.sum()
    if vol0 == 0:
        return np.zeros(n_steps), np.zeros(n_steps)

    radii     = np.linspace(0, r_max, n_steps)
    fractions = np.zeros(n_steps)

    for idx, r in enumerate(radii):
        r_pix = r / pixel_size
        if r_pix < 0.5:
            fractions[idx] = vol0
            continue
        # Spherical structuring element
        struct = _sphere_struct(r_pix)
        eroded = ndimage.binary_erosion(binary, structure=struct,
                                        border_value=0)
        fractions[idx] = eroded.sum()

    fractions = fractions / (vol0 + 1e-8)
    return radii, fractions


def _sphere_struct(r_pix: float) -> np.ndarray:
    """Create a binary spherical structuring element of radius r_pix pixels."""
    r = int(np.ceil(r_pix))
    coords = np.arange(-r, r + 1)
    cx, cy, cz = np.meshgrid(coords, coords, coords, indexing="ij")
    return (cx ** 2 + cy ** 2 + cz ** 2) <= r_pix ** 2


# ------------------------------------------------------------------ #
#  Bubble size distribution
# ------------------------------------------------------------------ #

def bubble_size_distribution(
    field: np.ndarray,
    threshold: float = 0.5,
    pixel_size: float = 1.0,
) -> np.ndarray:
    """
    Estimate bubble radii via the distance transform of the binary field.

    For each connected component, the effective radius is:
        R_eff = (3 V / 4π)^{1/3}

    Returns array of effective radii in cMpc/h.
    """
    binary = (field > threshold).astype(np.uint8)
    labelled, n = ndimage.label(binary)
    if n == 0:
        return np.array([])

    radii = []
    for comp in range(1, n + 1):
        vol = (labelled == comp).sum()
        R_eff = (3 * vol / (4 * np.pi)) ** (1 / 3) * pixel_size
        radii.append(R_eff)

    return np.array(sorted(radii, reverse=True))


# ------------------------------------------------------------------ #
#  Percolation
# ------------------------------------------------------------------ #

def percolation_fraction(
    field: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Fraction of ionized volume in the largest connected component.
    = 1.0 → fully percolated; << 1.0 → many isolated bubbles.
    """
    binary = (field > threshold).astype(np.uint8)
    labelled, n = ndimage.label(binary)
    if n == 0:
        return 0.0
    sizes = ndimage.sum(binary, labelled, range(1, n + 1))
    return float(max(sizes)) / (binary.sum() + 1e-8)


def connected_components(
    field: np.ndarray,
    threshold: float = 0.5,
) -> int:
    """Number of distinct ionized connected components."""
    binary = (field > threshold).astype(np.uint8)
    _, n = ndimage.label(binary)
    return n


# ------------------------------------------------------------------ #
#  Cross-correlation ξ_xρ(r)
# ------------------------------------------------------------------ #

def ion_density_cross_correlation(
    xbox: np.ndarray,       # (G, G, G) ionization fraction
    dbox: np.ndarray,       # (G, G, G) density contrast
    r_bins: np.ndarray,     # bin edges in cMpc/h
    pixel_size: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the cross-correlation ξ_{xρ}(r) via FFT convolution.

    Returns
    -------
    r_centres : (n_bins,) bin centres in cMpc/h
    xi_xrho   : (n_bins,) cross-correlation values
    """
    G = xbox.shape[0]

    # Normalise fields to zero mean
    dx = xbox - xbox.mean()
    dr = dbox - dbox.mean()

    # Cross-power spectrum via FFT
    Dx = np.fft.rfftn(dx)
    Dr = np.fft.rfftn(dr)
    cross_fft = Dx * np.conj(Dr)
    xi_3d = np.fft.irfftn(cross_fft, s=xbox.shape).real / (G ** 3)

    # Build r grid (shifted so origin at [0,0,0])
    xi_3d = np.roll(xi_3d, (G // 2, G // 2, G // 2), axis=(0, 1, 2))
    coords = (np.arange(G) - G // 2) * pixel_size
    cx, cy, cz = np.meshgrid(coords, coords, coords, indexing="ij")
    r_grid = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)

    r_flat    = r_grid.flatten()
    xi_flat   = xi_3d.flatten()

    n_bins    = len(r_bins) - 1
    r_centres = 0.5 * (r_bins[:-1] + r_bins[1:])
    xi_xrho   = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i + 1])
        if mask.sum() > 0:
            xi_xrho[i] = xi_flat[mask].mean()

    return r_centres, xi_xrho


# ------------------------------------------------------------------ #
#  Summary function
# ------------------------------------------------------------------ #

def compute_all_topology(
    xbox: np.ndarray,
    dbox: np.ndarray | None,
    box_size: float = 160.0,
    threshold: float = 0.5,
    r_max_gran: float = 30.0,
    n_gran_steps: int = 20,
) -> dict:
    """Compute all topology statistics for a single field."""
    G = xbox.shape[0]
    pixel_size = box_size / G

    result = {}

    r_gran, f_gran = granulometry(xbox, threshold, r_max_gran, n_gran_steps, pixel_size)
    result["granulometry_r"]   = r_gran
    result["granulometry_f"]   = f_gran

    bsd = bubble_size_distribution(xbox, threshold, pixel_size)
    result["bsd"] = bsd
    result["bsd_median"] = float(np.median(bsd)) if len(bsd) > 0 else 0.0
    result["bsd_mean"]   = float(np.mean(bsd))   if len(bsd) > 0 else 0.0

    result["percolation_fraction"]  = percolation_fraction(xbox, threshold)
    result["n_connected_components"] = connected_components(xbox, threshold)

    if dbox is not None:
        r_bins = np.linspace(0, box_size / 2, 21)
        r_c, xi = ion_density_cross_correlation(xbox, dbox, r_bins, pixel_size)
        result["xi_xrho_r"]  = r_c
        result["xi_xrho"]    = xi

    return result
