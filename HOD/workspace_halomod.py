import hmf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import pandas as pd
import astropy.units as u
from astropy.cosmology import Planck18
from importlib import reload
from . import HOD_FRAMEWORK_CCF as HOD_FRAMEWORK_CCF

# Reload the HOD framework
reload(HOD_FRAMEWORK_CCF)

# Set the redshift and cosmology
redshift = 6.0
cosmology_params = [0.3089, 0.6911, 0.6774, 0.8159, 0.9667]
Om0, Ov0, h0, sig8, ns = cosmology_params

# Set HOD parameters for galaxy 1 (OIII emitters) and galaxy 2
# hod_param = [Mmin, sigma, Msat, Mcut, alpha_s]
hod_params_gal1 = [10**10, 0.2, 10**13.0,   10**-99, 1.0] # galaxy 1: OIII emitters
hod_params_gal2 = [10**10, 0.2, 10**13.0, 10**-99, 1.0] # galaxy 2: Pop III absorbers
Mmin1, sigma1, Msat1, Mcut1, alpha_s1 = hod_params_gal1
Mmin2, sigma2, Msat2, Mcut2, alpha_s2 = hod_params_gal2

# Initialize HOD framework
HOD = HOD_FRAMEWORK_CCF.HOD_FRAMEWORK_CCF(Om0, Ov0, h0, sig8, ns, redshift)

# Define parameters for CCF computation
r = np.logspace(-2, 1.5, 50)  # cMpc/h

# Compute CCF for different minimum halo masses
acf = HOD.galaxy_cross_correlation_function(r, 
                Mmin1, sigma1, Msat1, Mcut1, alpha_s1, 
                Mmin2, sigma2, Msat2, Mcut2, alpha_s2 )

import halomod
from halomod.cross_correlations import ConstantCorr, CrossCorrelations, _HODCross

print(f"Using halomod v{halomod.__version__} and hmf v{hmf.__version__}")

cross = CrossCorrelations(
    cross_hod_model=ConstantCorr,
    halo_model_1_params={"hod_model": "Zheng05", "transfer_model": "EH", "z": redshift, "Mmin": 10},
    halo_model_2_params={"hod_model": "Zheng05", "transfer_model": "EH", "z": redshift, "Mmin": 10},
)

#cross.halo_model_1.Mmin=11
#cross.halo_model_2.Mmin=9
#cross.halo_model_1.z = 0.5

fig = plt.figure(figsize=(8, 6))
plt.plot(cross.halo_model_1.r, cross.corr_cross - 1, ls="-", label="total")
plt.plot(r, acf, label="HOD_FRAMEWORK_CCF")
# plt.plot(cross.halo_model_1.r, cross.corr_2h_cross, ls="--", label="2-halo")
# plt.plot(cross.halo_model_1.r, cross.corr_1h_cross, ls=":", label="1-halo")
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-5, 1e5)
plt.legend()
plt.ylabel(r"$\xi_{\rm cross}$")
plt.xlabel(r"r [Mpc/h]")

plt.show()