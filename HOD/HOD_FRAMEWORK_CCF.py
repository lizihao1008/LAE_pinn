from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import erf
from scipy.special import sici
from astropy.cosmology import Planck18
from pylab import *
#import numpy as np
import astropy.units as u

from . import basic_cosmology_toolkits

class HOD_FRAMEWORK_CCF:
    def __init__(self,Om0,Ov0,h0,sig8,ns,redshift):
        """
        A model framework for halo occuation distribution (HOD) approach
        to galaxy power spectrum and angular correlation function
        'Generalized to get cross-correlation function'

        Example
        --------
        >>> [Om0, Ov0, h0, sig8, ns] = [0.3089, 0.6911, 0.6774, 0.8159, 0.9667]
        >>> redshift = 6.0
        >>>
        >>> # set up cosmology and tabulate the halo related functions
        >>> HOD=HOD_Framework(Om0,Ov0,h0,sig8,ns,redshift)
        >>>
        >>> # set up HOD parameters and define the model
        >>>
        >>> [logMmin1,sigma1,logMsat1,logMcut1,alpha_s1]=[12.0, 0.2, 12.0, 10.0, 1.0]
        >>> [logMmin2,sigma2,logMsat2,logMcut2,alpha_s2]=[10.0, 0.2, 12.0, 10.0, 1.0]
        
        >>> HOD.define_model(logMmin1,sigma1,logMsat1,logMcut1,alpha_s1,
                             logMmin2,sigma2,logMsat2,logMcut2,alpha_s2)

        """
        # cosmolgical parameters
        self.Om0=Om0
        self.Ov0=Ov0
        self.h0=h0
        self.sig8=sig8
        self.ns=ns
        self.redshift=redshift

        # make a look-up table for bias, matter power spectrum, halo mass function
        bct=basic_cosmology_toolkits.basic_cosmology_tookits(h0,Om0,Ov0,sig8,ns)
        logMh_min=5.0    # log10(M_halo/[Msun/h]) min bin
        logMh_max=15.0   # log10(M_halo/[Msun/h]) max bin
        dlogMh=0.01      # log10 interval
        logMh=arange(logMh_min,logMh_max+dlogMh,dlogMh)
        Mh=10**logMh
        self.Mh=Mh
        self.logMh=logMh
        self.b=bct.bias(Mh,redshift)
        self.dndlogMh=bct.halo_mass_function(Mh,redshift)

        power_spectrum_type='non-linear' # linear or non-linear
        if power_spectrum_type=='linear':
            self.Pm=bct.P_linear
        elif power_spectrum_type=='non-linear':
            logkLmin=-6
            logkLmax=+6
            kL=logspace(logkLmin,logkLmax,2000)
            kNL,Delta2NL=bct.Delta2_NL(kL,redshift)
            P_nonlinear=((2*pi**2)/kNL**3)*Delta2NL
            self.Pm=interp1d(kNL,P_nonlinear)

    def define_model(self, logMmin1,sigma1,logMsat1,logMcut1,alpha_s1,
                           logMmin2,sigma2,logMsat2,logMcut2,alpha_s2):
        """
        Define CLF model parameters
            define_model(
                logMmin1, : log10(minimum halo mass [Msun/h])
                sigma1,   : scatter of log10(minimum halo mass)
                logMsat1  : satellite parameter
                logMcut1  :
                alpha_s1  :
                logMmin2, : log10(minimum halo mass [Msun/h])
                sigma2,   : scatter of log10(minimum halo mass)
                logMsat2  : satellite parameter
                logMcut2  :
                alpha_s2  :
            )
        """
        # halo 1
        self.Mmin1=10**logMmin1            # [Msun/h]
        self.sigma1=sigma1
        self.Msat1=10**logMsat1            # [Msun/h]
        self.Mcut1=10**logMcut1            # [Msun/h]
        self.alpha_s1=alpha_s1
        # halo 2
        self.Mmin2=10**logMmin2            # [Msun/h]
        self.sigma2=sigma2
        self.Msat2=10**logMsat2            # [Msun/h]
        self.Mcut2=10**logMcut2            # [Msun/h]
        self.alpha_s2=alpha_s2

    # HOD of central galaxies
    def HOD_cen(self,Mh, Mmin, sigma):
        HOD_cen=0.5*(1.+erf( (log10(Mh)-log10(Mmin))/(sigma) ))
        return HOD_cen
    # HOD of satellite galaxies
    def HOD_sat(self,Mh, Mmin, sigma, Mcut, Msat, alpha_s):
        HOD_sat=self.HOD_cen(Mh, Mmin, sigma)*((Mh-Mcut)/Msat)**alpha_s
        return HOD_sat
    # number density of galaxies
    def n_gal(self, Mmin, sigma, Mcut, Msat, alpha_s):
        HOD=self.HOD_cen(self.Mh, Mmin, sigma)+self.HOD_sat(self.Mh, Mmin, sigma, Mcut, Msat, alpha_s)
        n_gal=simps( HOD*self.dndlogMh, self.logMh )
        return n_gal
    # galaxy bias
    def galaxy_bias(self, Mmin, sigma, Mcut, Msat, alpha_s):
        HODcen=self.HOD_cen(self.Mh, Mmin, sigma)
        HODsat=self.HOD_sat(self.Mh, Mmin, sigma, Mcut, Msat, alpha_s)
        HODtot=HODcen+HODsat
        ngal=self.n_gal(Mmin, sigma, Mcut, Msat, alpha_s)
        galaxy_bias=simps( self.dndlogMh * self.b * HODtot/ngal, self.logMh )
        return galaxy_bias
    # average halo mass of galaxies    
    def galaxy_halomass(self, Mmin, sigma, Mcut, Msat, alpha_s):
        HODcen=self.HOD_cen(self.Mh, Mmin, sigma)
        HODsat=self.HOD_sat(self.Mh, Mmin, sigma, Mcut, Msat, alpha_s)
        HODtot=HODcen+HODsat
        ngal=self.n_gal(Mmin, sigma, Mcut, Msat, alpha_s)
        galaxy_halomass=simps( self.dndlogMh * self.Mh * HODtot/ngal, self.logMh )
        return galaxy_halomass
    # NFW profile
    def NFW(self,k,Mh):
        crit_density=2.775e11 # (Msun/h)/(Mpc/h)^3
        r200=( 3.0*Mh/(4.0*pi*200*self.Om0*crit_density) )**(1./3.) # Mpc/h
        c=10.14*(Mh/2e12)**(-0.081)*(1+self.redshift)**(-1.01) # concentration parameter (Duffy+08)
        delta200=200./3.*c**3/(log(1+c)-c/(1+c))
        rs=r200/c
        Si1,Ci1=sici(k*rs)
        Si2,Ci2=sici((1+c)*k*rs)
        NFW=3.*delta200/(200.*c**3)*(sin(k*rs)*(Si2-Si1)+cos(k*rs)*(Ci2-Ci1)
                                     -sin(c*k*rs)/((1+c)*k*rs))
        return NFW
    def galaxy_cross_power_spectrum(self,k, Mmin1, sigma1, Mcut1, Msat1, alpha_s1,
                                            Mmin2, sigma2, Mcut2, Msat2, alpha_s2):
        # halo 1
        HODcen1=self.HOD_cen(self.Mh, Mmin1, sigma1)
        HODsat1=self.HOD_sat(self.Mh, Mmin1, sigma1, Mcut1, Msat1, alpha_s1)
        ngal1=self.n_gal(Mmin1, sigma1, Mcut1, Msat1, alpha_s1)
        # halo 2
        HODcen2=self.HOD_cen(self.Mh, Mmin2, sigma2)
        HODsat2=self.HOD_sat(self.Mh, Mmin2, sigma2, Mcut2, Msat2, alpha_s2)
        ngal2=self.n_gal(Mmin2, sigma2, Mcut2, Msat2, alpha_s2)

        # 1 halo term
        P_1h=simps( self.dndlogMh * ( HODcen1/ngal1 * HODsat2/ngal2*self.NFW(k,self.Mh) +
                                      HODcen2/ngal2 * HODsat1/ngal1*self.NFW(k,self.Mh) +
                                      HODsat1/ngal1*self.NFW(k,self.Mh) * HODsat2/ngal2*self.NFW(k,self.Mh) ),
                    self.logMh )

        Icen1=simps( self.dndlogMh * self.b * HODcen1/ngal1, self.logMh )
        Icen2=simps( self.dndlogMh * self.b * HODcen2/ngal2, self.logMh )
        Isat1=simps( self.dndlogMh * self.b * HODsat1/ngal1*self.NFW(k,self.Mh), self.logMh )
        Isat2=simps( self.dndlogMh * self.b * HODsat2/ngal2*self.NFW(k,self.Mh), self.logMh )

        P_2h=self.Pm(k)*( Icen1*Icen2 + Icen1*Isat2 + Isat1*Icen2 + Isat1*Isat2 ) # nonlinear Pm(k)
        #P_2h=self.Pm(k,self.redshift)*( Icen*Icen + Icen*Isat + Isat*Icen + Isat*Isat )
        return P_1h, P_2h     
       
    def galaxy_power_spectrum(self,k,Mmin, sigma, Mcut, Msat, alpha_s):
        HODcen=self.HOD_cen(self.Mh,Mmin, sigma)
        HODsat=self.HOD_sat(self.Mh,Mmin, sigma, Mcut, Msat, alpha_s)
        ngal=self.n_gal(Mmin, sigma, Mcut, Msat, alpha_s)

        P_1h=simps( self.dndlogMh * ( HODcen/ngal * HODsat/ngal*self.NFW(k,self.Mh) +
                                      HODcen/ngal * HODsat/ngal*self.NFW(k,self.Mh) +
                                      HODsat/ngal*self.NFW(k,self.Mh) * HODsat/ngal*self.NFW(k,self.Mh) ),
                    self.logMh )

        Icen=simps( self.dndlogMh * self.b * HODcen/ngal, self.logMh )
        Isat=simps( self.dndlogMh * self.b * HODsat/ngal*self.NFW(k,self.Mh), self.logMh )

        P_2h=self.Pm(k)*( Icen*Icen + Icen*Isat + Isat*Icen + Isat*Isat ) # nonlinear Pm(k)
        return P_1h, P_2h 

    def galaxy_cross_correlation_function(self,r, Mmin1, sigma1, Mcut1, Msat1, alpha_s1,
                                                  Mmin2, sigma2, Mcut2, Msat2, alpha_s2,ignore_1h=False): 
        """
        Returns the real-space correlation function of galaxies

        Parameters
        ----------
        r : array_like
         radial seperation [Mpc/h]

        """
        N=200
        logkmin=-4.0
        logkmax=+4.0
        k=logspace(logkmin,logkmax,N)
        P_1h=zeros(k.size)
        P_2h=zeros(k.size)
        for i in range(k.size):
            P_1h[i],P_2h[i]=self.galaxy_cross_power_spectrum(k[i], Mmin1, sigma1, Mcut1, Msat1, alpha_s1,
                                                                   Mmin2, sigma2, Mcut2, Msat2, alpha_s2)

        # plt.figure()
        # plt.plot(k,P_1h,label='1h')
        # plt.plot(k,P_2h,label='2h')
        # plt.plot(k,P_1h+P_2h,label='total')
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.legend()
        # plt.show()
        if ignore_1h:
            P_1h = np.zeros(k.size)

        Pk=interp1d(k,P_1h+P_2h,bounds_error=False,fill_value=0.0)

        from hankel import SymmetricFourierTransform
        ft = SymmetricFourierTransform(ndim=3, N = 200, h = 0.03)
        corrfunc=ft.transform(Pk,r,ret_err=False,inverse=True) # int f(k)*sin(kr)/(kr)*k^2*dk/(2*pi^2)
        return corrfunc

    def matter_correlation_function(self,r):
        """
        Returns the real-space correlation function of matter

        Parameters
        ----------
        r : array_like
         radial seperation [Mpc/h]
        """
        from hankel import SymmetricFourierTransform
        ft = SymmetricFourierTransform(ndim=3, N = 200, h = 0.03)
        corrfunc=ft.transform(self.Pm,r,ret_err=False,inverse=True) # int f(k)*sin(kr)/(kr)*k^2*dk/(2*pi^2)
        return corrfunc
    
    def galaxy_angular_cross_correlation_function(self,r_perp,
                                                    Mmin1, sigma1, Mcut1, Msat1, alpha_s1,
                                                    Mmin2, sigma2, Mcut2, Msat2, alpha_s2,
                                                    dv_max = 1000*u.km/u.s,ignore_1h=False):
        from scipy import integrate
        import astropy.units as u
        from scipy.interpolate import UnivariateSpline as _interp1d
        # tabulate and interpolate the cross-correlation function
        r_sample=logspace(log10(0.01),log10(500),2000)
        xi_sample=self.galaxy_cross_correlation_function(r_sample, 
                                                         Mmin1, sigma1, Mcut1, Msat1, alpha_s1,
                                                         Mmin2, sigma2, Mcut2, Msat2, alpha_s2,ignore_1h=ignore_1h)
        xi=interp1d(r_sample,xi_sample)
        # plt.plot(r_sample,xi(r_sample),label='3D')
        # integrate
        r_los_max= (dv_max*(1+self.redshift)/Planck18.H(self.redshift) * Planck18.h).to('Mpc').value # cMpc/h
        acf=zeros(r_perp.size)
        N=1000
        r_los=linspace(0,r_los_max,N)
        for i in range(r_perp.size):
            acf[i] = (1/r_los_max) * simps( xi( sqrt(r_perp[i]**2+r_los**2) ), x=r_los )
            # print(r_perp[i],acf[i])
        # plt.plot(r_perp,acf,label='2D')
        # plt.legend()
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        return acf



    def angular_matter_correlation_function(self,r,approx='None',z=None,selection_function=None):
        if approx=='None':
            from scipy import integrate
            import astropy.units as u
            # get selection function
            norm=simps(selection_function,x=z)
            S=selection_function/norm
            Sz=interp1d(z,S,bounds_error=False,fill_value=0.0)
            # tabulate and interpolate the correlation function
            R=logspace(log10(r.min()),log10(100*r.max()),10000)
            xi_m=self.matter_correlation_function(R)
            xi=interp1d(R,xi_m)
            # define radius
            def radius(r_perp, z1, z2):
                r1=Planck18.comoving_distance(z1).to('Mpc').value*Planck18.h # cMpc/h
                r2=Planck18.comoving_distance(z2).to('Mpc').value*Planck18.h # cMpc/h
                R=sqrt(r_perp**2+(r1-r2)**2)
                return R
            # integral
            acf=zeros(r.size)
            for n in range(r.size):
                N=1000
                z1=linspace(z.min(),z.max(),N)
                integrand=zeros(N)
                for i in range(N):
                    z2=linspace(z.min(),z.max(),N)
                    integrand[i]=simps( Sz(z1[i])*Sz(z2)*xi( radius(r[n],z1[i],z2) ), x=z2 )
                acf[n]=simps( integrand, x=z1 )
                print(r[n],acf[n])
            angular_correlation_function=acf

        return angular_correlation_function




    def angular_galaxy_correlation_function(self,r,Mmin, sigma, Mcut, Msat, alpha_s,approx='Limber',z=None,selection_function=None):
        """
        Returns the anuglar correlation function of galaxies

        Parameters
        ----------
        r : array_like
         angular seperation [Mpc/h]
        Muv_min : scalar
         minimum absolute UV magnitude Muv of sample
        approx : string
         'Limber' for Limber approximation
         'None' for no approximation, i.e. full calcuation
        selection_function : array
         sampled selection function at redshift z
        z: array
         redshifts of sampled selection function
        """
        if approx=='Limber':
            # get galaxy power spectrum & interpolation
            N=100
            logkmin=-5.0
            logkmax=+5.0
            k=logspace(logkmin,logkmax,N)
            P_1h=zeros(k.size)
            P_2h=zeros(k.size)
            for i in range(k.size):
                P_1h[i],P_2h[i]=self.galaxy_power_spectrum(k[i],Mmin, sigma, Mcut, Msat, alpha_s)

                

            Pk=interp1d(k,P_1h+P_2h,bounds_error=False,fill_value=0.0)
            # get selection function
            norm=simps(selection_function,x=z)
            S=selection_function/norm
            RH0=2997.9 # c/H0 Mpc/h, present-day Hubble radius
            drdz=RH0/sqrt(self.Om0*(1+z)**3+self.Ov0)
            selection_factor=simps(S**2*(drdz)**(-1),x=z)
            # Henkel transform to get ACF from power spectrum
            from hankel import HankelTransform    # Hankel Tranformation package
            ht=HankelTransform(nu=0,N=120,h=0.01) # see https://hankel.readthedocs.io
            #ht=HankelTransform(nu=0,N=1000,h=0.01) # see https://hankel.readthedocs.io
            acf=1./(2*pi)*ht.transform(Pk,r,ret_err=False)
            angular_correlation_function=selection_factor*acf

        if approx=='None':
            from scipy import integrate
            import astropy.units as u
            # get selection function
            norm=simps(selection_function,x=z)
            S=selection_function/norm
            Sz=interp1d(z,S,bounds_error=False,fill_value=0.0)
            # tabulate and interpolate the correlation function
            R=logspace(log10(r.min()),log10(100*r.max()),1000)
            xi_g=self.galaxy_correlation_function(R,Mmin, sigma, Mcut, Msat, alpha_s)
            xi=interp1d(R,xi_g)
            # define radius
            def radius(r_perp, z1, z2):
                r1=Planck18.comoving_distance(z1).to('Mpc').value*Planck18.h # cMpc/h
                r2=Planck18.comoving_distance(z2).to('Mpc').value*Planck18.h # cMpc/h
                R=sqrt(r_perp**2+(r1-r2)**2)
                return R
            # integral
            acf=zeros(r.size)
            for n in range(r.size):
                N=500
                z1=linspace(z.min(),z.max(),N)
                integrand=zeros(N)
                for i in range(N):
                    z2=linspace(z.min(),z.max(),N)
                    # print(xi( radius(r[n],z1[i],z2) ))
                    integrand[i]=simps( Sz(z1[i])*Sz(z2)*xi( radius(r[n],z1[i],z2) ), x=z2 )
                acf[n]=simps(integrand, x=z1 )
                print(r[n],acf[n])

            # acf=zeros(r.size)
            # for n in range(r.size):
            #     N=500
            #     z1=linspace(z.min(),z.max(),N)
            #     integrand=zeros(N)
            #     for i in range(N):
            #         z2=linspace(z.min(),z.max(),N)
            #         integrand[i]=simps( Sz(z1[i])*Sz(z2)*xi( radius(r[n],z1[i],z2) ), x=z2 )
            #     acf[n]=simps( integrand, x=z1 )
            #     print(r[n],acf[n])
            # angular_correlation_function=acf

        return acf
    


    def volume_averaged_auto_correlation_function(self, r_min, r_max, Mmin, sigma, Mcut, Msat, alpha_s, z, selection_function=None):
        """
        Returns the volume-averaged auto-correlation function, optimized with vectorized integration
        
        Parameters
        ----------
        r_min, r_max : array_like
            Lower and upper limits of transverse comoving distance [cMpc/h]
        z : array
            Redshift sampling points 
        selection_function : array
            Normalized selection function S(z)
        """
        def V_shell(r_min, r_max, z1, z2):
            # cylindrical shell with inner and outer radii Rmin and Rmax, and height d2 - d1
            d1 = Planck18.comoving_distance(z1).to('Mpc').value*Planck18.h  # cMpc/h
            d2 = Planck18.comoving_distance(z2).to('Mpc').value*Planck18.h  # cMpc/h
            return np.pi * (r_max**2 - r_min**2) * (d2 - d1)
        
        # Maximum line-of-sight distance
        R_max = (Planck18.comoving_distance(z.max())-Planck18.comoving_distance(z.min())).to('Mpc').value*Planck18.h
        
        # Prepare correlation function interpolator
        R_sample = np.logspace(np.log10(0.01), np.log10(R_max*1.5), 1000)
        xi_sample = self.galaxy_correlation_function(R_sample, Mmin, sigma, Mcut, Msat, alpha_s)
        xi_interp = interp1d(R_sample, xi_sample, bounds_error=True, fill_value=0.0)
        
        # Precompute line-of-sight distances
        R_los = Planck18.comoving_distance(z).to('Mpc').value*Planck18.h
        R_los = R_los - R_los[0]
        
        acf = np.zeros_like(r_min)
        
        # Number of radial sampling points
        
        for i in range(r_min.size):
            R_low, R_high = r_min[i], r_max[i]
            V = V_shell(R_low, R_high, z.min(), z.max()) 
            # Create radial sampling points
            R_sample = np.linspace(R_low, R_high, 20)
            # Vectorized computation for all R_sample values
            integrand = np.zeros_like(R_sample)
            
            for j in range(R_sample.size):
                # For each R_perp, compute 3D distances to all z points
                R_perp = R_sample[j]
                R_3d = np.sqrt(R_perp**2 + R_los**2)
                # Integrate along line of sight
                integrand[j] = simps(xi_interp(R_3d), x=R_los)
            
            # Integrate over R (cylindrical coordinates: 2πr dr)
            acf[i] = simps(integrand * 2 * np.pi * R_sample, x=R_sample) / V
        
        return acf

    def volume_averaged_cross_correlation_function(self, r_min, r_max, Mmin1, sigma1, Mcut1, Msat1, alpha_s1, Mmin2, sigma2, Mcut2, Msat2, alpha_s2, dv_max=1000*u.km/u.s, selection_function=None,ignore_1h=False):
        """
        Returns the volume-averaged cross-correlation function, optimized with vectorized integration
        
        Parameters
        ----------
        r_min, r_max : array_like
            Lower and upper limits of transverse comoving distance [cMpc/h]
        z : array
            Redshift sampling points 
        selection_function : array
            Normalized selection function S(z)
        """
        def V_shell(r_min, r_max, r_los_max):

            return np.pi * (r_max**2 - r_min**2) * r_los_max
        
        r_los_max= (dv_max*(1+self.redshift)/Planck18.H(self.redshift) * Planck18.h).to('Mpc').value # cMpc/h
        
        # Prepare correlation function interpolator
        R_sample = np.logspace(np.log10(0.01), np.log10(r_los_max*1.1), 1000)
        xi_sample = self.galaxy_cross_correlation_function(R_sample, Mmin1, sigma1, Mcut1, Msat1, alpha_s1, Mmin2, sigma2, Mcut2, Msat2, alpha_s2,ignore_1h=ignore_1h)
        xi_interp = interp1d(R_sample, xi_sample, bounds_error=True, fill_value=0.0)
        
        N=1000
        r_los=linspace(0,r_los_max,N)
        
        acf = np.zeros_like(r_min)
        
        for i in range(r_min.size):
            R_low, R_high = r_min[i], r_max[i]
            V = V_shell(R_low, R_high, r_los_max) 
            R_sample = np.linspace(R_low, R_high, 50)
            integrand = np.zeros_like(R_sample)
            
            for j in range(R_sample.size):
                R_perp = R_sample[j]
                R_3d = np.sqrt(R_perp**2 + r_los**2)
                # Integrate along line of sight
                integrand[j] = simps(xi_interp(R_3d), x=r_los)
            
            # Integrate over R (cylindrical: 2πr dr)
            acf[i] = simps(integrand * 2 * np.pi * R_sample, x=R_sample) / V
        
        return acf
    

    def galaxy_correlation_function(self,r,Mmin, sigma, Mcut, Msat, alpha_s):
        """
        Returns the real-space correlation function of galaxies

        Parameters
        ----------
        r : array_like
         radial seperation [Mpc/h]

        """
        N=200
        logkmin=-4.0
        logkmax=+4.0
        k=logspace(logkmin,logkmax,N)
        P_1h=zeros(k.size)
        P_2h=zeros(k.size)
        for i in range(k.size):
            P_1h[i],P_2h[i]=self.galaxy_power_spectrum(k[i],Mmin, sigma, Mcut, Msat, alpha_s)
        Pk=interp1d(k,P_1h+P_2h,bounds_error=False,fill_value=0.0)

        from hankel import SymmetricFourierTransform
        ft = SymmetricFourierTransform(ndim=3, N = 200, h = 0.03)
        corrfunc=ft.transform(Pk,r,ret_err=False,inverse=True) # int f(k)*sin(kr)/(kr)*k^2*dk/(2*pi^2)

        return corrfunc
    
    # OIII luminosity function [1/cMpc3]
    def dndlogLOIII(self,LOIII):
        Phi_c=10**(-7.74)  # 1/cMpc3
        Lc=10**(46.94)      # erg/s
        alpha=-2.0                    # fixed
        dndL=Phi_c*(LOIII/Lc)**alpha*np.exp(-LOIII/Lc)/Lc
        dndlogL=np.log(10)*LOIII*dndL
        return dndlogL

    def n_OIII(self,L_lim):
        from scipy.integrate import simps
        logL=np.linspace(np.log10(L_lim),44,1000)
        n_OIII = simps( self.dndlogLOIII(10**logL), x=logL)
        return n_OIII

    def angular_bins(self,min_bin,max_bin,bins,type='linear-bin'):
        if type=='linear-bin':
            theta_edges=np.linspace(min_bin,max_bin,bins+1)
        if type=='log-bin':
            theta_edges=np.logspace(np.log10(min_bin),
                                    np.log10(max_bin),
                                    bins+1)
        theta_bins=np.zeros(bins)
        for i in range(bins):
            theta_bins[i]=(theta_edges[i]+theta_edges[i+1])/2
        return theta_bins, theta_edges



# ## test
if __name__ == "__main__":
    import pandas as pd
    import sys; sys.path.insert(1, '/Users/koki/Projects/HSC-tomography/analysis')
    import HSC_toolkits; hsc=HSC_toolkits.HSC_toolkits()

    redshift=4.89
    [Om0,Ov0,h0,sig8,ns]=[0.3089,0.6911,0.6774,0.8159,0.9667]
    logMmin=11.5
    sigma=0.2
    logMsat=13.0
    logMcut=-99 #-0.5*logMmin #-99 #log10( (10**logMmin)**(-0.5) )
    alpha_s=1.0
    HOD=HOD_FRAMEWORK(Om0,Ov0,h0,sig8,ns,redshift)
    HOD.define_model(logMmin,sigma,logMsat,logMcut,alpha_s)

    # galaxy power spectrum check
    k=logspace(-3,3)
    P1h=zeros(k.size)
    P2h=zeros(k.size)
    for i in range(k.size):
        P1h[i],P2h[i]=HOD.galaxy_power_spectrum(k[i])

    # angular correlation function check
    NB718=hsc.filter(filter='NB718')
    z=NB718['wavelength']/1215.67-1
    selection_function=NB718['filter_transmission']
    r=logspace(-1,2,30) # cMpc/h
    ACF=HOD.angular_galaxy_correlation_function(r,approx='Limber',selection_function=selection_function,z=z)
#    ACF_FULL=HOD.angular_galaxy_correlation_function(r,approx='None',selection_function=selection_function,z=z)

    # plot
    figure(figsize=(10,5))
    subplot(1,2,1)
    loglog(k,k**3*HOD.Pm(k),label='matter')
    loglog(k,k**3*P1h,label='1-halo term')
    loglog(k,k**3*P2h,label='2-halo term')
    xlabel('$k$')
    ylabel('$k^3P(k)$')
    legend()

    subplot(1,2,2)
    loglog(r,ACF,'r-',label='LImber')
#    loglog(r,ACF_FULL,'b-',label='Full')
    df=pd.read_csv('/Users/koki/Projects/HSC-tomography/analysis/results/LAE_clustering_measurement.dat')
    errorbar(df['theta [cMpc/h]'],df['ACF(theta)'],yerr=sqrt(df['Var[ACF(theta)]']),
             fmt='o',capsize=5,ms=10,color='black')
    xlabel('$r_\\perp$ [cMpc/h]')
    ylabel('$\\omega_g(r_\\perp)$')
    legend()
    tight_layout()
    show()