from scipy.integrate import romberg, quad

from pylab import *
import numpy as np

class basic_cosmology_tookits:
    def __init__(self,h0,om_m0,om_v0,sig8,ns):

        # cosmological parameters
        self.h=h0
        self.om_m0=om_m0
        self.om_v0=om_v0
        self.sig8=sig8
        self.ns=ns

        # Normalisation to linear power spectrum
        integrand=lambda k: ((k**3)*self.P_dummy(k,0.)/(2.*pi**2))*self.W2(k*8.,'tophat')/k
        normfactor=quad(integrand,0.0,inf,limit=500)[0]
        self.Ap=self.sig8**2/normfactor

    def E(self,z): # E(z) function H(z)=H0*E(z)
        E=sqrt(self.om_m0*(1.+z)**3+self.om_v0)
        return E
    def Omega_v(self,z):
        Omega_v=self.om_v0/(self.E(z)**2)
        return Omega_v
    def Omega_m(self,z):
        Omega_m=self.om_m0*(1.+z)**3/(self.E(z)**2)
        return Omega_m
    def W2(self,x,filter): # Window function, x=kR
        if filter == 'gauss':
            W=exp(-0.5*x*x)
        elif filter == 'sharp-k':
            if (1.0-x) > 0.0 :
                W=1.0
            elif (1.0-x) == 0.0 :
                W=0.5
            else:
                W=0.0
        elif filter == 'tophat':
            W=3./(x**3)*(sin(x)-x*cos(x))
        elif filter == 'tophat+cutoff':
            W_TH=3./(x**3)*(sin(x)-x*cos(x))
            W_EXP=(1.0+(0.4*x)**2)**(-2)
            W=W_TH*W_EXP
        elif filter == 'gauss2':
            W=exp(-0.5*x*x)**2
        elif filter == 'powerlaw':
            W=1./(1.+x*x)
        else:
            print('Window function is not speficied.')
        W2=W*W
        return W2
    # Carroll, Press, Turner 1992 fit to growth factor g=D/a
    def growth_factor(self,z):
        om_mz=self.Omega_m(z)
        om_vz=self.Omega_v(z)
        growth_factor=(5.*om_mz/2.)/(om_mz**(4./7.)-om_vz+(1.+om_mz/2.)*(1.+om_vz/70.))
        return growth_factor
    # growth rate D(z)
    def growth_rate(self,z):
        growth_rate=self.growth_factor(z)/(1.+z)
        return growth_rate
    # BBKS transfer function: k in [h/cMpc]
    def transfer_func(self,k):
        q=k/(self.om_m0*self.h)
        transfer_func=log(1.+2.34*q)/(2.34*q)*(1.+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**(-1./4.)
        return transfer_func
    # dummy linear power spectrum at z=0 with Ap=1 unit normalisation
    def P_dummy(self,k,z):
        P_dummy=(self.growth_rate(z)**2)*(self.transfer_func(k)**2)*(k**self.ns)
        return P_dummy
    # linear power spectrum
    def P_linear(self,k,z):
        """
        Returns the linear power spectrum, P_lin(k,z) [Mpc/h)^3]

        Parameters
        ----------
        k : array_like
         wavenumber [h/Mpc]
        z : scalar
         redshift

        Returns
        -------
        result : array
         linear power spectrum P_lin(k,z) [Mpc/h)^3]

        Examples
        --------
        >>> # fiducial cosmology (h,Om0,Ov0,sig8,ns)=(0.7,0.3,0.7,0.8,0.9)
        >>> basic=basic_cosmology_tookits(0.7,0.3,0.7,0.8,0.9)
        >>> N=100
        >>> redshift=6.0
        >>> k=np.logspace(-2,2,N)
        >>> P_lin=basic.P_linear(k,redshift)

        """
        P_linear=self.Ap*self.P_dummy(k,z)
        return P_linear
    # linear dimensionless power spectrum
    def Delta2(self,k,z):
        """
        Returns the dimensionless linear power spectrum,
        Delta^2(k)=k^3*P_lin(k,z)/(2*pi^2)

        See Also
        --------
        P_linear : linear power spectrum

        """
        Delta2=k**3*self.P_linear(k,z)/(2.*pi**2)
        return Delta2
    # Peacock & Dodds fit: nonlinear dimensionless power spectrum
    def Delta2_NL(self,kL,z):
        """
        Returns the non-linear dimensionless power spectrum, Delta2_NL(k,z)
        using Peacock&Dodds(1996) fitting function

        Parameters
        ----------
        kL : array_like
         linear wavenumber [h/Mpc]
        z : scalar
         redshift

        Returns
        -------
        k_NL : array
         non-linear wavelength [h/Mpc]
        result : array
         non-linear dimensionless power spectrum Delta2_NL(k,z)

        Examples
        --------
        >>> # fiducial cosmology (h,Om0,Ov0,sig8,ns)=(0.7,0.3,0.7,0.8,0.9)
        >>> basic=basic_cosmology_tookits(0.7,0.3,0.7,0.8,0.9)
        >>> N=100
        >>> redshift=6.0
        >>> kL=np.logspace(-2,2,N)
        >>> kNL,Delta2NL=basic.Delta2_NL(kL,redshift)

        """
        def PD96(x,z,neff):
            A=0.482*(1.+neff/3.)**(-0.947)
            B=0.226*(1.+neff/3.)**(-1.778)
            alpha=3.310*(1.+neff/3.)**(-0.244)
            beta=0.862*(1.+neff/3.)**(-0.287)
            V=11.55*(1.+neff/3.)**(-0.423)
            PD96=x*( (1+B*beta*x+(A*x)**(alpha*beta)) /
                     (1+((A*x)**alpha*self.growth_factor(z)**3/(V*x**0.5))**beta))**(1./beta)
            return PD96
        k=0.5*kL
        k2=1.1*k
        k1=0.9*k
        neff=log(self.P_linear(k2,z)/self.P_linear(k1,z))/log(k2/k1)
        Delta2_NL=PD96(self.Delta2(kL,z),z,neff)
        k_NL=kL*(1.+Delta2_NL)**(1./3.)
        return k_NL, Delta2_NL
    # rms fluctuation of gaussian random field and its moments
    def sigma2(self,M,z):
        kmin=1.0e-2      # minimum k [h/Mpc]
        kmax=1.0e5      # maximum k [h/Mpc]
        #kcutoff=1.e5     # wavenumber cutoff [h/Mpc]
        #kcutoff=1.e30    # wavenumber cutoff [h/Mpc] (no-cutoff option)
        rho_crit=2.775e11 # present critical density [(Msun/h)/(Mpc/h)^3]
        delta_c=1.686
        rho_m=self.om_m0*rho_crit
        R=((3.*M)/(4.*pi*rho_m))**(1./3.)
        integrand=lambda lnk : (self.Delta2(exp(lnk),z)*self.W2(exp(lnk)*R,'tophat'))
        #*(exp(-(exp(lnk)/kcutoff)**2))**2)
        #sigma2=quad(integrand,log(kmin),log(kmax),limit=500)[0]
        # sigma2=romberg(integrand,log(kmin),log(1./R)+3,divmax=10,rtol=1.e-4)
        sigma2=romberg(integrand,log(kmin),log(1./R)+3)

        return sigma2
    def bias(self,M,z,type='Tinker2010'):
        bias=[]
        #M=asarray([M])
        for m in M:
            delta_c=1.686
            nu=delta_c/sqrt(self.sigma2(m,z))
            if type=='Tinker2010':
                b=1.-nu**0.1325/(nu**0.1325+1.0716)+0.1830*nu**1.5+0.2652*nu**2.4
            if type=='Mo1996':
                b=1.+(nu**2-1.)/delta_c
            bias.append(b)
        return array(bias)
    def halo_mass_function(self,M,z,type='Tinker2010',output_type='dndlogM'):
        """
        Returns the halo mass function, dn/dlog10(M) [1/(Mpc/h)^3],
        where M [Msun/h] is halo mass

        Parameters
        ----------
        M : array
         halo mass [Msun/h]
        z : scalar
         redshift
        type : {'Tinker2010', 'Press-Schechter'}, optional
         fitting function or analytic function. Default is 'Tinker2010'
        output_type : {'dndlogM', 'dndlnM', 'dndM'}, optional
         defition of output halo mass function
         dndlogM [1/(Mpc/h)^3] (per log_10 mass bin), Default
         dndlnM [1/(Mpc/h)^3] (per ln mass bin)
         dndM [1/(Mpc/h)^3/(Msun/h)] (per mass)

        Returns
        -------
        result : array
         halo mass function, dn/dlog10(M)
         output_type can be changed. see output_type

        Examples
        --------
        >>> # fiducial cosmology (h,Om0,Ov0,sig8,ns)=(0.7,0.3,0.7,0.8,0.9)
        >>> basic=basic_cosmology_tookits(0.7,0.3,0.7,0.8,0.9)
        >>> N=100
        >>> redshift=6.0
        >>> M=np.logspace(8,14,N)
        >>> dndlogM=basic.halo_mass_function(M,redshift)


        """

        def f(nu,z):
            def f_dummy(nu,z):
                eta1= 0.589*(1.+z)**( 0.20)
                eta2=-0.729*(1.+z)**(-0.08)
                eta3=-0.243*(1.+z)**( 0.27)
                eta4= 0.864*(1.+z)**(-0.01)
                f=(1.+(eta1*nu)**(-2.*eta2))*nu**(2*eta3)*exp(-eta4*nu**2/2.)
                return f
            def b(nu):
                b=1.-nu**0.1325/(nu**0.1325+1.0716)+0.1830*nu**1.5+0.2652*nu**2.4
                return b
            if z<=3:
                integrand=lambda nu: b(nu)*f_dummy(nu,z)
            if z>3:  # Tinker+2010 recommendation, Section 4
                integrand=lambda nu: b(nu)*f_dummy(nu,3)
            norm=quad(integrand,0,inf)[0]
            f=f_dummy(nu,z)/norm
            return f

        delta_c=1.686
        rho_crit=2.775e11 # present critical density [(Msun/h)/(Mpc/h)^3]
        rho_m=self.om_m0*rho_crit

        sigM=zeros(M.size)
        for i in range(M.size):
            sigM[i]=sqrt(self.sigma2(M[i],z))

        halo_mass_function=zeros(M.size)
        for i in range(M.size-1):
            nu=delta_c/sqrt(self.sigma2(M[i],z))
            dlnsigdlnM=log(sigM[i+1]/sigM[i])/log(M[i+1]/M[i])
            if output_type=='dndM':
                halo_mass_function[i]=rho_m/(M[i]*M[i])* nu*f(nu,z) *abs(dlnsigdlnM)
            elif output_type=='dndlnM':
                halo_mass_function[i]=rho_m/M[i]* nu*f(nu,z) *abs(dlnsigdlnM)
            elif output_type=='dndlogM':
                halo_mass_function[i]=log(10.)*rho_m/M[i]* nu*f(nu,z) *abs(dlnsigdlnM)
        return halo_mass_function

#### TEST ####
if __name__ == "__main__":

    # Planck 15 cosmology
    Om0=0.3089
    Ov0=1.-Om0
    h0=0.6774
    sig8=0.8159
    ns=0.9667
    cosmo=basic_cosmology_tookits(h0,Om0,Ov0,sig8,ns)

    kL=logspace(-3,3)
    figure()
    loglog(kL,cosmo.Delta2(kL,6.0),'b-')

    k_NL,Delta2NL=cosmo.Delta2_NL(kL,6.0)
    loglog(k_NL,Delta2NL,'r:')

    #loading necessary data
    log10M_h_reed,log10dn=genfromtxt('Reed_z6.mf',usecols=(0,1),unpack=True)
    log10M_h=log10M_h_reed[::-1]


    M=logspace(8,14,100)
    figure()
    loglog(M*h0,cosmo.halo_mass_function(M,6.),'k-')
    loglog(10**log10M_h,10**log10dn[::-1],'r:')
    xlim(M.min(),M.max())
    ylim(1e-15,1e5)

    # bias function test
    M=1e11
    z=linspace(0,10)
    b=zeros(z.size)
    for i in range(z.size):
        b[i]=cosmo.bias(M,z[i],type='Tinker2010')

    figure()
    plot(z,b,'r')
    ylim(0,7)
    xlim(0,8)

    show()
