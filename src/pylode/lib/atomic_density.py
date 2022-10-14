# Class storing all information related to the atomic density

import numpy as np
from scipy.special import erf, gamma, gammainc
from scipy.integrate import quad

# Return the upper incomplete Gamma function Gamma(+3/2, x)
# This is used in the recursive evaluation of Gamma(-3/2), Gamma(-1/2) etc.
# which are in turn required to define the Fourier transformed density for
# general exponents
def gammainc_upper_three_half(xx):
    res = np.zeros_like(xx)
    integrand = lambda t: t**0.5 * np.exp(-t)
    for ix, x in enumerate(xx):
        res[ix] = quad(integrand, x, np.inf, epsrel=1e-12, epsabs=1e-12)[0]
    return res

# From a numerical point of view, it is unstable to evaluate
# the incomplete Gamma functions directly since these have a singularity
# at x=0 for a<0. Thus, we generate a function that returns Gamma(a,x)/x^a instead,
# which removes the singularity at the origin.
def incomplete_gamma_over_powerlaw(x):
    res = 2/3*np.exp(-x)*(1 - 2*x - 4*x**2)
    res += 8/3*x**(3/2) * gammainc_upper_three_half(x)
    return res

# Define the Fourier transform of the density for p=6
def density_dispersion(k, smearing):
    peff = -3/2
    p = 6
    prefac = np.pi**1.5 * 4**peff / gamma(p/2)
    prefac *= (smearing**2/2)**peff
    return prefac * incomplete_gamma_over_powerlaw(0.5*smearing**2*k**2)

# Numerical implementation of upper incomplete Gamma function
# using the integral definition that also works for negative
# first arguments.
def gammainc_upper_numerical(n, zz):
    """
    Implement upper incomplete Gamma function
    """
    yy = np.zeros_like(zz)
    integrand = lambda x: x**(n-1) * np.exp(-x)
    for iz, z in enumerate(zz):
        yy[iz] = quad(integrand, z, np.inf)[0]
    return yy

# Fourier transform of general smeared 1/r^p potential
def density_fourierspace(p, k, smearing):
    peff = 3-p
    prefac = np.pi**1.5 * 2**peff / gamma(p/2)
    return prefac * gammainc_upper_numerical(peff/2, 0.5*(k*smearing)**2) / k**peff

# Auxilary function to evaluate real space density
def gammainc_over_power(a, zz, cutoff=1e-5):
    """
    Compute gammainc(a,zz) / z^a, where gammainc is the lower incomplete gamma function
    """
    # Make sure all inputs are nonnegative
    assert (zz>=0).all()
    assert a > 0
    
    # Initialization
    yy = np.zeros_like(zz)
    
    # Split input values into those that are very close to zero (close to the singularity)
    # and the remaining part.
    idx_small = zz < cutoff
    idx_large = zz >= cutoff
       
    # Evaluate the function using the usual expression
    yy[idx_large] = gammainc(a, zz[idx_large]) / zz[idx_large]**a * gamma(a)
    
    # For small z close to the singularity, use the asymptotic expansion
    yy[idx_small] = 1/a - zz[idx_small]/(a+1) + zz[idx_small]**2/(2*(a+2))
    
    return yy

class AtomicDensity():
    """
    Class that contains all the necessary information to specify
    the type of atomic density (Gaussian, Dirac delta, Coulombic, etc.)
    used in a descriptor.
    In particular, the class also contains information about the
    respective Fourier transforms.
    """
    def __init__(self, smearing, potential_exponent):
        self.smearing = smearing
        self.potential_exponent = potential_exponent

    def get_atomic_density(self, xx):
        """
        Return the real space atomic density evaluated at the points
        contained in the numpy array xx.
        Note that even for a single input value, xx has to be an array
        for this to work.
        """
        assert type(xx) == np.ndarray
        smearing = self.smearing

        # Gaussian density with L2 normalization, i.e.
        # a Gaussian normalized such that int f^2 d^3x = 1.
        if self.potential_exponent == 0:
            prefac = 1 / (np.pi * smearing**2)**(3/4)
            return prefac * np.exp(-0.5 * xx**2 / smearing**2)
 
        # Smeared 1/r^p density
        elif self.potential_exponent in [1, 2, 3, 4, 5, 6]:
            zz = 0.5 * xx**2 / smearing**2
            peff = self.potential_exponent / 2
            prefac = 1./(2*smearing**2)**peff / gamma(peff)
            return prefac * gammainc_over_power(peff, zz)

    def get_fourier_transform(self, kk):
        # Fourier transform of density times Fourier transform of potential
        # This is the line where using Gaussian or 1/r^p for different p are
        # distinguished
        smearing = self.smearing

        # Gaussian density
        if self.potential_exponent == 0:
            prefac = (4 * np.pi * self.smearing**2)**(3/4)
            return prefac * np.exp(-0.5 * (kk*self.smearing)**2)

        # Smeared Coulomb density
        elif self.potential_exponent == 1:
            prefac = 4 * np.pi
            return prefac * np.exp(-0.5 * (kk*self.smearing)**2) / kk**2

        # Smeared dispersion density
        elif self.potential_exponent == 6:
            return density_dispersion(kk, self.smearing)

        elif self.potential_exponent in [2,3,4,5,6]:
            peff = 3-self.potential_exponent
            p = self.potential_exponent
            prefac = np.pi**1.5 * 2**peff / gamma(p/2)
            return prefac * gammainc_upper_numerical(peff/2, 0.5*(kk*smearing)**2) / kk**peff

    # Get the (real space) density function g(r) evaluated at zero.
    def get_density_at_zero(self):
        return self.get_atomic_density(np.array([0]))[0]
    
    # Get the Fourier transform of the atomic density evaluated at zero.
    def get_fourier_transform_at_zero(self):
        smearing = self.smearing 
        if self.potential_exponent == 0:
            return (4 * np.pi * smearing**2)**(3/4)

        elif self.potential_exponent in [1, 2, 3, 4, 5, 6]:
            p = self.potential_exponent
            peff = (3-p) / 2
            return - np.pi**1.5 * (2*smearing**2)**peff / gamma(p/2) / peff