# Class storing all information related to the atomic density

import numpy as np
from scipy.special import erf, gamma, gammainc, expi, erfc
from scipy.integrate import quad

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
    assert a > 0

    if type(zz) == np.ndarray:
        # Make sure all inputs are nonnegative
        assert (zz>=0).all()
        
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

    else: # zz is a single float variable
        if zz < cutoff:
            return 1/a - zz/(a+1) + zz**2/(2*(a+2))
        else:
            return gammainc(a, zz) / zz**a * gamma(a)


# Auxilary function for stable Fourier transform implementation
def gammainc_upper_over_powerlaw(p, zz):
    if p==2:
        return np.sqrt(np.pi/zz) * erfc(np.sqrt(zz))
    elif p==3:
        return -expi(-zz)
    elif p==4:
        return 2*(np.exp(-zz) - np.sqrt(np.pi*zz)*erfc(np.sqrt(zz)))
    elif p==5:
        return np.exp(-zz) + zz*expi(-zz)
    elif p==6:
        return ((2-4*zz)*np.exp(-zz) + 4*np.sqrt(np.pi)*zz**1.5*erfc(np.sqrt(zz)))/3

class AtomicDensity:
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

        # Fourier transform of general smeared 1/r^p potential
        elif self.potential_exponent in [2,3,4,5,6]:
            p = self.potential_exponent
            peff = 3-p
            prefac = np.pi**1.5 / gamma(p/2) * (2*smearing**2)**(peff/2)
            zz = 0.5*smearing**2*kk**2
            #prefac = np.pi**1.5 * 2**peff / gamma(p/2)
            #    return prefac * gammainc_upper_numerical(peff/2, 0.5*(kk*smearing)**2) / kk**peff
            return prefac * gammainc_upper_over_powerlaw(p, zz) 

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
