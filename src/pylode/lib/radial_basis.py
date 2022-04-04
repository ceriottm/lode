# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:42:22 2021

@author: kevin
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import erf, spherical_jn  # arguments (n,z)

try:
    from scipy.integrate import simpson
except ImportError:  # scipy <= 1.5.4
    from scipy.integrate import simps as simpson


def innerprod(xx, yy1, yy2):
    """
    Compute the inner product of two radially symmetric functions.

    Uses the inner product derived from the spherical integral without 
    the factor of 4pi. Use simpson integration.

    Generates the integrand according to int_0^inf x^2*f1(x)*f2(x)
    """
    integrand = xx * xx * yy1 * yy2
    return simpson(integrand, xx)


class RadialBasis():
    """
    Class for precomputing and storing all results related to the radial basis.
    
    These include:
    * The splined projection of the spherical Bessel functions onto
      radial basis used in the k-space implementation of LODE
    * The exact values of the center contributions
    * The transformation matrix between the orthogonalized and primitive
      radial basis (if applicable).

    All the needed splines that only depend on the hyperparameters
    are prepared as well by storing the values.

    Parameters
    ----------
    max_radial : int
        Number of radial functions
    max_angular : int
        Number of angular functions
    cutoff_radius : float
        Environment cutoff (Å)
    smearing : float
        Smearing of the Gaussain (Å). Note that computational cost scales
        cubically with 1/smearing.
    radial_basis : str
        The radial basis. Currently implemented are
        'GTO_primitive', 'GTO', 'monomial'.
        For monomial: Only use one radial basis r^l for each angular 
        channel l leading to a total of (lmax+1)^2 features.
    exclude_center : bool
        Exclude contribution from the central atom.
    density_function : callable
        TODO

    Attributes
    ----------
    radial_spline : scipy.interpolate.CubicSpline instance
        Spline function that takes in k-vectors (one or many) and returns
        the projections of the spherical Bessel function j_l(kr) onto the
        specified basis.
    center_contributions : array
        center_contributions
    orthonormalization_matrix : array
        orthonormalization_matrix
    """
    def __init__(self,
                 max_radial,
                 max_angular,
                 cutoff_radius,
                 smearing,
                 radial_basis,
                 subtract_self=False,
                 potential_exponent=0):
        # Store the provided hyperparameters
        self.smearing = smearing
        self.max_radial = max_radial
        self.max_angular = max_angular
        self.cutoff_radius = cutoff_radius
        self.radial_basis = radial_basis.lower()

        # Store the optional parameters related to the self contribution
        self.subtract_center_contribution = subtract_self

        # Preparation for the extra steps in case the contribution to
        # the density by the center atom is to be subtracted
        if self.subtract_center_contribution:
            if potential_exponent == 0:
                prefac = 1./np.power(2*np.pi*self.smearing**2,1.5)
                density = lambda x: prefac * np.exp(-0.5*x**2/self.smearing)
            elif potential_exponent == 1:
                lim = np.sqrt(2./np.pi) / self.smearing
                density = lambda x: np.nan_to_num(erf(x/self.smearing/np.sqrt(2))/x,
                                                  nan=lim, posinf=lim)
            else:
                raise ValueError("Potential exponent has to be one of 0 or 1!")
            self.density_function = density

        if self.radial_basis not in ["monomial", "gto", "gto_primitive"]:
            raise ValueError(f"{self.radial_basis} is not an implemented basis"
                              ". Try 'monomial', 'GTO' or GTO_primitive.")

    def compute(self, kmax, Nradial=1000, Nspline=200):
        """
        Obtain the projection coefficients of the spherical Bessel functions
        onto the chosen basis.

        Parameters
        ----------
        kmax : FLOAT
            Wave vector cutoff, all k-vectors with |k| < kmax are used in the
            Fourier space summation. Often chosen to be pi/sigma, where
            sigma is the width of the Gaussian smearing for good convergence.
        Nsplie : INT, optional
            Number of values in which to partition domain. The default is 100.
        Nradial : INT, optional
            Number of nodes to use in the numerical integration
        """
        # Define shortcuts for more readable code
        nmax = self.max_radial
        lmax = self.max_angular
        rcut = self.cutoff_radius

        self.center_contributions = np.zeros(nmax)

        # Array of k-vectors for numerical integration used
        # for all radial bases
        kk = np.linspace(0, kmax, Nspline)
        projcoeffs = np.zeros((Nspline, nmax, lmax + 1))

        # Start main part: Store all the desired values for the specified
        # radial basis.
        if self.radial_basis in ['gto', 'gto_primitive']:
            # Generate length scales sigma_n for R_n(x)
            sigma = np.ones(nmax, dtype=float)
            for i in range(1, nmax):
                sigma[i] = np.sqrt(i)
            sigma *= rcut / nmax

            # Define primitive GTO-like radial basis functions
            f_gto = lambda n, x: x**n * np.exp(-0.5 * (x / sigma[n])**2)
            xx = np.linspace(0, rcut * 2.5, Nradial)
            R_n = np.array([f_gto(n, xx)
                            for n in range(nmax)])  # nmax x Nradial

            # Orthonormalize
            innerprods = np.zeros((nmax, nmax))
            for i in range(nmax):
                for j in range(nmax):
                    innerprods[i, j] = innerprod(xx, R_n[i], R_n[j])
            eigvals, eigvecs = np.linalg.eigh(innerprods)
            transformation = eigvecs @ np.diag(np.sqrt(
                1. / eigvals)) @ eigvecs.T
            R_n_ortho = transformation @ R_n
            self.orthonormalization_matrix = transformation

            # For testing purposes, allow usage of nonprimitive
            # radial basis.
            if self.radial_basis == 'gto_primitive':
                R_n_ortho = R_n

            # Start evaluation of spherical Bessel functions
            for l in range(lmax + 1):
                for n in range(nmax):
                    for ik, k in enumerate(kk):
                        bessel = spherical_jn(l, k * xx)
                        projcoeffs[ik, n, l] = innerprod(
                            xx, R_n_ortho[n], bessel)

            # Compute self contribution to the l=0 components
            if self.subtract_center_contribution:
                prefac = np.sqrt(4 * np.pi)
                density = self.density_function(xx)
                for n in range(nmax):
                    self.center_contributions[n] = prefac * innerprod(
                        xx, R_n_ortho[n], density)

        elif self.radial_basis == 'monomial':
            assert nmax == 1

            # Initialization of the arrays in which to store function values
            xx = np.linspace(0, rcut, Nradial)
            normalization = np.sqrt(np.array(
                [(2*l + 3) / rcut**(2 * l + 3) for l in range(lmax + 1)]))
            self.orthonormalization_matrix = np.diag(normalization)

            # Evaluate the target function and generate spline approximation
            for l in range(lmax + 1):
                for ik, k in enumerate(kk):
                    bessel = spherical_jn(l, k * xx)
                    projcoeffs[ik, 0, l] = innerprod(
                        xx, xx**l, bessel) * normalization[l]

            # Compute self contribution to the l=0 components
            if self.subtract_center_contribution:
                density = self.density_function(xx)
                prefac = np.sqrt(4 * np.pi)
                self.center_contributions[0] = prefac * innerprod(
                    xx, np.ones_like(xx), density) * normalization[0]

        self.radial_spline = CubicSpline(kk, projcoeffs)
