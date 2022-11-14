# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:42:22 2021

@author: kevin
"""

from tempfile import tempdir
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import dblquad
from scipy.special import erf, spherical_jn  # arguments (n,z)
from scipy.special import eval_legendre, gamma, gammainc, hyp1f1, hyp2f1

from .atomic_density import AtomicDensity

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


class RadialBasis:
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
    radial_basis_radius : float
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
                 radial_basis_radius,
                 smearing,
                 radial_basis,
                 subtract_self=False,
                 potential_exponent=1):
        # Store the provided hyperparameters
        self.smearing = smearing
        self.max_radial = max_radial
        self.max_angular = max_angular
        self.radial_basis_radius = radial_basis_radius
        self.radial_basis = radial_basis.lower()
        self.potential_exponent = potential_exponent

        # Store the optional parameters related to the self contribution
        self.subtract_center_contribution = subtract_self

        # Preparation for the extra steps in case the contribution to
        # the density by the center atom is to be subtracted
        self.atomic_density = AtomicDensity(smearing, potential_exponent)
        self.density_function = self.atomic_density.get_atomic_density

        if self.radial_basis not in ["monomial", "gto", "gto_primitive", "gto_analytical", "gto_normalized"]:
            raise ValueError(f"{self.radial_basis} is not an implemented basis"
                              ". Try 'monomial', 'GTO' or GTO_primitive.")


    def compute(self, kmax, Nradial=5000, Nspline=1000):
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
        rcut = self.radial_basis_radius

        self.center_contributions = np.zeros(nmax)

        # Array of k-vectors for numerical integration used
        # for all radial bases
        kk = np.linspace(0, kmax, Nspline)
        projcoeffs = np.zeros((Nspline, nmax, lmax + 1))

        # Start main part: Store all the desired values for the specified
        # radial basis.
        if self.radial_basis in ['gto', 'gto_primitive', 'gto_normalized']:
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
            
            # Define normalizations of GTOs
            normalizations = np.zeros((nmax,))
            for n in range(nmax):
                normalizations[n] = np.sqrt(2 / (sigma[n]**(3+2*n) * gamma(1.5 + n)))
                if self.radial_basis != 'gto_primitive':
                    R_n[n] *= normalizations[n]
            self.normalizations = normalizations

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
            if self.radial_basis in ['gto_primitive', 'gto_normalized']:
                R_n_ortho = R_n

            # Start evaluation of spherical Bessel functions
            for l in range(lmax + 1):
                for ik, k in enumerate(kk):
                    for n in range(nmax):
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


        elif self.radial_basis == 'gto_analytical':
            # Generate length scales sigma_n for R_n(x)
            sigma = np.ones(nmax, dtype=float)
            for i in range(1, nmax):
                sigma[i] = np.sqrt(i)
            sigma *= rcut / nmax

            # Define normalizations of GTOs
            normalizations = np.zeros((nmax,))
            for n in range(nmax):
                normalizations[n] = np.sqrt(2 / (sigma[n]**(3+2*n) * gamma(1.5 + n)))
            self.normalizations = normalizations

            # Compute the inner product matrix from the analytical expression
            innerprods = np.zeros((nmax, nmax))
            for n1 in range(nmax):
                for n2 in range(nmax):
                    neff = (3+n1+n2)/2
                    alpha = 0.5 * (1/sigma[n1]**2 + 1/sigma[n2]**2)
                    innerprods[n1, n2] = alpha**(-neff) * gamma(neff) / 2
                    innerprods[n1, n2] *= normalizations[n1]
                    innerprods[n1, n2] *= normalizations[n2]

            # Diagonalize the inner product matrix
            eigvals, eigvecs = np.linalg.eigh(innerprods)
            transformation = eigvecs @ np.diag(np.sqrt(
                1. / eigvals)) @ eigvecs.T
            self.orthonormalization_matrix = transformation

            # Start evaluation of spherical Bessel functions
            for l in range(lmax + 1):
                for ik, k in enumerate(kk):
                    coeffs = normalizations * np.sqrt(np.pi)
                    for n in range(nmax):
                        neff = 3 + n + l
                        coeffs[n] *= 2**((n-l-1)/2) * k**l
                        coeffs[n] *= sigma[n]**neff * gamma(neff/2) / gamma(1.5+l)
                        coeffs[n] *= hyp1f1(neff/2, 1.5+l, -0.5*k**2*sigma[n]**2)
                    projcoeffs[ik, :, l] = transformation @ coeffs

            # Compute self contribution to the l=0 components
            smearing = self.smearing
            center_contribs = normalizations.copy() # initialize with normalization factors
            if self.potential_exponent == 0:
                # Precompute the global prefactor that does not depend on n
                prefac = np.sqrt(4*np.pi) / (np.pi * smearing**2)**0.75 / 2
                center_contribs *= prefac
                
                # Compute center contributions
                for n in range(nmax):
                    alpha = 0.5*(1/smearing**2 + 1/sigma[n]**2) 
                    center_contribs[n] *= gamma((3+n)/2) / alpha**((3+n)/2)
            
            elif self.potential_exponent == 1:
                # Precompute the global prefactor that does not depend on n
                angular_prefac = np.sqrt(4*np.pi)
                radial_prefac = 1./(np.sqrt(np.pi) * smearing)
                prefac = angular_prefac * radial_prefac 
                center_contribs *= prefac

                # Compute center contributions
                for n in range(nmax):
                    # n-dependent part of "prefactor" (anything apart from hyp2f1)
                    center_contribs[n] *= 2**((2+n)/2)*sigma[n]**(3+n)*gamma((3+n)/2)

                    # hypergeometric function arising from radial integration
                    hyparg = -sigma[n]**2/smearing**2
                    center_contribs[n] *= hyp2f1(0.5, (n+3)/2, 1.5, hyparg)
            
            elif self.potential_exponent in [2,3,4,5,6]:
                p = self.potential_exponent
                prefac = np.sqrt(4*np.pi) / gamma(p/2)
                center_contribs *= prefac
 
                # Compute center contributions
                for n in range(nmax):
                    neff = (3+n)/2
                    center_contribs[n] *= 2**((1+n-p)/2) * smearing**(3+n-p) * 2 * gamma(neff) / p
                    s = smearing**2 / sigma[n]**2
                    hyparg = 1/(1+s)
                    center_contribs[n] *= hyp2f1(1,neff,((p+2)/2),hyparg) * hyparg**neff

            self.center_contributions = transformation @ center_contribs

        elif self.radial_basis == 'monomial':
            if nmax != 1:
                raise ValueError("Only nmax = 1 is allowed for "
                                 "monomial basis.")

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

    def compute_realspace_spline_from_analytical(self, rcut, Nspline=100, smooth_cutoff_width=0.):
        # Define shortcuts for commonly used variables
        nmax = self.max_radial
        lmax = self.max_angular
        # rcut = self.radial_basis_radius
        smearing = self.smearing
        width = smooth_cutoff_width
        transformation = self.orthonormalization_matrix
        ls = np.arange(lmax+1)

        # Define the dimer distances over which to spline
        rmin = 1e-6
        radii = np.linspace(rmin, rcut, Nspline)

        # Auxilary quantity: widths of GTO basis functions
        gto_sigma = np.ones(nmax, dtype=float)
        for i in range(1, nmax):
            gto_sigma[i] = np.sqrt(i)
        gto_sigma *= rcut / nmax

        # Compute radial integrals for each (n,l) for the GTO basis
        integrals = np.zeros((Nspline, nmax, lmax + 1))
        for l in range(lmax+1):
            # Define auxilary quantities and prefactors
            a = 1. / (2 * smearing**2)
            lplus3half = l + 1.5
            prefac_global = np.pi**1.5 * a**l / gamma(lplus3half)
            prefac_global *= radii**l * np.exp(-a*radii**2)
            prefac_global /= (np.pi * smearing**2)**(3/4)

            # Start main loop
            featvec = np.zeros((Nspline, nmax))
            for n in range(nmax):
                nlplus3half = (3 + n + l) / 2
                b = 1. / (2 * gto_sigma[n]**2)
                prefac_n_dep = gamma(nlplus3half) / (a+b)**nlplus3half
                hyp = hyp1f1(nlplus3half, lplus3half, a**2*radii**2/(a+b))
                hyp *= self.normalizations[n] * prefac_n_dep
                featvec[:, n] = hyp
            
            featvec_orthonormal = (transformation @ featvec.T).T

            for n in range(nmax):
                integrals[:, n, l] = prefac_global * featvec_orthonormal[:, n]

        # Generate spline class object
        self.radial_spline_realspace = CubicSpline(radii, integrals)

    def compute_realspace_spline(self, Nspline = 5, smooth_cutoff_width=0.):
        """
        Numerically evaluate the double integral over the radius r and the
        angle theta (or its cosine) appearing in the real space evaluation
        of the projection coefficients and spline the result as a function
        of the neighbor distance rij.
        Warning: Currently, only the monomial basis is supported.
        The density, on the other hand, is arbitrary and can be both
        a Gaussian or a smeared Coulomb density.

        Parameters
        ----------
        Nradii : INT, optional
            Number of nodes to use in the spline
        """
        # Define shortcuts for more readable code
        nmax = self.max_radial
        lmax = self.max_angular
        rcut = self.radial_basis_radius
        smearing = self.smearing
        width = smooth_cutoff_width
        ls = np.arange(lmax+1)

        # Define the dimer distances over which to spline
        rmin = 1e-6
        radii = np.linspace(rmin, rcut, Nspline)

        # If desired, add a smooth cutoff function that results in a
        # continuous behavior of the coefficients as atoms enter or
        # leave the cutoff ball.
        f_smooth = lambda x: 0.5 * np.cos((x-rcut+width)*np.pi/width) + 0.5
        f_cutoff = lambda x: np.where(rcut - x > width, 1, f_smooth(x))

        # Start computing real space evaluation of density
        # contribution for a neighbor atom as a function of the
        # radial distance rij for different l-channels.
        # Note that only the monomial basis is supported.
        """integrals = np.zeros((Nspline, lmax+1))
        for l in ls:
            for ir, rij in enumerate(radii):
                reff = lambda r, c: np.sqrt(r**2+rij**2-2*r*rij*c)
                density = lambda r, c: self.density_function(reff(r, c))
                prefacs = lambda r, c: f_cutoff(rij) * r**(2+l) * eval_legendre(l, c)
                integrand = lambda r, c: prefacs(r,c) * density(r,c)
                integrals[ir, l] = dblquad(integrand, 0, rcut, lambda x: -1, lambda x: 1)
        """

        #################################################
        # General version that works for any radial basis
        #################################################
        integrals = np.zeros((Nspline, nmax, lmax + 1))
        radial_basis = self.radial_basis

        # Define functions used for the integration bounds over
        # the angle theta. In practice, the integral is parametrized
        # in terms of cos(theta) which goes from -1 to 1.
        # Due to the specific syntax of scipy.integrate.dblquad,
        # it is required to pass the bounds (which are fixed numbers)
        # as functions.
        cosmax = lambda x: -1
        cosmin = lambda x: 1

        
        # Generate length scales sigma_n for R_n(x)
        if radial_basis == 'gto':
            sigma = np.ones(nmax, dtype=float)
            for i in range(1, nmax):
                sigma[i] = np.sqrt(i)
            sigma *= rcut / nmax

        # Compute the radial and angular integrals numerically
        # for different values of the pair distance r_ij and
        # store the results in an array to generate the splines later on.
        for ir, rij in enumerate(radii):
            print(f'Radial distance {ir+1} out of {len(radii)}')
            for l in range(lmax+1):
                prefac = lambda r, c: f_cutoff(rij) * r**2 * eval_legendre(l, c)
                dist = lambda r, c: np.sqrt(r**2+rij**2-2*r*rij*c + 1e-13)
                density = lambda r, c: self.density_function(dist(r, c))
                if radial_basis == 'gto':
                    print('Use gto basis')
                    transformation = self.orthonormalization_matrix
                    for n in range(nmax):
                        R_n_prim = lambda r: r**n*np.exp(-0.5*r**2/sigma[n]**2)
                        R_n = lambda r: self.normalizations[n] * R_n_prim(r)
                        integrand = lambda r,c: prefac(r,c)*R_n(r)*density(r,c)
                        integrals[ir, n, l] = dblquad(integrand, 0, rcut, cosmin, cosmax, epsabs=1e-3, epsrel=1e-3)[0]
                    integrals[ir, :, l] = transformation @ integrals[ir, :, l]

                elif radial_basis == 'monomial':
                    print('use monomials')
                    normalization = np.sqrt((3 + 2*l) / (rcut**(3 + 2*l)))
                    R_n = lambda r: normalization * r**l
                    integrand = lambda r,c: prefac(r,c)*R_n(r)*density(r,c)
                    integrals[ir, 0, l] = dblquad(integrand, 0, rcut, lambda x: -1, lambda x: 1)[0]

        # Generate spline class object
        self.radial_spline_realspace = CubicSpline(radii, integrals)




    def compute_realspace_spline_numerically(self, Nspline = 5, smooth_cutoff_width=0., Nradial=100, Ntheta=100):
        """
        Numerically evaluate the double integral over the radius r and the
        angle theta (or its cosine) appearing in the real space evaluation
        of the projection coefficients and spline the result as a function
        of the neighbor distance rij.
        Warning: Currently, only the monomial basis is supported.
        The density, on the other hand, is arbitrary and can be both
        a Gaussian or a smeared Coulomb density.

        Parameters
        ----------
        Nradii : INT, optional
            Number of nodes to use in the spline
        """
        # Define shortcuts for more readable code
        nmax = self.max_radial
        lmax = self.max_angular
        rcut = self.radial_basis_radius
        smearing = self.smearing
        width = smooth_cutoff_width
        ls = np.arange(lmax+1)

        # Define the dimer distances over which to spline
        rmin = 1e-6
        radii = np.linspace(rmin, rcut, Nspline)

        # If desired, add a smooth cutoff function that results in a
        # continuous behavior of the coefficients as atoms enter or
        # leave the cutoff ball.
        f_smooth = lambda x: 0.5 * np.cos((x-rcut+width)*np.pi/width) + 0.5
        f_cutoff = lambda x: np.where(rcut - x > width, 1, f_smooth(x))

        # Start computing real space evaluation of density
        # contribution for a neighbor atom as a function of the
        # radial distance rij for different l-channels.
        # Note that only the monomial basis is supported.
        """integrals = np.zeros((Nspline, lmax+1))
        for l in ls:
            for ir, rij in enumerate(radii):
                reff = lambda r, c: np.sqrt(r**2+rij**2-2*r*rij*c)
                density = lambda r, c: self.density_function(reff(r, c))
                prefacs = lambda r, c: f_cutoff(rij) * r**(2+l) * eval_legendre(l, c)
                integrand = lambda r, c: prefacs(r,c) * density(r,c)
                integrals[ir, l] = dblquad(integrand, 0, rcut, lambda x: -1, lambda x: 1)
        """

        #################################################
        # General version that works for any radial basis
        #################################################
        integrals = np.zeros((Nspline, nmax, lmax + 1))
        radial_basis = self.radial_basis

        
        # Generate length scales sigma_n for R_n(x)
        if radial_basis in ['gto', 'gto_analytical', 'gto_primitive']:
            sigma = np.ones(nmax, dtype=float)
            for i in range(1, nmax):
                sigma[i] = np.sqrt(i)
            sigma *= rcut / nmax

        # Compute the radial and angular integrals numerically
        # for different values of the pair distance r_ij and
        # store the results in an array to generate the splines later on.
        for ir, rij in enumerate(radii):
            print(f'Radial distance {ir+1} out of {len(radii)}')
            for l in range(lmax+1):
                prefac = lambda r, c: f_cutoff(rij) * r**2 * eval_legendre(l, c)
                dist = lambda r, c: np.sqrt(r**2+rij**2-2*r*rij*c + 1e-13)
                density = lambda r, c: self.density_function(dist(r, c))
                if radial_basis == 'gto':
                    print('Use gto basis')
                    transformation = self.orthonormalization_matrix
                    for n in range(nmax):
                        R_n_prim = lambda r: r**n*np.exp(-0.5*r**2/sigma[n]**2)
                        R_n = lambda r: self.normalizations[n] * R_n_prim(r)
                        integrand = lambda r,c: prefac(r,c)*R_n(r)*density(r,c)
                        integrals[ir, n, l] = dblquad(integrand, 0, rcut, cosmin, cosmax, epsabs=1e-3, epsrel=1e-3)[0]
                    integrals[ir, :, l] = transformation @ integrals[ir, :, l]

                elif radial_basis == 'monomial':
                    print('use monomials')
                    normalization = np.sqrt((3 + 2*l) / (rcut**(3 + 2*l)))
                    R_n = lambda r: normalization * r**l
                    integrand = lambda r,c: prefac(r,c)*R_n(r)*density(r,c)
                    integrals[ir, 0, l] = dblquad(integrand, 0, rcut, lambda x: -1, lambda x: 1)[0]

        # Generate spline class object
        self.radial_spline_realspace = CubicSpline(radii, integrals)



