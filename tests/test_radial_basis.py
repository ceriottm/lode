# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:32:41 2022

@author: kevin
"""

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import gamma, hyp1f1, erf, hyp2f1
from scipy.integrate import quad
from pylode.lib.radial_basis import innerprod, RadialBasis


class TestRadialProjection:
    """Test correct behavior of radial projection code that generates the
    orthonormalized radial basis functions and the splined projections
    of the spherical Bessel functions onto them.
    """
    rcut = 5
    nmax = 10
    lmax = 6  # polynomials can become unstable for large exponents, so this should stay small

    def test_radial_projection(self):
        """Inner product: Make sure that the implementation of inner products works correctly."""
        Nradial = 100000
        xx = np.linspace(0, self.rcut, Nradial)
        for i in range(self.lmax):
            for j in range(i, self.lmax):
                monomialprod = innerprod(xx, xx**i, xx**j)
                exponent = i + j + 2
                assert abs(monomialprod - self.rcut**(exponent + 1) /
                           (exponent + 1)) < 1e-5

    def test_radial_projection_gto_orthogonalization(self):
        """GTO: Since the code only returnes the splines, we rerun a copied
        code fragment which is identical to the main one to test the
        orthogonality of the obtained radial basis.
        Generate length scales sigma_n for R_n(x)"""
        nmax = 8
        Nradial = 1000
        sigma = np.ones(nmax, dtype=float)
        for i in range(1, nmax):
            sigma[i] = np.sqrt(i)
        sigma *= self.rcut / nmax

        # Define primitive GTO-like radial basis functions
        f_gto = lambda n, x: x**n * np.exp(-0.5 * (x / sigma[n])**2)
        xx = np.linspace(0, self.rcut * 2.5, Nradial)
        R_n = np.array([f_gto(n, xx) for n in range(nmax)])

        # Orthonormalize
        innerprods = np.zeros((nmax, nmax))
        for i in range(nmax):
            for j in range(nmax):
                innerprods[i, j] = innerprod(xx, R_n[i], R_n[j])
        eigvals, eigvecs = np.linalg.eigh(innerprods)
        transformation = eigvecs @ np.diag(np.sqrt(1. / eigvals)) @ eigvecs.T
        R_n_ortho = transformation @ R_n

        # start computing overlap
        overlap = np.zeros((nmax, nmax))
        for i in range(nmax):
            for j in range(nmax):
                overlap[i, j] = innerprod(xx, R_n_ortho[i], R_n_ortho[j])
        ortho_error = np.eye(nmax) - overlap
        assert np.linalg.norm(ortho_error) / nmax**2 < 1e-7

    def test_radial_projection_gto_exact(self):
        """Compare GTO projection coefficients with exact value"""
        prefac = np.sqrt(np.pi)
        lmax = 5
        nmax = 8
        rcut = 5.
        sigma = np.ones(nmax, dtype=float)
        for i in range(1, nmax):
            sigma[i] = np.sqrt(i)
        sigma *= rcut / nmax
        smearing = 1.5
        kmax = np.pi / smearing
        Neval = 562  # choose number different from Nspline
        kk = np.linspace(0, kmax, Neval)
        radial_basis = RadialBasis(nmax,
                                   lmax,
                                   rcut,
                                   smearing,
                                   radial_basis="gto_primitive")
        radial_basis.compute(kmax, Nspline=200, Nradial=1000)
        coeffs = radial_basis.radial_spline(kk)

        # Compare to analytical results
        factors = prefac * np.ones((nmax, lmax + 1))
        coeffs_exact = np.zeros((Neval, nmax, lmax + 1))
        for l in range(lmax + 1):
            for n in range(nmax):
                i1 = 0.5 * (3 + n + l)
                i2 = 1.5 + l
                factors[n, l] *= 2**(
                    0.5 *
                    (n - l - 1)) * gamma(i1) / gamma(i2) * sigma[n]**(2 * i1)
                coeffs_exact[:, n, l] = factors[n, l] * kk**l * hyp1f1(
                    i1, i2, -0.5 * (kk * sigma[n])**2)

        error = coeffs - coeffs_exact
        assert np.linalg.norm(error) / error.size < 1e-6

    
    def test_center_contribution_gto_gaussian(self):
        # Define hyperparameters
        nmax = 6
        lmax = 2
        rcut = 5.
        sigma = 1.0
        radial_basis = 'gto_primitive'

        # Define density function and compute center contributions
        prefac = 1./np.power(2*np.pi*sigma**2,1.5)
        density = lambda x: prefac * np.exp(-0.5*x**2/sigma)
        radproj = RadialBasis(nmax, lmax, rcut, sigma,
                              radial_basis, True, density)
        radproj.compute(np.pi/sigma, Nradial=10000)
        center_contr = radproj.center_contributions 

        # Analytical evaluation of center contributions
        normalization = 1./np.sqrt(2*np.pi*sigma**2)**3
        sigma_radial = np.ones(nmax, dtype=float)
        for i in range(1,nmax):
            sigma_radial[i] = np.sqrt(i)
        sigma_radial *= rcut/nmax

        center_contr_analytical = np.zeros((nmax))
        for n in range(nmax):
            sigmatempsq = 1./(1./sigma**2 + 1./sigma_radial[n]**2)
            neff = 0.5 * (3 + n)
            center_contr_analytical[n] = (2*sigmatempsq)**neff * gamma(neff)
        
        center_contr_analytical *= normalization * 2 * np.pi / np.sqrt(4*np.pi)

        # Numerical evaluation of center contributions
        center_contr_numerical = np.zeros((nmax))
        for n in range(nmax):
            Rn = lambda r: r**n * np.exp(-0.5*r**2/sigma_radial[n]**2)
            integrand = lambda r: np.sqrt(4 * np.pi) * Rn(r) * density(r) * r**2
            center_contr_numerical[n] = quad(integrand, 0., np.inf)[0]

        # Check that the three methods agree with one another
        assert_allclose(center_contr, center_contr_analytical, rtol=5e-10)
        assert_allclose(center_contr_numerical, center_contr_analytical, rtol=1e-11)


    def test_center_contribution_gto_longrange(self):
        # Define hyperparameters
        nmax = 6
        lmax = 2
        rcut = 5.
        sigma = 1.0
        radial_basis = 'gto_primitive'

        # Analytical evaluation of center contributions
        normalization = 1./np.sqrt(2*np.pi*sigma**2)**3
        sigma_radial = np.ones(nmax, dtype=float)
        for i in range(1,nmax):
            sigma_radial[i] = np.sqrt(i)
        sigma_radial *= rcut/nmax

        # Define density function and compute center contributions
        lim = np.sqrt(2./np.pi) / sigma
        V_sr = lambda x: lim*(1. - (x/sigma/np.sqrt(2))**2/3)
        V_lr = lambda x: erf(x/sigma/np.sqrt(2))/x
        density = lambda x: np.where(x>1e-5, V_lr(x), V_sr(x))

        # Analytical evaluation of center contributions
        center_contr_analytical = np.zeros((nmax))
        for n in range(nmax):
            prefac = 2**(2+n/2) * sigma_radial[n]**(n+3) / sigma
            neff = 0.5 * (3 + n)
            arg = -(sigma_radial[n] / sigma)**2
            center_contr_analytical[n] = hyp2f1(0.5,neff,1.5,arg) * gamma(neff)
            center_contr_analytical[n] *= prefac

        with np.errstate(divide='ignore', invalid='ignore'):
            radproj_lr = RadialBasis(nmax, lmax, rcut, sigma,
                                radial_basis, True, density)
            radproj_lr.compute(np.pi/sigma, Nradial=2000)
            center_contr = radproj_lr.center_contributions

            # Numerical evaluation of center contributions
            center_contr_numerical = np.zeros((nmax))
            for n in range(nmax):
                Rn = lambda r: r**n * np.exp(-0.5*r**2/sigma_radial[n]**2)
                integrand = lambda r: np.sqrt(4 * np.pi) * Rn(r) * density(r) * r**2
                center_contr_numerical[n] = quad(integrand, 0., np.inf)[0]

            # The three methods of computation should all agree 
            assert_allclose(center_contr, center_contr_analytical, rtol=2e-7)
            assert_allclose(center_contr_numerical, center_contr_analytical, rtol=1e-14)