# -*- coding: utf-8 -*-
"""Tests for calculating the radial projection."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import erf, gamma, hyp1f1, hyp2f1
import pytest

from pylode.lib.radial_basis import RadialBasis, innerprod


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
        sigma = 0.8
        radial_basis = 'gto_primitive'

        # Define density function and compute center contributions
        prefac = 1 / (np.pi*sigma**2)**(3/4)
        density = lambda x: prefac * np.exp(-0.5*x**2/sigma**2)
        radproj = RadialBasis(nmax, lmax, rcut, sigma,
                              radial_basis, True, potential_exponent=0)
        radproj.compute(np.pi/sigma, Nradial=10000)
        center_contr = radproj.center_contributions 

        # Analytical evaluation of center contributions
        normalization = 1./(np.pi*sigma**2)**(3/4)
        sigma_radial = np.ones(nmax, dtype=float)
        for n in range(1,nmax):
            sigma_radial[n] = np.sqrt(n)
        sigma_radial *= rcut/nmax
        
        # Define center coefficients
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
        assert_allclose(center_contr_numerical, center_contr_analytical, rtol=1e-11)
        assert_allclose(center_contr, center_contr_analytical, rtol=1e-9)

    def test_center_contribution_gto_longrange(self):
        # Define hyperparameters
        nmax = 6
        lmax = 2
        rcut = 5.
        sigma = 1.0
        radial_basis = 'gto_primitive'

        # Analytical evaluation of center contributions
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

        radproj = RadialBasis(nmax, lmax, rcut, sigma,
                            radial_basis, True, potential_exponent=1)
        radproj.compute(np.pi/sigma, Nradial=2000)
        center_contr = radproj.center_contributions

        # Numerical evaluation of center contributions
        center_contr_numerical = np.zeros((nmax))
        for n in range(nmax):
            Rn = lambda r: r**n * np.exp(-0.5*r**2/sigma_radial[n]**2)
            integrand = lambda r: np.sqrt(4 * np.pi) * Rn(r) * density(r) * r**2
            center_contr_numerical[n] = quad(integrand, 0., np.inf)[0]

        # The three methods of computation should all agree 
        assert_allclose(center_contr, center_contr_analytical, rtol=2e-7)
        assert_allclose(center_contr_numerical, center_contr_analytical, rtol=1e-14)

    smearings = [0.5, 1.2]
    nmaxs = [5, 6]
    rcuts = [0.1, 5]
    @pytest.mark.parametrize("smearing", smearings)
    @pytest.mark.parametrize("rcut", rcuts)
    @pytest.mark.parametrize("nmax", nmaxs)
    def test_exact_center_contribution_gto_gaussian(self, smearing, rcut, nmax):
        # Generate length scales sigma_n for R_n(x)
        sigma = np.ones(nmax, dtype=float)
        for i in range(1, nmax):
            sigma[i] = np.sqrt(i)
        sigma *= rcut / nmax
        
        # Precompute the global prefactor that does not depend on n
        prefac = np.sqrt(4*np.pi) / (np.pi * smearing**2)**0.75 / 2
        
        # Compute center contributions
        center_contribs = np.ones((nmax,)) * prefac
        for n in range(nmax):
            alpha = 0.5*(1/smearing**2 + 1/sigma[n]**2) 
            center_contribs[n] *= gamma((3+n)/2) / alpha**((3+n)/2)

        # Center contributions from pyLODE
        lmax = 0 # irrelevant for this code, but we still need to provide it
        radial_basis_hypers = {'max_radial':nmax,
                            'max_angular':lmax,
                            'radial_basis_radius':rcut,
                            'smearing':smearing,
                            'radial_basis':'gto_primitive',
                            'subtract_self':True,
                            'potential_exponent':0}
        radial_basis = RadialBasis(**radial_basis_hypers)
        radial_basis.compute(kmax=1)
        center_contribs_pylode = radial_basis.center_contributions

        # Test agreement between the coefficients
        assert_allclose(center_contribs, center_contribs_pylode, rtol=2e-4)

    @pytest.mark.parametrize("smearing", smearings)
    @pytest.mark.parametrize("rcut", rcuts)
    @pytest.mark.parametrize("nmax", nmaxs)
    def test_exact_center_contribution_gto_coulomb(self, smearing, rcut, nmax):
        # Generate length scales sigma_n for R_n(x)
        sigma = np.ones(nmax, dtype=float)
        for i in range(1, nmax):
            sigma[i] = np.sqrt(i)
        sigma *= rcut / nmax
        
        # Precompute the global prefactor that does not depend on n
        angular_prefac = np.sqrt(4*np.pi)
        radial_prefac = 1./(np.sqrt(np.pi) * smearing)
        prefac = angular_prefac * radial_prefac 
        
        # Compute center contributions
        center_contribs = np.ones((nmax,)) * prefac
        for n in range(nmax):
            # n-dependent part of "prefactor" (anything apart from hyp2f1)
            center_contribs[n] *= 2**((2+n)/2)*sigma[n]**(3+n)*gamma((3+n)/2)

            # hypergeometric function arising from radial integration
            hyparg = -sigma[n]**2/smearing**2
            center_contribs[n] *= hyp2f1(0.5, (n+3)/2, 1.5, hyparg)

        # Center contributions from pyLODE
        lmax = 0 # irrelevant for this code, but we still need to provide it
        radial_basis_hypers = {'max_radial':nmax,
                            'max_angular':lmax,
                            'radial_basis_radius':rcut,
                            'smearing':smearing,
                            'radial_basis':'gto_primitive',
                            'subtract_self':True,
                            'potential_exponent':1}
        radial_basis = RadialBasis(**radial_basis_hypers)
        radial_basis.compute(kmax=1)
        center_contribs_pylode = radial_basis.center_contributions

        # Test agreement between the coefficients
        assert_allclose(center_contribs, center_contribs_pylode, rtol=1e-5)

    @pytest.mark.parametrize("smearing", smearings)
    @pytest.mark.parametrize("rcut", rcuts)
    @pytest.mark.parametrize("nmax", nmaxs)
    @pytest.mark.parametrize("p", [1,2,3,4,5,6])
    def test_exact_center_contribution_gto_genexp(self, smearing, rcut, nmax, p):
        # Generate length scales sigma_n for R_n(x)
        sigma = np.ones(nmax, dtype=float)
        for i in range(1, nmax):
            sigma[i] = np.sqrt(i)
        sigma *= rcut / nmax

        # Precompute the global prefactor that does not depend on n
        prefac = np.sqrt(4*np.pi) / gamma(p/2)
    
        # Compute center contributions
        center_contribs = prefac * np.ones((nmax,))
        for n in range(nmax):
            neff = (3+n)/2
            center_contribs[n] *= 2**((1+n-p)/2) * smearing**(3+n-p) * 2 * gamma(neff) / p
            s = smearing**2 / sigma[n]**2
            hyparg = 1/(1+s)
            center_contribs[n] *= hyp2f1(1,neff,((p+2)/2),hyparg) * hyparg**neff

        # Center contributions from pyLODE
        lmax = 0 # irrelevant for this code, but we still need to provide it
        radial_basis_hypers = {'max_radial':nmax,
                            'max_angular':lmax,
                            'radial_basis_radius':rcut,
                            'smearing':smearing,
                            'radial_basis':'gto_primitive',
                            'subtract_self':True,
                            'potential_exponent':p}
        radial_basis = RadialBasis(**radial_basis_hypers)
        radial_basis.compute(kmax=1)
        center_contribs_pylode = radial_basis.center_contributions

        # Test agreement between the coefficients
        assert_allclose(center_contribs, center_contribs_pylode, rtol=1e-5)

    # The GTOs are implemented in pyLODE using two different methods.
    # The first one uses the general implementation that works for any
    # radial basis, evaluating the radial basis on a grid and then using
    # quadratures for the subsequent integrals.
    # The second one uses the analytical expressions instead, both for
    # the center contributions as well as 
    @pytest.mark.parametrize("sigma", smearings)
    @pytest.mark.parametrize("rcut", rcuts)
    @pytest.mark.parametrize("nmax", nmaxs)
    @pytest.mark.parametrize("lmax", [1])
    @pytest.mark.parametrize("p", [0, 1, 3, 6])
    def test_agreement_gto_with_analytical_center(self, sigma, rcut, nmax, p, lmax):
        radproj_numerical = RadialBasis(nmax, lmax, rcut, sigma, 'gto', True, potential_exponent=p)
        radproj_numerical.compute(2 * np.pi/sigma, Nradial=2000)
        center_contrib_numerical = radproj_numerical.center_contributions
        
        radproj_analytical = RadialBasis(nmax, lmax, rcut, sigma, 'gto_analytical', True, potential_exponent=p)
        radproj_analytical.compute(2 * np.pi/sigma, Nradial=2000)
        center_contrib_analytical = radproj_analytical.center_contributions

        assert_allclose(center_contrib_numerical, center_contrib_analytical, rtol=2e-4)

    # Same tests as above, but for the splines.
    @pytest.mark.parametrize("sigma", smearings)
    @pytest.mark.parametrize("rcut", rcuts)
    @pytest.mark.parametrize("nmax", nmaxs)
    @pytest.mark.parametrize("lmax", [1])
    @pytest.mark.parametrize("p", [0, 1, 3, 6])
    def test_agreement_gto_with_analytical_spline(self, sigma, rcut, nmax, p, lmax):
        radproj_numerical = RadialBasis(nmax, lmax, rcut, sigma, 'gto', True, potential_exponent=p)
        radproj_numerical.compute(2 * np.pi/sigma, Nradial=2000)
        spline_numerical = lambda x: radproj_numerical.radial_spline(x)
        
        radproj_analytical = RadialBasis(nmax, lmax, rcut, sigma, 'gto_analytical', True, potential_exponent=p)
        radproj_analytical.compute(2 * np.pi/sigma, Nradial=2000)
        spline_analytical = lambda x: radproj_analytical.radial_spline(x)

        # Evaluate the splines across the entire domain
        kk = np.linspace(0, 2*np.pi/sigma, 500)
        yy_numerical = spline_numerical(kk)
        yy_analytical = spline_analytical(kk)
        assert_allclose(yy_numerical, yy_analytical, rtol=3e-4, atol=2e-5)

    def test_center_contribution_monomial_longrange(self):
        # Define hyperparameters
        nmax = 1
        lmax = 2
        rcut = 5.
        sigma = 1.0
        radial_basis = 'monomial'

        # Define density function
        lim = np.sqrt(2./np.pi) / sigma
        V_sr = lambda x: lim*(1. - (x/sigma/np.sqrt(2))**2/3)
        V_lr = lambda x: erf(x/sigma/np.sqrt(2))/x
        density = lambda x: np.where(x>1e-5, V_lr(x), V_sr(x))

        # Get center contributions from Radial Basis class
        radproj = RadialBasis(nmax, lmax, rcut, sigma,
                                 radial_basis, True, potential_exponent=1)
        center_contr = 0.
        radproj.compute(np.pi/sigma, Nradial=2000)
        center_contr = radproj.center_contributions
        assert center_contr.shape == (1,)

        # Analytical evaluation of center contribution
        f = lambda x: 0.25 * ((2*x**2-1) * erf(x) + 2/np.sqrt(np.pi)*x*np.exp(-x**2))
        arg = rcut / sigma / np.sqrt(2)
        center_contr_analytical = 4*sigma**2 * np.sqrt(3*np.pi/rcut**3) * f(arg)

        # Numerical evaluation of center contribution
        Rn = lambda r: np.sqrt(3/rcut**3) * np.ones_like(r)
        integrand = lambda r: np.sqrt(4 * np.pi) * Rn(r) * density(r) * r**2
        center_contr_numerical = quad(integrand, 0., rcut)[0]
        
        # Run checks
        assert abs(center_contr_analytical - center_contr[0]) < 6e-10        
        assert abs(center_contr_analytical - center_contr_numerical) < 1e-11