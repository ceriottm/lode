# -*- coding: utf-8 -*-
"""Tests for calculating the spherical projection."""

import numpy as np
from numpy.testing import assert_allclose

from pylode.lib.spherical_harmonics import evaluate_spherical_harmonics


class TestSphericalHarmonics:
    """Test correct behavior of spherical harmonics code"""

    def test_spherical_harmonics(self):
        """Start by evaluating spherical harmonics at some special points"""
        vectors_zdir = np.array([[0, 0, 1], [0, 0, 2]])
        lmax = 8
        coeffs = evaluate_spherical_harmonics(vectors_zdir, lmax)

        # spherical harmonics should be independent of length
        assert np.linalg.norm(coeffs[0] - coeffs[1]) < 1e-14

        # Compare to exact values of Y_lm for vectors in +z-direction
        nonzero_indices = np.array([l**2 + l for l in range(lmax + 1)])
        coeffs_nonzero = coeffs[0, nonzero_indices]
        exact_vals = np.sqrt((2 * np.arange(lmax + 1) + 1) / 4 / np.pi)
        assert np.linalg.norm(coeffs_nonzero - exact_vals) < 1e-14

        # Make sure that all other values are (essentially) zero
        assert abs(np.sum(coeffs[0]**2) - np.sum(exact_vals**2)) < 1e-14

    def test_spherical_harmonics_x_y(self):
        """use vectors confined on x-y plane"""
        rng = np.random.default_rng(3218932)
        N = 10
        lmax = 8
        vectors_xy = np.zeros((N, 3))
        vectors_xy[:, :2] = rng.normal(0, 1, (N, 2))

        # Certain coefficients need to vanish by symmetry
        coeffs = evaluate_spherical_harmonics(vectors_xy, lmax)
        for l in range(lmax + 1):
            for im, m in enumerate(np.arange(-l, l + 1)):
                if l + m % 2 == 1:
                    assert np.linalg.norm(coeffs[:, l**2 + im]) / N < 1e-14

    def test_spherical_harmonics_addition(self):
        """Verify addition theorem of spherical harmonics 
        evaluated at large number of random points """
        N = 1000
        lmax = 8
        rng = np.random.default_rng(3218932)
        vectors = rng.normal(0, 1, (N, 3))
        coeffs = evaluate_spherical_harmonics(vectors, lmax)
        num_coeffs = (lmax + 1)**2
        assert coeffs.shape == (N, num_coeffs)

        # Verify addition theorem
        exact_vals = (2 * np.arange(lmax + 1) + 1) / (4. * np.pi)
        for l in range(lmax + 1):
            prod = np.sum(coeffs[:, l**2:(l + 1)**2]**2, axis=1)
            error = np.linalg.norm(prod - exact_vals[l])
            assert error / N < 1e-15

        # In the limit of infinitely many points, the columns should
        # be orthonormal. Reuse the values from above for a Monte Carlo
        # integration (if this was the sole purpose, there would be much
        # more efficient methods for quadrature)
        innerprod_matrix = coeffs.T @ coeffs / N * np.pi * 4
        difference = innerprod_matrix - np.eye(num_coeffs)
        assert np.linalg.norm(difference) / num_coeffs**2 < 1e-2

    def test_spherical_harmonics_orthogonality(self):
        """
        The spherical harmonics form an orthonormal basis of the
        L2 space of functions defined on the 2-sphere S2.
        In simpler terms, any reasonable function f(theta, phi)
        that only depends on the two spherical angles can be
        expressed as a linear combination of spherical harmonics,
        which need to satisfy certain orthogonality relations that
        are expressed in terms of integrals over the two angles.
        In this test, we perform a numerical integration to verify
        the orthogonality of the used spherical harmonics.
        """
        # Define the two spherical angles over a grid
        N_theta = 200
        N_phi = 200
        phis = np.linspace(0, 2*np.pi, N_phi, endpoint=False)
        thetas = np.arccos(np.linspace(-1,1,N_theta+1, endpoint=True))
        thetas = 0.5 * (thetas[1:] + thetas[:-1])
        #thetas = np.arccos(np.linspace(-1,1,N_theta, endpoint=True))
        phis_2d, thetas_2d = np.meshgrid(phis, thetas, indexing='ij')
        assert phis_2d.shape == (N_phi, N_theta) # first index is for phis

        # Generate unit vectors along the specified directions
        unit_vectors = np.zeros((N_phi, N_theta, 3))
        unit_vectors[:,:,0] = np.cos(phis_2d) * np.sin(thetas_2d)
        unit_vectors[:,:,1] = np.sin(phis_2d) * np.sin(thetas_2d)
        unit_vectors[:,:,2] = np.cos(thetas_2d)

        # Evaluate the real spherical harmonics using the general code
        # from the library
        lmax = 5
        sph_harm_values = np.zeros(((lmax+1)**2, N_phi * N_theta))
        for i, vecs in enumerate(unit_vectors):
            # Pass a 2D array of unit vectors for each fixed value of theta
            sph_harm_values[:,i*N_theta:(i+1)*N_theta] =  evaluate_spherical_harmonics(vecs, lmax=lmax).T

        # Orthogonality matrix:
        prefac = 4 * np.pi / (N_phi * N_theta)
        ortho_matrix = prefac * sph_harm_values @ sph_harm_values.T
        ortho_matrix_ref = np.eye((lmax+1)**2)
        assert_allclose(ortho_matrix, ortho_matrix_ref, atol=1e-2, rtol=1e-2)

    def test_spherical_harmonics_analytical_small_l(self):
        """
        For small values of l=0,1,2 we compare the obtained values
        for the spherical harmonics evaluated at various points with
        the analytical expressions.
        l=0 is mostly a sanity check, while l=1,2 are the first nontrivial
        spherical harmonics of odd / even degree, respectively.
        Perfect agreement of these 1+3+5=9 functions is highly likely to
        catch any potential discrepancies in the used conventions of the
        spherical harmonics.
        """
        # Define the two spherical angles over a grid
        N_theta = 53
        N_phi = 119
        phis = np.linspace(0, 2*np.pi, N_phi)
        thetas = np.arccos(np.linspace(-1,1,N_theta))
        phis_2d, thetas_2d = np.meshgrid(phis, thetas, indexing='ij')
        assert phis_2d.shape == (N_phi, N_theta) # first index is for phis

        # Generate unit vectors along the specified directions
        unit_vectors = np.zeros((N_phi, N_theta, 3))
        unit_vectors[:,:,0] = np.cos(phis_2d) * np.sin(thetas_2d)
        unit_vectors[:,:,1] = np.sin(phis_2d) * np.sin(thetas_2d)
        unit_vectors[:,:,2] = np.cos(thetas_2d)

        # Define exact spherical harmonics
        prefac1 = 1./np.sqrt(4*np.pi)
        prefac2 = np.sqrt(3 / (4 * np.pi))
        prefac3 = np.sqrt(15 / (4 * np.pi))
        prefac20 = np.sqrt(5 / (16 * np.pi))
        prefac21 = np.sqrt(15 / (4 * np.pi))
        prefac22 = np.sqrt(15/ (16 * np.pi))
        cart_x = lambda theta, phi: np.cos(phi) * np.sin(theta)
        cart_y = lambda theta, phi: np.sin(phi) * np.sin(theta)
        cart_z = lambda theta, phi: np.cos(theta)
        Y00 = lambda theta, phi: prefac1 * np.ones_like(theta)
        Y1m = lambda theta, phi: prefac2 * np.sin(theta) * np.sin(phi)
        Y10 = lambda theta, phi: prefac2 * np.cos(theta)
        Y11 = lambda theta, phi: prefac2 * np.sin(theta) * np.cos(phi)
        Y2n = lambda theta, phi: prefac21 * cart_y(theta, phi) * cart_x(theta, phi)
        Y2m = lambda theta, phi: prefac21 * cart_y(theta, phi) * cart_z(theta, phi)
        Y20 = lambda theta, phi: prefac20 * (3 * np.cos(theta)**2 - 1)
        Y21 = lambda theta, phi: prefac21 * cart_x(theta, phi) * cart_z(theta, phi)
        Y22 = lambda theta, phi: prefac22 * (cart_x(theta, phi)**2 - cart_y(theta, phi)**2)

        # Evaluate the real spherical harmonics using the
        # analytical expressions
        sph_harm_exact = np.zeros((9, N_phi, N_theta))
        sph_harm_exact[0] = Y00(thetas_2d, phis_2d)
        sph_harm_exact[1] = Y1m(thetas_2d, phis_2d)
        sph_harm_exact[2] = Y10(thetas_2d, phis_2d)
        sph_harm_exact[3] = Y11(thetas_2d, phis_2d)
        sph_harm_exact[4] = Y2n(thetas_2d, phis_2d)
        sph_harm_exact[5] = Y2m(thetas_2d, phis_2d)
        sph_harm_exact[6] = Y20(thetas_2d, phis_2d)
        sph_harm_exact[7] = Y21(thetas_2d, phis_2d)
        sph_harm_exact[8] = Y22(thetas_2d, phis_2d)

        # Evaluate the real spherical harmonics using the general code
        # from the library
        sph_harm_check = np.zeros_like(sph_harm_exact)
        for i, vecs in enumerate(unit_vectors):
            # Pass a 2D array of unit vectors for each fixed value of theta
            sph_harm_check[:,i] =  evaluate_spherical_harmonics(vecs, lmax=2).T

        # Check agreement of the pyLODE spherical harmonics with the exact values
        assert_allclose(sph_harm_exact, sph_harm_check, rtol=1e-15, atol=6e-16) 