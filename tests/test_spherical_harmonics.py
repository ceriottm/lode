# -*- coding: utf-8 -*-
"""Tests for calculating the spherical projection."""

import numpy as np

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
        np.random.seed(324238)
        N = 10
        lmax = 8
        vectors_xy = np.zeros((N, 3))
        vectors_xy[:, :2] = np.random.normal(0, 1, (N, 2))

        # Certain coefficients need to vanish by symmetry
        coeffs = evaluate_spherical_harmonics(vectors_xy, lmax)
        for l in range(lmax + 1):
            for im, m in enumerate(np.arange(-l, l + 1)):
                if l + m % 2 == 1:
                    assert np.linalg.norm(coeffs[:, l**2 + im]) / N < 1e-14

    def test_spherical_harmonics_addition(self):
        """Verify addition theorem and orthogonality of spherical harmonics 
        evaluated at large number of random points """
        N = 1000
        lmax = 8
        vectors = np.random.normal(0, 1, (N, 3))
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
