# -*- coding: utf-8 -*-
"""

Generate modules to evaluate the spherical harmonics needed for LODE.

"""

import numpy as np
from scipy.special import sph_harm


def convert_to_spherical(vectors):
    """
    Convert an array of 3D vectors into an array containing the spherical
    coordinate angles theta, phi.

    Parameters
    ----------
    vectors : np.ndarray
        Array of shape N x 3, where vectors[i] is the i-th vector which will
        be converted into its spherical coordinate representation.

    Returns
    -------
    theta, phi : np.1darray
        Arrays of size N, where containing the spherical angles.
        If a single vector (array of shape (3,)) is passed, the result will
        be returned as scalars rather than arrays.

    """
    theta = np.arctan2(np.sqrt(vectors[:,0]**2+vectors[:,1]**2), vectors[:,2])
    phi = np.arctan2(vectors[:,1], vectors[:,0])
    return theta, phi


def evaluate_spherical_harmonics(vectors, lmax):
    """
    For a given set of vectors, evaluate all spherical harmonics up to
    maximal frequency lmax.

    Parameters
    ----------
    vectors : np.ndarray
        Array of shape N x 3, where vectors[i] is the i-th vector.
    lmax : INT
        Maximal frequency: all spherical harmonics coefficients Ylm will
        be evaluated up to l=0,1,...,lmax and all |m|<l, leading to a
        total of (lmax+1)**2 coefficients.

    Returns
    -------
    Ylm : np.ndarray
        Array of shape N x (lmax+1)**2 containing the real spherical harmonics
        coefficients, ordered according to the "natural" increasing order:
            (l,m) = (0,0), (1,-1), (1,0), (1,1), (2,-2), (2,-1), ...

    """
    theta, phi = convert_to_spherical(vectors)
    # print('Theta = ', theta)
    # print('phi =   ', phi)
    num_vectors = len(vectors)
    num_coeffs = (lmax+1)**2

    # Evaluate spherical harmonics coefficients
    spherical_harmonics_array = np.zeros((num_vectors, num_coeffs))
    for l in range(lmax+1):
        for im, m in enumerate(np.arange(-l, l+1)):
            Ylm = sph_harm(abs(m), l, phi, theta)
            # Linear combination of Y_l,m and Y_l,-m to create the real form.
            if m < 0:
                Ylm = np.sqrt(2) * (-1)**abs(m) * Ylm.imag
            elif m > 0:
                Ylm = np.sqrt(2) * (-1)**m * Ylm.real
            elif m == 0:
                Ylm = Ylm.real
            spherical_harmonics_array[:,l**2+im] = Ylm

    return spherical_harmonics_array
