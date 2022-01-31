# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:42:22 2021

@author: kevin
"""

import numpy as np
from scipy.special import spherical_jn # arguments (n,z)
from scipy.interpolate import CubicSpline
from scipy import integrate


# Compute the inner product of two functions defined over [0,rcut]
# using the inner product derived from the spherical integral
def innerprod(xx, yy1, yy2):
    # Generate the integrand according to int_0^inf x^2*f1(x)*f2(x)
    integrand = xx * xx * yy1 * yy2
    return integrate.simpson(integrand, xx)


def radial_projection_lode(lmax, rcut, kmax, Nradial=1000,
                           Nspline=200, plot_test=False):
    """
    Obtain the projection coefficients of the spherical Bessel functions
    onto the monomial basis r^l, which is the optimal radial basis for
    a 1/r (Coulombic) LODE

    Parameters
    ----------
    lmax : INT
        Angular cutoff. For a given lmax, the lmax+1 values l=0,1,2,...,lmax
        are used.
    rcut : FLOAT
        Cutoff radius defining the domain of the local density.
    kmax : FLOAT
        Wave vector cutoff, all k-vectors with |k| < kmax are used in the
        Fourier space summation. Often chosen to be pi/sigma, where
        sigma is the width of the Gaussian smearing for good convergence.
    Nsplie : INT, optional
        Number of values in which to partition domain. The default is 100.
    Nradial : INT, optional
        Number of nodes to use in the numerical integration

    Returns
    -------
    Spline function that takes in k-vectors (one or many) and returns
    the projections of the spherical Bessel function j_l(kr) onto the
    orthonormalized radial basis consisting of functions of the form r^l.

    """
    # Initialization of the arrays in which to store function values
    xx = np.linspace(0, rcut, Nradial)
    normalizationsq = np.array([rcut**(2*l + 3)/(2*l + 3) for l in range(lmax+1)])
    kk = np.linspace(0, kmax, Nspline)
    projcoeffs = np.zeros((Nspline, lmax+1))

    # Evaluate the target function and generate spline approximation
    for l in range(lmax+1):
        for ik, k in enumerate(kk):
            bessel = spherical_jn(l, k*xx)
            projcoeffs[ik, l] = innerprod(xx, xx**l, bessel) / normalizationsq[l]
    spline = CubicSpline(kk, projcoeffs)

    # Testing the splines:
    if plot_test:
        import matplotlib.pyplot as plt
        plt.figure()
        splinefits = spline(kk)
        fiterror = np.linalg.norm(projcoeffs - splinefits) # should be zero
        assert fiterror < 1e-14, "Error in splining"
        for l in range(lmax+1):
            plt.plot(kk, projcoeffs[:,l], '-', label=f'target l={l}')
            plt.plot(kk, splinefits[:,l], '--', label=f'spline l={l}')
            if l > 2: break
        plt.legend()

    return spline


def radial_projection_gto(lmax, nmax, rcut, kmax, Nradial=1000,
                           Nspline=200, primitive=False):
    """
    Obtain the projection coefficients of the spherical Bessel functions
    onto the GTO basis, which is used for comparison with the real space
    implementation.

    Parameters
    ----------
    lmax, nmax : INT
        Angular and radial cutoff. The used values of (l,n) are given by:
        n = 0,1,...,nmax-1 and l=0,1,2,...,lmax.
        Note that nmax n-values but lmax+1 l-values are used.
    rcut : FLOAT
        Cutoff radius defining the domain of the local density.
    kmax : FLOAT
        Wave vector cutoff, all k-vectors with |k| < kmax are used in the
        Fourier space summation. Often chosen to be pi/sigma, where
        sigma is the width of the Gaussian smearing for good convergence.
    Nsplie : INT, optional
        Number of values in which to partition domain. The default is 100.
    Nradial : INT, optional
        Number of nodes to use in the numerical integration

    Returns
    -------
    Spline function that takes in k-vectors (one or many) and returns
    the projections of the spherical Bessel function j_l(kr) onto the
    orthonormalized GTO radial basis.

    """
    # Generate length scales sigma_n for R_n(x)
    sigma = np.ones(nmax, dtype=float)
    for i in range(1,nmax):
        sigma[i] = np.sqrt(i)
    sigma *= rcut/nmax

    # Define primitive GTO-like radial basis functions
    f_gto = lambda n,x: x**n*np.exp(-0.5*(x/sigma[n])**2)
    xx = np.linspace(0,rcut*2.5,Nradial)
    R_n = np.array([f_gto(n,xx) for n in range(nmax)])

    # Orthonormalize
    innerprods = np.zeros((nmax,nmax))
    for i in range(nmax):
        for j in range(nmax):
            innerprods[i,j] = innerprod(xx, R_n[i], R_n[j])
    eigvals, eigvecs = np.linalg.eigh(innerprods)
    transformation = eigvecs @ np.diag(np.sqrt(1./eigvals)) @ eigvecs.T
    R_n_ortho = transformation @ R_n

    if primitive: R_n_ortho = R_n # only used for testing

    # Start evaluation of spherical Bessel functions
    kk = np.linspace(0, kmax, Nspline)
    projcoeffs = np.zeros((Nspline, nmax, lmax+1))
    for l in range(lmax+1):
        for n in range(nmax):
            for ik, k in enumerate(kk):
                bessel = spherical_jn(l, k*xx)
                projcoeffs[ik, n, l] = innerprod(xx, R_n_ortho[n], bessel)
    spline = CubicSpline(kk, projcoeffs)

    return spline
