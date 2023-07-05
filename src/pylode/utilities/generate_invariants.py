#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Generate invariants from density projection coefficients.

The code is also able to combine descriptors of different types, e.g.
short range descriptors based on Gaussian densities + LODE to
generate multiscale descriptors.
"""
import numpy as np


def generate_degree2_invariants(coeffs):
    """
    Generate degree 2 invariants from density projection coefficients.

    Parameters
    ----------
    coeffs : array[num_envs, num_species, nmax1, (lmax+1)**2]
        Density projection coefficients. This could include the spherical
        expansion coefficients in librascal terminology, or the coefficients
        obtained from pyLODE.

    Returns
    -------
    Array of degree 2 invariants

    """

    # Get the hyperparameters from the shape of the features
    # pyLODE features have indices features[i,a,n,lm], where
    # i is the index of the atom (within the whole data set)
    # a is the chemical species
    # n is the radial index
    # lm is one index for the pair (l,m), l=0,1,...,lmax and |m|<=l
    num_env, num_species, num_n, num_lm = coeffs.shape
    lmax = int(np.round(np.sqrt(lmmax))) - 1
    
    # Prepare array for invariants
    # If both inputs are different, use all possible species and radial 
    # combinations.
    num_radial_inv = (num_n * (num_n+1))//2
    num_species_inv = num_species**2 
    deg2_invariants = np.zeros((num_env, num_species_inv,
                                num_radial_inv, lmax+1))
    
    # Start generating invariants
    radial_idx = 0
    for ia1 in range(num_species):
        for ia2 in range(num_species):
            species_idx = ia1 * num_species + ia2
            for in1 in range(num_n):
                for in2 in range(in1,num_n):
                    for l in range(lmax+1):
                        vec1 = coeffs[:,ia1,in1,l**2:(l+1)**2]
                        vec2 = coeffs[:,ia2,in2,l**2:(l+1)**2]
                        inv = np.sum(vec1 * vec2, axis=1) / np.sqrt(2*l+1) 
                        deg2_invariants[:, species_idx, radial_idx, l] = inv
                        
                    radial_idx += 1

    return deg2_invariants
    

def generate_degree2_invariants_from_different(coeffs1, coeffs2):
    """
    Generate degree 2 invariants from density projection coefficients.

    Parameters
    ----------
    coeffs1 : array[num_envs, num_species, nmax1, (lmax+1)**2]
        Density projection coefficients. This could include the spherical
        expansion coefficients in librascal terminology, or the coefficients
        obtained from pyLODE.
    coeffs2 : array[num_envs, num_species, nmax2, (lmax+1)**2]
        A second set of coefficients, as for coeffs1. Note that nmax1 and nmax2
        do not need to agree, but all other dimensions need to be the same.

    Returns
    -------
    Array of degree 2 invariants

    """
    # Make sure the shapes match
    # Since it is allowed to use a different nmax for both inputs,
    # there are no checks for nmax.
    num_env, num_species, nmax1, lmmax = coeffs1.shape
    assert num_env == coeffs2.shape[0]
    assert num_species == coeffs2.shape[1]
    assert lmmax == coeffs2.shape[3]
    lmax = int(np.round(np.sqrt(lmmax))) - 1
    nmax2 = coeffs2.shape[2]
    
    # Prepare array for invariants
    # If both inputs are different, use all possible species and radial 
    # combinations.
    num_radial_inv = nmax1 * nmax2
    num_species_inv = num_species**2 
    deg2_invariants = np.zeros((num_env, num_species_inv,
                                num_radial_inv, lmax+1))
    
    # Start generating invariants
    for ia1 in range(num_species):
        for ia2 in range(num_species):
            species_idx = ia1 * num_species + ia2
            for in1 in range(nmax1):
                for in2 in range(nmax2):
                    for l in range(lmax+1):
                        radial_idx = in1 * nmax2 + in2
                        vec1 = coeffs1[:,ia1,in1,l**2:(l+1)**2]
                        vec2 = coeffs2[:,ia2,in2,l**2:(l+1)**2]
                        inv = np.sum(vec1 * vec2, axis=1) / np.sqrt(2*l+1) 
                        deg2_invariants[:, species_idx, radial_idx, l] = inv
    
    return deg2_invariants