# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:32:41 2022

@author: kevin
"""
# Functions / classes to test
from kvec_generator import Kvector_Generator
from spherical_harmonics import evaluate_spherical_harmonics
from radial_projection import innerprod, radial_projection_gto
from projection_coeffs import Density_Projection_Calculator

# Generic imports
import numpy as np
from scipy.special import gamma, hyp1f1
from ase import Atoms
from matplotlib import pyplot as plt

# Run the tests
def main():
    # Test kvector generator
    print('Start testing kvector generator')
    run_test_kvecgen()    
      
    # Test correct behavior of spherical harmonics generator
    print('\nStart testing spherical harmonics')
    run_test_spherical_harmonics()  
      
    # Test radial projection
    print('\nStart testing radial projection')
    run_test_radial_projection()
    
    # Test LODE implementation
    print('\nStart testing LODE implementation')
    run_test_lode()
    
    
# Test correct behavior of k-vector generation.
# The tests used here are essentially the same the first two tests implemented
# in librascal.
def run_test_kvecgen():
    # Test 1: use simple cubic cell for which the solution is known exactly
    cell = np.eye(3)
    eps = 1e-2
    cutoffs = [1+eps, np.sqrt(2)-eps, np.sqrt(2)+eps, np.sqrt(3)-eps, np.sqrt(3)+eps, 2-eps, np.sqrt(5)-eps]
    num_vectors_correct = [3, 3, 9, 9, 13, 13, 16, 16]
    for ik, kcut in enumerate(cutoffs):
        # Generate k vectors
        kvecgen = Kvector_Generator(cell, kcut, is_reciprocal_cell=True)
        kvectors = kvecgen.get_kvectors()
        kvecnorms = kvecgen.get_kvector_norms()
        
        # Check whether number of obtained vectors agrees with exact result
        assert kvecgen.get_kvector_number() == len(kvectors)
        assert len(kvectors) == num_vectors_correct[ik]
        
        # Check that the obtained normes are indeed the norms of the
        # corresponding k-vectors and that they lie in the cutoff ball
        assert (kvecnorms == np.linalg.norm(kvectors, axis=1)).all()
        assert (kvecnorms < kcut).all()
    print(' - Test 1 = cubic cell: passed')     
    
    # Test 2: use triclinic input vectors (describing the same cell as above)
    cell = np.array([[1,0,0],[5,1,0],[3,4,1]])
    for ik, kcut in enumerate(cutoffs):
        # Generate k vectors
        kvecgen = Kvector_Generator(cell, kcut, is_reciprocal_cell=True)
        kvectors = kvecgen.get_kvectors()
        kvecnorms = kvecgen.get_kvector_norms()
        
        # Check whether number of obtained vectors agrees with exact result
        assert kvecgen.get_kvector_number() == len(kvectors)
        assert len(kvectors) == num_vectors_correct[ik]
        
        # Check that the obtained normes are indeed the norms of the
        # corresponding k-vectors and that they lie in the cutoff ball
        assert (kvecnorms == np.linalg.norm(kvectors, axis=1)).all()
        assert (kvecnorms < kcut).all()
    print(' - Test 2 = triclinic cells: passed') 
    
    # Test 3: use rotated cell
    # Generate random rotation matrix
    np.random.seed(12419)
    import scipy
    Q, R = scipy.linalg.qr(np.random.normal(0,1,(3,3)))
    assert np.linalg.norm(Q.T @ Q - np.eye(3)) < 1e-13
    cell = np.array([[1,0,0],[5,1,0],[3,4,1]]) @ Q
    for ik, kcut in enumerate(cutoffs):
        # Generate k vectors
        kvecgen = Kvector_Generator(cell, kcut, is_reciprocal_cell=True)
        kvectors = kvecgen.get_kvectors()
        kvecnorms = kvecgen.get_kvector_norms()
        
        # Check whether number of obtained vectors agrees with exact result
        assert kvecgen.get_kvector_number() == len(kvectors)
        assert len(kvectors) == num_vectors_correct[ik]
        
        # Check that the obtained normes are indeed the norms of the
        # corresponding k-vectors and that they lie in the cutoff ball
        assert (kvecnorms < kcut).all()
    print(' - Test 3 = rotated cells: passed')    
    
    return 0


# Test correct behavior of spherical harmonics code
def run_test_spherical_harmonics():
    # Test 1: Start by evaluating spherical harmonics at some special points
    vectors_zdir = np.array([[0,0,1],[0,0,2]])
    lmax = 8
    coeffs = evaluate_spherical_harmonics(vectors_zdir, lmax)

    # spherical harmonics should be independent of length
    assert np.linalg.norm(coeffs[0]-coeffs[1]) < 1e-14

    # Compare to exact values of Y_lm for vectors in +z-direction
    nonzero_indices = np.array([l**2+l for l in range(lmax+1)])
    coeffs_nonzero = coeffs[0,nonzero_indices]
    exact_vals = np.sqrt((2*np.arange(lmax+1)+1)/4/np.pi)
    assert np.linalg.norm(coeffs_nonzero - exact_vals) < 1e-14

    # Make sure that all other values are (essentially) zero
    assert abs(np.sum(coeffs[0]**2) - np.sum(exact_vals**2)) < 1e-14
    print(' - Test 1 = Special values along z-axis: passed')

    # Test 2: use vectors confined on x-y plane
    np.random.seed(324238)
    N = 10
    lmax = 8
    vectors_xy = np.zeros((N,3))
    vectors_xy[:,:2] = np.random.normal(0, 1, (N,2))
    
    # Certain coefficients need to vanish by symmetry
    coeffs = evaluate_spherical_harmonics(vectors_xy, lmax)
    for l in range(lmax+1):
        for im, m in enumerate(np.arange(-l, l+1)):
            if l+m %2 == 1:
                assert np.linalg.norm(coeffs[:,l**2+im]) / N < 1e-14
    print(' - Test 2 = Special values in x-y plane: passed')

    # Test 2: Verify addition theorem and orthogonality of spherical
    # harmonics evaluated at large number of random points
    N = 1000
    lmax = 8
    vectors = np.random.normal(0, 1, (N,3))
    coeffs = evaluate_spherical_harmonics(vectors, lmax)
    num_coeffs = (lmax+1)**2
    assert coeffs.shape == (N, num_coeffs)

    # Verify addition theorem
    exact_vals = (2*np.arange(lmax+1)+1)/ (4.*np.pi)
    for l in range(lmax+1):
        prod = np.sum(coeffs[:,l**2:(l+1)**2]**2,axis=1)
        error = np.linalg.norm(prod - exact_vals[l]) 
        assert error / N < 1e-15
    print(' - Test 3 = Addition theorem: passed')

    # In the limit of infinitely many points, the columns should
    # be orthonormal. Reuse the values from above for a Monte Carlo
    # integration (if this was the sole purpose, there would be much
    # more efficient methods for quadrature)
    innerprod_matrix = coeffs.T @ coeffs / N * np.pi * 4
    difference = innerprod_matrix - np.eye(num_coeffs)
    assert np.linalg.norm(difference) / num_coeffs**2 < 1e-2   
    print(' - Test 4 = Orthogonality: passed')

    return 0


# Test correct behavior of radial projection code that generates the
# orthonormalized radial basis functions and the splined projections
# of the spherical Bessel functions onto them.
def run_test_radial_projection():
    # Test 1 = Inner product: Make sure that the implementation of inner
    # products works correctly.
    lmax=6 # polynomials can become unstable for large exponents, so this should stay small
    nmax=10
    rcut=5
    Nradial = 100000
    xx = np.linspace(0, rcut, Nradial)
    for i in range(lmax):
        for j in range(i,lmax):
            monomialprod = innerprod(xx, xx**i, xx**j)
            exponent = i+j+2 
            assert abs(monomialprod - rcut**(exponent+1)/(exponent+1)) < 1e-5
    print(' - Test 1 = inner products up to nmax=8: passed')
    
    # Test 2 = GTO: Since the code only returnes the splines, we rerun a copied
    # code fragment which is identical to the main one to test the
    # orthogonality of the obtained radial basis.
    # Generate length scales sigma_n for R_n(x)
    nmax = 8
    Nradial = 1000
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
    
    # start computing overlap
    overlap = np.zeros((nmax,nmax))
    for i in range(nmax):
        for j in range(nmax):
            overlap[i,j] = innerprod(xx, R_n_ortho[i], R_n_ortho[j])    
    ortho_error = np.eye(nmax) - overlap
    assert np.linalg.norm(ortho_error) / nmax**2 < 1e-8
    print(' - Test 2 = GTO orthogonalization: passed')
    
    # Test 3: Compare GTO projection coefficients with exact value
    prefac = np.sqrt(np.pi)
    lmax = 5
    nmax = 8
    rcut = 5.
    sigma = np.ones(nmax, dtype=float)
    for i in range(1,nmax):
        sigma[i] = np.sqrt(i)
    sigma *= rcut/nmax
    kmax = np.pi/1.5
    Neval = 562 # choose number different from Nspline
    kk = np.linspace(0, kmax, Neval)
    spline = radial_projection_gto(lmax, nmax, rcut, kmax, Nspline=200,
                                   Nradial=1000, primitive=True)
    coeffs = spline(kk)
    
    # Compare to analytical results
    factors = prefac * np.ones((nmax,lmax+1))
    coeffs_exact = np.zeros((Neval, nmax, lmax+1))
    for l in range(lmax+1):
        for n in range(nmax):
            i1 = 0.5*(3+n+l)
            i2 = 1.5+l
            factors[n,l] *= 2**(0.5*(n-l-1)) * gamma(i1) / gamma(i2) * sigma[n]**(2*i1)
            coeffs_exact[:,n,l] = factors[n,l] * kk**l * hyp1f1(i1, i2, -0.5*(kk*sigma[n])**2)
    
    error = coeffs - coeffs_exact
    assert np.linalg.norm(error) / error.size < 1e-6
    print(' - Test 3 = GTO numerical vs analytical evaluation: passed')
    
    return 0

def run_test_lode():
    # Test 1: Convergence of norm
    frames = []
    cell = np.eye(3) *14
    distances = np.linspace(2, 3., 5)    
    for d in distances:
        positions2 = [[1,1,1],[1,1,d+1]]
        frame = Atoms('O2', positions=positions2, cell=cell, pbc=True)
        frames.append(frame)
        
    species_dict = {'O':0}
    sigma = 1.5
    
    ns = [2, 4, 6, 8, 10]
    ls = [1, 3, 5, 7, 9]
    norms = np.zeros((len(ns),len(ls)))
    for i, n in enumerate(ns):
        for j, l in enumerate(ls):
            hypers = {
                'smearing':sigma,
                'max_angular':l,
                'max_radial':n,
                'cutoff_radius':5.,
                'potential_exponent':0,
                'compute_gradients':True       
            }
            calculator = Density_Projection_Calculator(**hypers)
            calculator.transform(frames, species_dict)
            features_temp = calculator.get_features()
            norms[i,j] = np.linalg.norm(features_temp[0,0])
    
    for i, n in enumerate(ns):
        plt.plot(ls, norms[i], label=f'n={n}')
    plt.legend()
    plt.xlabel('angular l')
    plt.ylabel('Norm of feature vector for one structure')
    # print('Final norm = ', norms[-1,-1])
    print(' - Test 1 = Convergence of norm for large nmax, lmax: passed')
    
    # Test 2: Evaluation of coefficients by manual integration
    
    return 0


if __name__ == '__main__':
    main()