# -*- coding: utf-8 -*-
"""Tests for the various implemented atomic densities.
More concretely, the tests make sure that:
1.  the provided Fourier transforms of the respective
    densities are indeed correct.
2.  the smeared 1/r^p potentials do indeed asymptotically
    behave as 1/r^p (including potential global prefactors)
3.  For many of the atomic densities and their Fourier transforms
    the value at r=0 or k=0, respectively, need to be evaluated
    explicitly due to potential singularities. Thus, the center
    contributions are checked explicitly.
"""

from re import I
import numpy as np
import pytest
import scipy
from scipy.special import gamma
from numpy.testing import assert_allclose, assert_array_less

from pylode.lib.atomic_density import AtomicDensity

np.random.seed(12419)


class TestAtomicDensity:
    """
    Tests for the various implemented atomic densities.
    """
    # Define values of smearings and potential exponents
    smearings = [0.1, 0.4, 1.0, 1.8]
    potential_exponents = [1, 2, 3, 4, 5, 6]

    # Define how much the smeared 1/r^p potentials 
    # deviate from exact 1/r^p behavior
    distances_for_asymptotics = np.array([2.,3.,4.,5.])
    rel_diffs = {}
    rel_diffs['1'] = np.array([5e-2, 3e-3, 7e-5, 6e-7])
    rel_diffs['2'] = np.array([2e-1, 2e-2, 4e-4, 4e-6])
    rel_diffs['3'] = np.array([3e-1, 3e-2, 2e-3, 2e-5])
    rel_diffs['4'] = np.array([5e-1, 7e-2, 4e-3, 6e-5])
    rel_diffs['5'] = np.array([6e-1, 2e-1, 7e-3, 2e-4])
    rel_diffs['6'] = np.array([7e-1, 2e-1, 2e-2, 4e-4])

    @pytest.mark.parametrize("smearing", smearings)
    @pytest.mark.parametrize("p", potential_exponents)
    def test_asymptotic_behavior(self, smearing, p):
        xx = smearing * self.distances_for_asymptotics
        yy_pure = 1/xx**p
        atomic_density = AtomicDensity(smearing, p)
        yy_smeared = atomic_density.get_atomic_density(xx)

        rel_deviation = (yy_pure - yy_smeared) / yy_pure
        zeros_ref = np.zeros_like(rel_deviation)
        rel_deviation_ref = self.rel_diffs[str(p)]
        assert_array_less(zeros_ref, rel_deviation)
        assert_array_less(rel_deviation, rel_deviation_ref)
    
    # Check that the value at d=0 is correct
    distances_to_test_near_field = np.array([1e-8])
    @pytest.mark.parametrize("smearing", smearings)
    @pytest.mark.parametrize("p", potential_exponents)
    def test_value_at_zero(self, smearing, p):
        # Get atomic density at tiny distance
        atomic_density = AtomicDensity(smearing, p)
        xx = self.distances_to_test_near_field
        yy = atomic_density.get_atomic_density(xx)
        assert yy.shape == (1,)
        
        # Compare to 
        f0 = 1. / (2*smearing**2)**(p/2) / gamma(p/2+1)
        relerr = abs(yy[0] - f0) / f0
        assert relerr < 2e-14

    potential_exponents_sr = [4, 6]
    # Check that the value at k=0 is correct
    kvecs_to_test_near_field = np.array([1e-12, 1e-11])
    @pytest.mark.parametrize("smearing", smearings)
    @pytest.mark.parametrize("p", potential_exponents_sr)
    def test_convergence_at_zero_kspace(self, smearing, p):
        # Get atomic density at tiny distance
        atomic_density = AtomicDensity(smearing, p)
        kk = self.kvecs_to_test_near_field
        yy = atomic_density.get_fourier_transform(kk)
        
        # Compare to value provided for k=0
        target = atomic_density.get_fourier_transform_at_zero()
        assert_allclose(np.ones_like(yy)*target, yy, rtol=1e-10, atol=1e-14)
    

    potential_exponents_sr = [4, 6]
    # Check that the value at k=0 is correct
    # For now, the exponent p=5 is excluded since
    # there are extra singularities (due to how p=5 maps back to p=3)
    kvecs_to_test_near_field = np.array([0])
    @pytest.mark.parametrize("smearing", smearings)
    @pytest.mark.parametrize("p", potential_exponents_sr)
    def test_value_at_zero_kspace(self, smearing, p):
        # Get atomic density at tiny distance
        atomic_density = AtomicDensity(smearing, p)
        kk = self.kvecs_to_test_near_field
        yy = atomic_density.get_fourier_transform(kk)
        
        # Compare to value provided for k=0
        target = atomic_density.get_fourier_transform_at_zero()
        assert_allclose(np.ones_like(yy)*target, yy, rtol=1e-15, atol=1e-15)
    
