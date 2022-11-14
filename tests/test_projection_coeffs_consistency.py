# -*- coding: utf-8 -*-
"""Tests for projection coefficients.

These are the main tests for calculating the LODE features.
"""

# Generic imports
import os
import numpy as np
import pytest
from numpy.testing import assert_allclose
from time import time

# ASE imports
from ase import Atoms
from ase.build import make_supercell
from ase.io import read

# Library specific imports
from pylode.lib.projection_coeffs import DensityProjectionCalculator
from pylode.lib.projection_coeffs_summed_strucwise import DensityProjectionCalculatorSummed

REF_STRUCTS = os.path.join(os.path.dirname(__file__), 'reference_structures')


class TestSuperCell():
    """Class for testing invariance under cell replications."""

    @pytest.fixture
    def frames(self):
        """Frames for two oxygen atoms at different positions"""
        frame_list = []
        cell = 5 * np.eye(3)
        distances = np.linspace(1, 2, 5)
        positions = np.zeros([2, 3])
        for d in distances:
            positions[1, 2] = d
            frame = Atoms('NaCl', positions=positions, cell=cell, pbc=True)
            frame_list.append(frame)
        return frame_list

    def test_supercell(self, frames):
        """Test if features are invariant by cell replications.
        The original unit cell is replicated two times.
        """
        n_atoms = len(frames[0].get_atomic_numbers())

        hypers = dict(max_radial=2,
                      max_angular=2,
                      radial_basis_radius=1,
                      smearing=2,
                      radial_basis='GTO')

        # Original cell
        calculator = DensityProjectionCalculator(**hypers)
        calculator.transform(frames, show_progress=True)
        features = calculator.features.reshape(len(frames), n_atoms, *calculator.features.shape[1:])

        # Super cell
        n_replica_per_dim = 2
        n_replica = n_replica_per_dim**3

        frames_super = [make_supercell(f, n_replica_per_dim * np.eye(3)) for f in frames]
        calculator_super = DensityProjectionCalculator(**hypers)
        calculator_super.transform(frames_super, show_progress=True)
        features_super = calculator_super.features.reshape(
            len(frames), n_replica * n_atoms,
            *calculator_super.features.shape[1:])

        # Compare contribution of first atom
        # I don't know why we have to round here...
        assert_allclose((features[:, 0]).round(10),
                        (features_super[:, ::2].mean(axis=1)).round(10))

        # Compare contribution of second atom
        assert_allclose((features[:, 1]).round(10),
                        (features_super[:, 1::2].mean(axis=1)).round(10))


class TestSummedImplementation():
    """
    Class checking the summed implementation.
    """
    @pytest.mark.parametrize("potential_exponent", [0,1])
    @pytest.mark.parametrize("smearing", [1.8, 1.1])
    @pytest.mark.parametrize("radial_basis", ['monomial', 'gto'])
    def test_agreement_summed_implementation(self, potential_exponent, smearing, radial_basis):
        # Use the frames of the Coulomb test that contain
        # 3 frames of NaCl with 8 atoms each.
        # The first 4 atoms are Na and the last 4 Cl.
        # This makes sure that the fast implementation also
        # works for multi-species systems having no special
        # symmetry.
        frames = read(os.path.join(REF_STRUCTS, "coulomb_test_frames.xyz"),
                      ":")

        # Define the hyperparameters
        lmax = 3
        hypers = {
            'smearing':smearing,
            'max_angular':lmax,
            'max_radial':1,
            'radial_basis_radius':0.1,
            'potential_exponent':potential_exponent,
            'radial_basis': radial_basis,
            'compute_gradients':False,
            'fast_implementation':True,
            'subtract_center_contribution':True
        }

        calculator = DensityProjectionCalculator(**hypers)
        calculator.transform(frames)
        descriptors = calculator.features

        calculator_summed = DensityProjectionCalculatorSummed(**hypers)
        calculator_summed.transform(frames)
        descriptors_summed = calculator_summed.features


        for iframe in range(len(frames)):
            # Na atoms
            sum_from_normal_Na = np.sum(descriptors[8*iframe:8*iframe+4], axis=0)
            assert_allclose(descriptors_summed[iframe,0], sum_from_normal_Na, atol=2e-12)
            # Cl atoms
            sum_from_normal_Cl = np.sum(descriptors[8*iframe+4:8*iframe+8], axis=0)
            assert_allclose(descriptors_summed[iframe,1], sum_from_normal_Cl, atol=2e-12)


class TestGradients():
    """
    Class checking that the gradients are implemented correctly.
    """
    def test_gradients(self):
        # Define original structure, which is a cluster of 4 atoms
        # without any special symmetries.
        cell = 10 * np.eye(3)
        frames = []
        pos_0 = np.array([[1,1,1],[3.1,1,1],[3,3.2,1.3],[2.3,3.5,3.5]])
        frames.append(Atoms('O4', positions=pos_0, cell=cell, pbc=True))

        # Define structures in which each of the 4 atoms is displaced
        # by a distance dx in the x,y and z directions, leading to
        # 4 x 3 = 12 extra structures (13 in total).
        dx = 1e-7
        for iatom in range(4):
            for direction in range(3):
                pos_new = pos_0.copy()
                pos_new[iatom, direction] += dx
                frames.append(Atoms('O4', positions=pos_new, cell=cell, pbc=True))

        # Define calculator and compute features
        nmax = 3
        lmax = 2
        rcut = 6.
        smearing = 2.1
        hypers = {
            'smearing':smearing,
            'max_angular':lmax,
            'max_radial':nmax,
            'radial_basis_radius':rcut,
            'potential_exponent':0,
            'radial_basis': 'gto',
            'compute_gradients':True,
            'subtract_center_contribution':False,
            'fast_implementation':True
            }
        calculator_pylode = DensityProjectionCalculator(**hypers)
        gradients_finite_difference = []
        calculator_pylode.transform(frames)
        features_pylode = calculator_pylode.features[:,0]

        # Get the features of the original (not distorted) frame
        # used for the finite difference calculation and its
        # gradients.
        feat_ref = features_pylode[:4]
        gradients_all = calculator_pylode.feature_gradients[:,:,0]
        gradients_pylode_firstframe = gradients_all[:16,:]

        # Compute the gradients using the finite difference method
        gradients_finite_difference = np.zeros_like(gradients_pylode_firstframe)
        atompair_idx = 0
        for i_center in range(4):
            for i_neigh in range(4):
                for i_direction in range(3):
                    idx_feat = 4 + 12*i_neigh + 4*i_direction + i_center
                    feat_new = features_pylode[idx_feat].copy()
                    grad_finite = (feat_new - feat_ref[i_center]) / dx
                    gradients_finite_difference[atompair_idx,i_direction] = grad_finite
                
                atompair_idx += 1

        # Check that the coefficients agree.
        # Note that due to the finite difference approach to gradients,
        # an absolute error on the order of the displacement dx is expected.
        assert_allclose(gradients_finite_difference, gradients_pylode_firstframe, rtol=1e-12, atol=2*dx)


class TestSlowVSFastImplementation():
    """Class checking that the slow implementation using
    explicit for loops (kept for better comparison with C++ versions)
    produces the same results as the faster implementation using np.sum.
    ."""

    def test_agreement_slow_vs_fast_implementation(self):
        # Use a simple data set only having one chemical species
        frames = read(os.path.join(REF_STRUCTS, "dispersion_test_frames.xyz"),
                      ":")

        # Define hyperparameters to run tests
        hypers = {
            'smearing': 2.5,
            'max_angular': 6,
            'max_radial': 1,
            'radial_basis_radius': 5.,
            'potential_exponent': 1,
            'radial_basis': 'monomial',
            'compute_gradients': True,
            'fast_implementation': False
        }

        
        # Run the slow implementation using manual for loops
        # This version is kept for comparison with the C++/Rust
        # versions in which the sums need to be looped explicitly.
        tstart = time()
        calculator_slow = DensityProjectionCalculator(**hypers)
        calculator_slow.transform(frames)
        tend = time()
        descriptors_slow = calculator_slow.features
        gradients_slow = calculator_slow.feature_gradients
        dt_slow = tend - tstart

        # Fast implementation ver. 1:
        # Use np.sum for the sum over k-vectors.
        # The gain in computational cost is especially
        # significant if we need to sum over a large number of k-vectors,
        # i.e. for large cells or a small smearing.
        # For these tests, relatively reasonable values are used.
        hypers['fast_implementation'] = True
        tstart = time()
        calculator_fast = DensityProjectionCalculator(**hypers)
        calculator_fast.transform(frames)
        tend = time()
        descriptors_fast = calculator_fast.features
        gradients_fast = calculator_fast.feature_gradients
        dt_fast = tend - tstart

        # Check agreement between the coefficients obtained using
        # the two implementations
        assert_allclose(descriptors_slow,
                        descriptors_fast,
                        rtol=1e-14,
                        atol=2e-13)
        assert_allclose(gradients_slow, gradients_fast, rtol=1e-14, atol=1e-14)
        assert (dt_slow > dt_fast)