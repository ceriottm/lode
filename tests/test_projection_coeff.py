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


class TestCoulombRandomStructures():
    """
    Class for checking the agreement of both the potential and forces
    with exact 1/r potentials on random structures.
    This test complements the Madelung tests by using more complicated
    structures which also require forces.
    """

    def test_coulomb_random_structure(self):
        # Get the predefined frames with the
        # Coulomb energy and forces computed by GROMACS using PME
        # using parameters as defined in the GROMACS manual
        # https://manual.gromacs.org/documentation/current/user-guide/mdp-options.html#ewald
        #
        # coulombtype = PME
        # fourierspacing = 0.01  ; 1/nm
        # pme_order = 8
        # rcoulomb = 0.3  ; nm
        frames = read(os.path.join(REF_STRUCTS, "coulomb_test_frames.xyz"),
                      ":")

        # Energies in Gaussian units (without e²/[4 π ɛ_0] prefactor)
        energy_target = np.array([frame.info["energy"] for frame in frames])
        # Forces in Gaussian units per Å
        forces_target = np.array([frame.arrays["forces"] for frame in frames])

        # Define hyperparameters to run tests
        rcut = 0.2
        hypers = {
            'smearing': .4,
            'max_angular': 1,
            'max_radial': 1,
            'cutoff_radius': rcut,
            'potential_exponent': 1,
            'radial_basis': 'monomial',
            'compute_gradients': True,
            'fast_implementation': True,
            'subtract_center_contribution':True
        }

        # Run the slow implementation using manual for loops
        # This version is kept for comparison with the C++/Rust versions
        # in which the sums need to be looped explicitly.
        calculator = DensityProjectionCalculator(**hypers)
        calculator.transform(frames)
        descriptors = calculator.features
        gradients = calculator.feature_gradients

        # Compute the analytically expected expressions for the
        # energies and forces
        energy = np.zeros(3)
        forces = np.zeros((3, 8, 3))
        for iframe in range(len(frames)):
            # Define indices to specify blocks
            i1 = 8 * iframe  # start of Na block
            i2 = 8 * iframe + 4  # end of Na / start of Cl block
            i3 = 8 * iframe + 8  # end of Cl block

            # Add contributions in the following order:
            # Na-Na, Na-Cl, Cl-Na, Cl-Cl contributions
            energy[iframe] += np.sum(descriptors[i1:i2, 0, 0, 0])
            energy[iframe] -= np.sum(descriptors[i1:i2, 1, 0, 0])
            energy[iframe] -= np.sum(descriptors[i2:i3, 0, 0, 0])
            energy[iframe] += np.sum(descriptors[i2:i3, 1, 0, 0])

            # For the gradients, the l=1 components of the projection
            # coefficients provide us with the dipole moment of the
            # exterior charges, which provides the force on the center
            # atom. Note that the real spherical harmonics are ordered
            # such that m=(-1,0,1) is mapped to (y,z,x).
            # x-component of forces
            forces[iframe, :, 0] += descriptors[i1:i3, 0, 0, 3]
            forces[iframe, :, 0] -= descriptors[i1:i3, 1, 0, 3]

            # y-component of forces
            forces[iframe, :, 1] += descriptors[i1:i3, 0, 0, 1]
            forces[iframe, :, 1] -= descriptors[i1:i3, 1, 0, 1]

            # z-component of forces
            forces[iframe, :, 2] += descriptors[i1:i3, 0, 0, 2]
            forces[iframe, :, 2] -= descriptors[i1:i3, 1, 0, 2]

            # flip sign for Na atoms
            forces[iframe, :4] *= -1

        # Prefactor used to convert into Gaussian units
        # assuming that the cutoff is sufficiently small
        # For the energy, an extra factor of 1/2 takes into
        # account that all (i,j) pairs are counted twice
        prefac_e = np.sqrt(3 / (4 * np.pi * rcut**3))
        prefac_f = np.sqrt(15 / (4 * np.pi * rcut**5))
        energy *= prefac_e / 2
        forces *= prefac_f

        # Make sure that the values agree
        assert_allclose(energy_target, energy, rtol = 1e-3)
        assert_allclose(forces_target, forces, rtol = 3e-2)

        # Average rel. error of forces should be less than 1%
        diff = np.abs(forces - forces_target).flatten()
        assert np.mean(diff/forces.flatten()) < 1e-2


class TestMadelung:
    """Test LODE feature against Madelung constant of different crystals."""

    scaling_factors = np.array([0.5, 1, 2.1, 3.3])
    crystal_list = ["NaCl", "CsCl", "ZnS", "ZnSO4"]

    def build_frames(self, symbols, positions, cell, scaling_factors):
        """Build an list of `ase.Atoms` instances.

        Starting from a cell and atomic positions specifying an ASE Atoms
        object, generate a list containing scaled versions of the original
        frame.

        Parameters
        ----------
        symbols : list[str]
            list of symbols
        positions : list of xyz-positions
            Atomic positions
        cell : 3x3 matrix or length 3 or 6 vector
            Unit cell vectors.
        scaling_factors : list[float]
            scaling factor for the positions and the cell
        """
        if len(positions.shape) != 2:
            raise ValueError("Positions must be a (N, 3) array!")
        if positions.shape[1] != 3:
            raise ValueError("Positions must have 3 columns!")
        if cell.shape != (3, 3):
            raise ValueError("Cell must be a 3x3 matrix!")

        frames = []
        for a in scaling_factors:
            frames.append(
                Atoms(symbols=symbols,
                      positions=a * positions,
                      cell=a * cell,
                      pbc=True))

        return frames

    @pytest.fixture
    def crystal_dictionary(self):
        """
        Define the parameters of the three binary crystal structures:
        NaCl, CsCl and ZnCl. The reference values of the Madelung
        constants is taken from the book "Solid State Physics"
        by Ashcroft and Mermin.

        Note: Symbols and charges keys have to be sorted according to their
        atomic number in ascending alternating order! For an example see 
        ZnS04 in the wurtzite structure.
        """
        # Initialize dictionary for crystal paramaters
        d = {k: {} for k in self.crystal_list}

        # NaCl structure
        # Using a primitive unit cell, the distance between the
        # closest Na-Cl pair is exactly 1. The cubic unit cell
        # in these units would have a length of 2.
        d["NaCl"]["symbols"] = ['Na', 'Cl']
        d["NaCl"]["charges"] = np.array([1, -1])
        d["NaCl"]["positions"] = np.array([[0, 0, 0], [1, 0, 0]])
        d["NaCl"]["cell"] = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        d["NaCl"]["madelung"] = 1.7476

        frames = self.build_frames(symbols=d["NaCl"]["symbols"],
                                   positions=d["NaCl"]["positions"],
                                   cell=d["NaCl"]["cell"],
                                   scaling_factors=self.scaling_factors)
        d["NaCl"]["frames"] = frames

        # CsCl structure
        # This structure is simple since the primitive unit cell
        # is just the usual cubic cell with side length set to one.
        # The closest Cs-Cl distance is sqrt(3)/2. We thus divide
        # the Madelung constant by this value to match the reference.
        d["CsCl"]["symbols"] = ["Cl", "Cs"]
        d["CsCl"]["charges"] = np.array([1, -1])
        d["CsCl"]["positions"] = np.array([[0, 0, 0], [.5, .5, .5]])
        d["CsCl"]["cell"] = np.diag([1, 1, 1])
        d["CsCl"]["madelung"] = 2 * 1.7626 / np.sqrt(3)

        frames = self.build_frames(symbols=d["CsCl"]["symbols"],
                                   positions=d["CsCl"]["positions"],
                                   cell=d["CsCl"]["cell"],
                                   scaling_factors=self.scaling_factors)
        d["CsCl"]["frames"] = frames

        # ZnS (zincblende) structure
        # As for NaCl, a primitive unit cell is used which makes
        # the lattice parameter of the cubic cell equal to 2.
        # In these units, the closest Zn-S distance is sqrt(3)/2.
        # We thus divide the Madelung constant by this value.
        # If, on the other hand, we set the lattice constant of
        # the cubic cell equal to 1, the Zn-S distance is sqrt(3)/4.
        d["ZnS"]["symbols"] = ["S", "Zn"]
        d["ZnS"]["charges"] = np.array([1, -1])
        d["ZnS"]["positions"] = np.array([[0, 0, 0], [.5, .5, .5]])
        d["ZnS"]["cell"] = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        d["ZnS"]["madelung"] = 2 * 1.6381 / np.sqrt(3)

        frames = self.build_frames(symbols=d["ZnS"]["symbols"],
                                   positions=d["ZnS"]["positions"],
                                   cell=d["ZnS"]["cell"],
                                   scaling_factors=self.scaling_factors)
        d["ZnS"]["frames"] = frames

        # ZnS (O4) in wurtzite structure (triclinic cell)
        u = 3 / 8
        c = np.sqrt(1 / u)

        d["ZnSO4"]["symbols"] = ["S", "Zn", "S", "Zn"]
        d["ZnSO4"]["charges"] = np.array([1, -1, 1, -1])
        d["ZnSO4"]["positions"] = np.array([[.5, .5 / np.sqrt(3), 0.],
                                            [.5, .5 / np.sqrt(3), u * c],
                                            [.5, -.5 / np.sqrt(3), 0.5 * c],
                                            [.5, -.5 / np.sqrt(3), (.5 + u) * c]])
        d["ZnSO4"]["cell"] = np.array([[.5, -0.5 * np.sqrt(3), 0],
                                       [.5, .5 * np.sqrt(3), 0],
                                       [0, 0, c]])
                            
        d["ZnSO4"]["madelung"] = 1.6413 / (u * c)

        frames = self.build_frames(symbols=d["ZnSO4"]["symbols"],
                                   positions=d["ZnSO4"]["positions"],
                                   cell=d["ZnSO4"]["cell"],
                                   scaling_factors=self.scaling_factors)
        d["ZnSO4"]["frames"] = frames

        return d

    @pytest.mark.parametrize("crystal_name", crystal_list)
    @pytest.mark.parametrize("smearing", [0.2, 0.1])
    @pytest.mark.parametrize("rcut", np.geomspace(1e-2, 0.2, 4))
    @pytest.mark.parametrize("radial_basis", ["monomial", "gto"])
    def test_madelung(self, crystal_dictionary, crystal_name, smearing, rcut, radial_basis):
        frames = crystal_dictionary[crystal_name]["frames"]
        n_atoms = len(crystal_dictionary[crystal_name]["symbols"])

        calculator = DensityProjectionCalculator(
            max_radial=1,
            max_angular=0,
            cutoff_radius=rcut,
            smearing=smearing,
            radial_basis=radial_basis,
            subtract_center_contribution=True)

        calculator.transform(frames=frames)
        features = calculator.features
        features = features.reshape(len(frames), n_atoms, *features.shape[1:])

        # Contribution of second atom on first atom
        madelung = crystal_dictionary[crystal_name]["charges"][0] * features[:, 0, 0, :]
        madelung += crystal_dictionary[crystal_name]["charges"][1] * features[:, 0, 1, :]

        # Convert pyLODE coefficients into actual potential
        if radial_basis == 'monomial':
            conversion_factor = -np.sqrt(4 * np.pi * rcut**3 / 3)
        elif radial_basis == 'gto':
            smear_gto = rcut
            conversion_factor = -(4 * np.pi * smear_gto**2)**0.75
        madelung /= conversion_factor

        assert_allclose(madelung.flatten(),
                        crystal_dictionary[crystal_name]["madelung"] /
                        self.scaling_factors,
                        rtol=6e-1)


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
                      cutoff_radius=1,
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

    def test_agreement_summed_implementation(self):
        # Use the frames of the Coulomb test that contain
        # 3 frames of NaCl with 8 atoms each.
        # The first 4 atoms are Na and the last 4 Cl.
        # This makes sure that the fast implementation also
        # works for multi-species systems having no special
        # symmetry.
        frames = read(os.path.join(REF_STRUCTS, "coulomb_test_frames.xyz"),
                      ":")

        # Define the hyperparameters
        lmax = 4
        hypers = {
            'smearing':1.,
            'max_angular':lmax,
            'max_radial':1,
            'cutoff_radius':0.1,
            'potential_exponent':1,
            'radial_basis': 'monomial',
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
        cell = 15 * np.eye(3)
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
        smearing = 1.
        hypers = {
            'smearing':smearing,
            'max_angular':lmax,
            'max_radial':nmax,
            'cutoff_radius':rcut,
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
        # Generate a simple data set containing O2 molecules
        # TODO: use more realistic structures,
        # ideally, some having a single chemical species (e.g. Phosphorus),
        # + some binary systems (e.g. water) + more complex ones (BaTiO3)
        frames = []
        cell = np.eye(3) * 16
        distances = np.linspace(1.5, 2., 20)
        for d in distances:
            positions = [[1, 1, 1], [1, 1, d + 1]]
            frame = Atoms('O2', positions=positions, cell=cell, pbc=True)
            frames.append(frame)

        # Define hyperparameters to run tests
        hypers = {
            'smearing': 1.,
            'max_angular': 6,
            'max_radial': 1,
            'cutoff_radius': 5.,
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
        assert (dt_slow > 3 * dt_fast)