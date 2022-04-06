# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:32:41 2022

@author: kevin
"""

import pytest
from ase import Atoms
from ase.build import make_supercell
import numpy as np
from numpy.testing import assert_allclose

from pylode.lib.projection_coeffs import DensityProjectionCalculator


class TestMadelung:
    """Test LODE feature against Madelung constant of different crystals."""

    scaling_factors = np.array([0.5, 1, 2, 3, 5, 10])
    crystal_list = ["NaCl", "CsCl", "ZnS"]

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
        atomic number!
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

        return d

    @pytest.mark.parametrize("crystal_name", crystal_list)
    @pytest.mark.parametrize("smearing", [0.2, 0.15, 0.1])
    @pytest.mark.parametrize("rcut", [0.2, 0.1, 0.05, 0.01])
    def test_madelung(self, crystal_dictionary, crystal_name, smearing, rcut):

        frames = crystal_dictionary[crystal_name]["frames"]
        n_atoms = len(crystal_dictionary[crystal_name]["symbols"])

        calculator = DensityProjectionCalculator(
            max_radial=1,
            max_angular=0,
            cutoff_radius=rcut,
            smearing=smearing,
            radial_basis="monomial",
            subtract_center_contribution=True)

        calculator.transform(frames=frames)
        features = calculator.features
        features = features.reshape(len(frames), n_atoms, *features.shape[1:])

        # Contribution of second atom on first atom
        madelung = crystal_dictionary[crystal_name]["charges"][0] * features[:, 0, 0, :]
        madelung += crystal_dictionary[crystal_name]["charges"][1] * features[:, 0, 1, :]

        # Normalization
        madelung /= -np.sqrt(4 * np.pi / 3 * rcut**3)

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

        hypers = dict(
            max_radial=2,
            max_angular=2,
            cutoff_radius=1,
            smearing=2,
            radial_basis='GTO')

        # Original cell
        calculator = DensityProjectionCalculator(**hypers)
        calculator.transform(frames, show_progress=True)
        features = calculator.features.reshape(
            len(frames),
            n_atoms,
            *calculator.features.shape[1:])

        # Super cell
        n_replica_per_dim = 2
        n_replica = n_replica_per_dim**3

        frames_super = [make_supercell(f, n_replica_per_dim * np.eye(3)) for f in frames]
        calculator_super = DensityProjectionCalculator(**hypers)
        calculator_super.transform(frames_super, show_progress=True)
        features_super = calculator_super.features.reshape(
            len(frames),
            n_replica * n_atoms,
            *calculator_super.features.shape[1:])

        # Compare contribution of first atom
        # I don't know why we have to round here...
        assert_allclose((features[:,0]).round(10),
                        (features_super[:,::2].mean(axis=1)).round(10))

        # Compare contribution of second atom
        assert_allclose((features[:,1]).round(10),
                        (features_super[:,1::2].mean(axis=1)).round(10))
