# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:32:41 2022

@author: kevin
"""

import pytest
from ase import Atoms
from matplotlib import pyplot as plt
import numpy as np
from numpy.testing import assert_allclose

from pylode.lib.projection_coeffs import DensityProjectionCalculator


def test_lode():
    # Test 1: Convergence of norm
    # NOTE: Currently, this is only used to check that the code actually runs
    frames = []
    cell = 14 * np.eye(3)
    distances = np.linspace(2, 3, 5)
    for d in distances:
        positions2 = [[1, 1, 1], [1, 1, d + 1]]
        frame = Atoms('O2', positions=positions2, cell=cell, pbc=True)
        frames.append(frame)

    ns = [2, 4, 6]
    ls = [1, 3, 5]
    norms = np.zeros((len(ns), len(ls)))
    for i, n in enumerate(ns):
        for j, l in enumerate(ls):
            hypers = {
                'smearing': 1.5,
                'max_angular': l,
                'max_radial': n,
                'cutoff_radius': 5.,
                'potential_exponent': 0,
                'radial_basis': 'gto',
                'compute_gradients': True
            }
            calculator = DensityProjectionCalculator(**hypers)
            calculator.transform(frames)
            features_temp = calculator.features
            norms[i, j] = np.linalg.norm(features_temp[0, 0])

        plt.plot(ls, norms[i], label=f'n={n}')

    plt.legend()
    plt.xlabel('angular l')
    plt.ylabel('Norm of feature vector for one structure')


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
        """Init dictionary for crystal paramaters."""
        d = {k: {} for k in self.crystal_list}

        d["NaCl"]["symbols"] = ['Na', 'Cl']
        d["NaCl"]["positions"] = np.array([[0, 0, 0], [1,0,0]])
        d["NaCl"]["cell"] = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        d["NaCl"]["madelung"] = 1.7476

        frames = self.build_frames(symbols=d["NaCl"]["symbols"],
                                   positions=d["NaCl"]["positions"],
                                   cell=d["NaCl"]["cell"],
                                   scaling_factors=self.scaling_factors)
        d["NaCl"]["frames"] = frames

        d["CsCl"]["symbols"] = ["Cs", "Cl"]
        d["CsCl"]["positions"] = np.array([[0, 0, 0], [.5, .5, .5]])
        d["CsCl"]["cell"] = np.diag([1, 1, 1])
        d["CsCl"]["madelung"] = 2 * 1.7626 / np.sqrt(3)

        frames = self.build_frames(symbols=d["CsCl"]["symbols"],
                                   positions=d["CsCl"]["positions"],
                                   cell=d["CsCl"]["cell"],
                                   scaling_factors=self.scaling_factors)
        d["CsCl"]["frames"] = frames

        d["ZnS"]["symbols"] = ["Zn", "S"]
        d["ZnS"]["positions"] = np.array([[0, 0, 0], [.5, .5, .5]])
        d["ZnS"]["cell"] = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        d["ZnS"]["madelung"] = 2 * 1.6381 / np.sqrt(3)
 
        frames = self.build_frames(symbols=d["ZnS"]["symbols"],
                                   positions=d["ZnS"]["positions"],
                                   cell=d["ZnS"]["cell"],
                                   scaling_factors=self.scaling_factors)
        d["ZnS"]["frames"] = frames

        return d

    @pytest.mark.parametrize("smearing", [0.1, 0.15, 0.2, 0.3])
    @pytest.mark.parametrize("rcut", [0.01, 0.05, 0.1, 0.2])
    @pytest.mark.parametrize("crystal_name", crystal_list)
    def test_madelung(self, crystal_dictionary, smearing, rcut, crystal_name):

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
        global_factor = 1 / np.sqrt(4 * np.pi/3 * rcut**3)
        X = -global_factor * (features[:, 0, 0, :] - features[:, 0, 1, :])

        assert_allclose(X.flatten(),
                        crystal_dictionary[crystal_name]["madelung"] / self.scaling_factors,
                        rtol=6e-1)
