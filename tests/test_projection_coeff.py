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
    cell = np.eye(3) * 14
    distances = np.linspace(2, 3., 5)
    for d in distances:
        positions2 = [[1, 1, 1], [1, 1, d + 1]]
        frame = Atoms('O2', positions=positions2, cell=cell, pbc=True)
        frames.append(frame)

    species_dict = {'O': 0}
    sigma = 1.5

    ns = [2, 4, 6]
    ls = [1, 3, 5]
    norms = np.zeros((len(ns), len(ls)))
    for i, n in enumerate(ns):
        for j, l in enumerate(ls):
            hypers = {
                'smearing': sigma,
                'max_angular': l,
                'max_radial': n,
                'cutoff_radius': 5.,
                'potential_exponent': 0,
                'radial_basis': 'gto',
                'compute_gradients': True
            }
            calculator = DensityProjectionCalculator(**hypers)
            calculator.transform(frames, species_dict)
            features_temp = calculator.features
            norms[i, j] = np.linalg.norm(features_temp[0, 0])

        plt.plot(ls, norms[i], label=f'n={n}')

    plt.legend()
    plt.xlabel('angular l')
    plt.ylabel('Norm of feature vector for one structure')


class TestMadelung:
    """Test LODE feature against Madelung constant of different crystal."""

    scaling_factors = np.array([.8, 1, 2, 5, 10])
    crystal_list = ["NaCl", "CsCl", "ZnS"]

    def build_frames(self, symbols, positions, cell, scaling_factors):
        """Build an list of ase Atoms instances based.

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

        d["NaCl"]["symbols"] = 4 * ['Na'] + 4 * ['Cl']
        d["NaCl"]["positions"] = np.array([[.0, .0, .0], [.5, .5, .0],
                                           [.5, .0, .5], [.0, .5, .5],
                                           [.5, .0, .0], [.0, .5, .0],
                                           [.0, .0, .5], [.5, .5, .5]])
        d["NaCl"]["cell"] = np.diag([1, 1, 1])
        d["NaCl"]["madelung"] = 1.7476 / 2
        frames = self.build_frames(symbols=d["NaCl"]["symbols"],
                                   positions=d["NaCl"]["positions"],
                                   cell=d["NaCl"]["cell"],
                                   scaling_factors=self.scaling_factors)
        d["NaCl"]["frames"] = frames

        d["CsCl"]["symbols"] = ["Cs", "Cl"]
        d["CsCl"]["positions"] = np.array([[0, 0, 0], [.5, .5, .5]])
        d["CsCl"]["cell"] = np.diag([1, 1, 1])
        d["CsCl"]["madelung"] = 1.7626 / 2 / np.sqrt(3)
        frames = self.build_frames(symbols=d["CsCl"]["symbols"],
                                   positions=d["CsCl"]["positions"],
                                   cell=d["CsCl"]["cell"],
                                   scaling_factors=self.scaling_factors)
        d["CsCl"]["frames"] = frames

        d["ZnS"]["symbols"] = ["Zn", "S"]
        d["ZnS"]["positions"] = np.array([[0, 0, 0], [.5, .5, .5]])
        d["ZnS"]["cell"] = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        d["ZnS"]["madelung"] = 1.63805505338879 / 4 / np.sqrt(3)
        frames = self.build_frames(symbols=d["ZnS"]["symbols"],
                                   positions=d["ZnS"]["positions"],
                                   cell=d["ZnS"]["cell"],
                                   scaling_factors=self.scaling_factors)
        d["ZnS"]["frames"] = frames

        return d

    @pytest.mark.parametrize("crystal", crystal_list)
    def test_madelung(self, crystal_dictionary, crystal):
        frames = crystal_dictionary[crystal]["frames"]
        n_atoms = len(crystal_dictionary[crystal]["symbols"])

        species_dict = {atom.symbol: atom.number for atom in frames[0]}
        print(species_dict)

        calculator = DensityProjectionCalculator(
            max_radial=1,
            max_angular=0,
            cutoff_radius=0.5,
            smearing=0.3,
            radial_basis="monomial",
            subtract_center_contribution=True)

        calculator.transform(frames=frames, species_dict=species_dict)
        features = calculator.features
        features = features.reshape(len(frames), n_atoms, *features.shape[1:])

        # Contribution of second atom on first atom
        X = features[:, 0, 0, :] - features[:, 0, 1, :]
        assert_allclose(-X, crystal_dictionary[crystal]["madelung"])
