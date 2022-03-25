# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:32:41 2022

@author: kevin
"""

from ase import Atoms
from matplotlib import pyplot as plt
import numpy as np

from pylode.lib.projection_coeffs import DensityProjectionCalculator


def test_lode():
    # Test 1: Convergence of norm
    frames = []
    cell = np.eye(3) * 14
    distances = np.linspace(2, 3., 5)
    for d in distances:
        positions2 = [[1, 1, 1], [1, 1, d + 1]]
        frame = Atoms('O2', positions=positions2, cell=cell, pbc=True)
        frames.append(frame)

    species_dict = {'O': 0}
    sigma = 1.5

    ns = [2, 4, 6, 8, 10]
    ls = [1, 3, 5, 7, 9]
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
    plt.savefig("norm.png")
