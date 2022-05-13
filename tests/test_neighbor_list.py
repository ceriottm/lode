# -*- coding: utf-8 -*-
"""Tests for neighbor list."""

import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose

from ase import Atoms

from pylode.lib.neighbor_list import NeighborList

class TestNeighborList:
    """
    Test correct behavior of neighbor list
    """

    def test_neighbor_list_on_single_atom_frame(self):
        """
        Generate simple cubic frames containing a single atom.
        For this system, it is easy to compute the exact number of neighbors
        that needs to be found by the NeighborList class.
        """
        # Pick cutoff radii right around the values at which a jump in the
        # number of neighbors is expected
        eps = 1e-5
        cutoffs = []
        critical_cutoffs = [1, np.sqrt(2), np.sqrt(3)]
        for crit in critical_cutoffs:
            cutoffs.append(crit-eps)
            cutoffs.append(crit+eps)
        cutoffs = np.array(cutoffs)

        # For each of the cutoff radii, check that the number of found
        # neighbors within the cutoff is correct.
        cell = np.eye(3)
        positions = [[0, 0, 0]]
        species_dict = {'O':0}
        num_neighbors_correct = [0, 6, 6, 18, 18, 26]
        for icut, cutoff in enumerate(cutoffs):
            frame = Atoms('O', positions=positions, cell=cell, pbc=True)
            nl = NeighborList(frame, species_dict, cutoff)
            num_neighbors_found = nl.neighbor_list[0][0].entries['number_of_neighbors']
            assert num_neighbors_found == num_neighbors_correct[icut]

        # Scale both the cell and the cutoffs by some factor and make
        # sure that the results stay the same
        cutoffs *= 2 * np.pi
        cell *= 2 * np.pi
        for icut, cutoff in enumerate(cutoffs):
            frame = Atoms('O', positions=positions, cell=cell, pbc=True)
            nl = NeighborList(frame, species_dict, cutoff)
            num_neighbors_found = nl.neighbor_list[0][0].entries['number_of_neighbors']
            assert num_neighbors_found == num_neighbors_correct[icut]
    
    def test_neighbor_list_on_dimers(self):
        """
        Test whether the neighbor list implementation works correctly
        for a set of dimers. The distance between the atom as well as the
        cutoff radii are varied. The two atoms should only be registered
        as neighbors if the distance is smaller than the cutoff.
        """
        # Define the used distances and cutoffs
        # To avoid issues due to rounding, we make sure
        # that the cutoffs are always off from the bond
        # lengths.
        cell = np.eye(3) * 16
        eps = 1e-2
        Ndimers = 15
        distances = np.linspace(1., 2.5, Ndimers)
        cutoffs = np.linspace(1.+eps, 2.5+eps, Ndimers)
        species_dict = {'O':0}

        # For each of the distances and cutoffs, check that
        # the results obtained from the NeighborList class
        # is correct.
        for dist in distances:
            positions = [[1,1,1],[1,1,1+dist]]
            frame = Atoms('O2', positions=positions, cell=cell, pbc=True)
            for cutoff in cutoffs:
                nl = NeighborList(frame, species_dict, cutoff)
                num_neighbors_found = nl.neighbor_list[0][0].entries['number_of_neighbors']

                if cutoff > dist:
                    assert num_neighbors_found == 1
                    dist_found = nl.neighbor_list[0][0].entries['pair_distances'][0]
                    assert np.isclose(dist, dist_found)
                    dist_vec_found = nl.neighbor_list[0][0].entries['pair_vectors'][0]
                    assert_allclose(dist_vec_found, np.array([0, 0, dist]))
                else:
                    assert num_neighbors_found == 0