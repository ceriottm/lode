
# -*- coding: utf-8 -*-
"""
Build the neighbor list for a single ase.Atoms frame.
The list is used in the real space implementation of the
projection coefficients.
"""

from re import I
import numpy as np
from ase.neighborlist import get_distance_matrix

class NeighborList_Entry():
    """
    Class for storing the neighbor information of a fixed center atom i
    and neighbor species a.
    """
    def __init__(self, input_list):
        # Store all neighbor informations as numpy arrays
        N_neighbors = len(input_list)
        atom_indices = np.zeros((N_neighbors,), dtype=int)
        pair_distances = np.zeros((N_neighbors,), dtype=float)
        pair_vectors = np.zeros((N_neighbors, 3), dtype=float)
        for i, entry in enumerate(input_list):
            atom_indices[i] = entry[0]
            pair_distances[i] = entry[1]
            pair_vectors[i] = np.array([entry[2], entry[3], entry[4]])
        
        # Store provided entries in a convenient dictionary format
        self.entries = {}
        self.entries['number_of_neighbors'] = N_neighbors
        self.entries['indices'] = atom_indices
        self.entries['pair_distances'] = pair_distances
        self.entries['pair_vectors'] = pair_vectors

class NeighborList():
    """
    Class for generating a neighbor list including periodic images
    for a single ase.Atoms frame.
    """
    def __init__(self,
                 frame,
                 species_dict,
                 cutoff,
                 group_species=False):
        self.frame = frame
        self.species_dict = species_dict
        self.cutoff = cutoff
        self.build_neighborlist()

    def build_neighborlist(self):
        """
        Primitive implementation of neighbor list not taking into account
        periodic images. All the pairs are counted twice in this version.
        This makes the actual implementation slower but leads to a simpler
        code. In future updates, half-neighbor lists will be used instead.
        """
        # Initialization
        distance_matrix = self.frame.get_distance_matrix()
        positions = self.frame.get_positions()
        chem_species = self.frame.get_chemical_species()
        num_atoms = len(distance_matrix)
        assert num_atoms == len(chem_species)
        num_species = len(self.species_dict)
        self.neighbor_list = []

        for icenter in range(num_atoms):
            # For each center atom i, the neighbor list is a python list
            # containing num_species elements. Each of those will be a
            # numpy array containing all relevant pair distances of the
            # neighbors that are within a cutoff radius and the vectors
            # connecting atom i to atom j.
            r_i = positions[icenter] 
            neighbors_i = []
            for a in range(num_species):
                    neighbors_i.append([])

            # Start filling up the neighbor list
            for ineigh, r_neigh, aneigh in enumerate(zip(positions, chem_species)):
                dist2 = distance_matrix[icenter, ineigh]
                r_ij = r_neigh - r_i
                dist = np.linalg.norm(r_ij)
                assert dist == dist2

                if dist < self.cutoff:
                    species_idx = self.species_dict[aneigh]
                    neighbors_i[species_idx].append([ineigh, dist, r_ij[0], r_ij[1], r_ij[2]])

            # Convert the lists of distances to numpy arrays
            for a in range(num_species):
                neighbors_i[a] = NeighborList_Entry(neighbors_i[a])
            self.neighbor_list.append(neighbors_i)