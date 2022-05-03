
# -*- coding: utf-8 -*-
"""
Build the neighbor list for a single ase.Atoms frame.
The list is used in the real space implementation of the
projection coefficients.
"""

from re import I
import numpy as np
from ase.neighborlist import get_distance_matrix

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

        if group_species:
            self.build_neighborlist_by_species()
        else:
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
        chem_species = self.frame.get_chemical_species()
        num_atoms = len(distance_matrix)
        assert num_atoms == len(chem_species)
        num_species = len(self.species_dict)
        self.neighbor_list = []

        for icenter in range(num_atoms):
            # For each center atom i, the neighbor list is a python list
            # containing num_species elements. Each of those will be a
            # numpy array containing all relevant pair distances of the
            # neighbors that are within a cutoff radius
            neighbors_i = []
            for a in range(num_species):
                    neighbors_i.append([])

            # Start filling up the neighbor list
            for ineigh, aneigh in enumerate(chem_species):
                dist = distance_matrix[icenter, ineigh]
                if dist < self.cutoff:
                        species_idx = self.species_dict[aneigh]
                        neighbors_i[species_idx].append(dist)

            # Convert the lists of distances to numpy arrays
            for a in range(num_species):
                    neighbors_i[a] = np.array(neighbors_i[a])
            self.neighbor_list.append(neighbors_i)

    def build_neighborlist_by_species(self):
        """
        Primitive implementation of neighbor list not yet
        taking into account periodic images. The distances of
        all neighbors belonging to the same chemical species
        are grouped together for a faster evaluation using built-in
        numpy functions rather than explicit loops in python for faster
        speed.
        """
        # Initialization
        distance_matrix = self.frame.get_distance_matrix()
        chem_species = self.frame.get_chemical_species()
        num_atoms = len(distance_matrix)
        assert num_atoms == len(chem_species)
        num_species = len(self.species_dict)
        self.neighbor_list_by_species = []

        # Start main loop
        for icenter in range(num_atoms):
            # For each center atom i, the neighbor list is a python list
            # containing num_species elements. Each of those will be a
            # numpy array containing all relevant pair distances of the
            # neighbors that are within the cutoff radius.
            neighbors_i = []
            for a in range(num_species):
                    neighbors_i.append([])

            # Start filling up the neighbor list
            for ineigh, aneigh in enumerate(chem_species):
                pair_distance = distance_matrix[icenter, ineigh]
                if pair_distance < self.cutoff:
                        species_idx = self.species_dict[aneigh]
                        neighbors_i[species_idx].append(pair_distance)

            # Convert the lists of distances to numpy arrays
            for a in range(num_species):
                    neighbors_i[a] = np.array(neighbors_i[a])
            self.neighbor_list_by_species.append(neighbors_i)
