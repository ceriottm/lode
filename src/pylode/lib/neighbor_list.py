
# -*- coding: utf-8 -*-
"""
Build the neighbor list for a single ase.Atoms frame.
The list is used in the real space implementation of the
projection coefficients.
"""

import numpy as np

class NeighborList_Entry:
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
            pair_vectors[i] = entry[2]
        
        # Store provided entries in a convenient dictionary format
        self.entries = {}
        self.entries['number_of_neighbors'] = N_neighbors
        self.entries['indices'] = atom_indices
        self.entries['pair_distances'] = pair_distances
        self.entries['pair_vectors'] = pair_vectors

class NeighborList:
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
        Primitive implementation of neighbor list.
        Currently, the original atom + 26 periodic images are considered.
        Every periodic image that falls within the cutoff is included
        as a neighbor.
        Note that all pairs are counted twice in this version.
        This makes the actual implementation slower but leads to simpler code.
        code.
        """
        # Initialization
        cell = self.frame.get_cell()
        positions = self.frame.get_positions()
        chem_species = self.frame.get_chemical_symbols()
        num_atoms = len(chem_species)
        assert num_atoms == len(positions)
        num_species = len(self.species_dict)
        self.neighbor_list = []

        # Create matrix that generates all periodic images in the cells
        # neighboring the center one
        periodic_shifts = np.zeros((27, 3))
        idx = 0
        for ix in [-1,0,1]:
            for iy in [-1, 0, 1]:
                for iz in [-1, 0, 1]:
                    shift = ix * cell[0] + iy * cell[1] + iz * cell[2]
                    periodic_shifts[idx] = shift
                    idx += 1

        for icenter in range(num_atoms):
            # For each center atom i, the neighbor list is a python list
            # containing num_species elements. Each of those will be a
            # numpy array containing all relevant pair distances of the
            # neighbors that are within a cutoff radius and the vectors
            # connecting atom i to atom j.
            r_i = positions[icenter] 
            neighbors_i = []

            # The a-th entry of this list contains the information about
            # neighbors of species a
            for a in range(num_species):
                    neighbors_i.append([])

            # Start filling up the neighbor list
            for j, (r_j, a_j) in enumerate(zip(positions, chem_species)):
                for shift in periodic_shifts:
                    if j == icenter and np.allclose(shift, np.array([0,0,0])):
                        continue 
                    r_j_tot = r_j + shift
                    r_ij = r_j_tot - r_i
                    dist = np.linalg.norm(r_ij)

                    if dist < self.cutoff:
                        species_idx = self.species_dict[a_j]
                        neighbors_i[species_idx].append([j, dist, r_ij])

            # Convert the lists of distances to numpy arrays
            for a in range(num_species):
                neighbors_i[a] = NeighborList_Entry(neighbors_i[a])
            self.neighbor_list.append(neighbors_i)
