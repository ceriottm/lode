
# -*- coding: utf-8 -*-
"""
Generate the k-vectors (also called reciprocal or Fourier vectors)
needed for the k-space implementation of LODE and SOAP.
More specifically, these are all points of a (reciprocal space)
lattice that lie within a ball of a specified cutoff radius.

Please check out the file src/rascal/math/kvec_generator.cc in the main branch
of librascal for more details on the algorithm. The API and the implementation
are essentially identical.
"""

import numpy as np
from ase.neighborlist import get_distance_matrix

class NeighborList():
    """
    Class for generating a neighbor list including periodic images
    for a single ase.Atoms frame.
    """
    def __init__(self, frame, species_dict, cutoff):
        self.frame = frame
        self.species_dict = species_dict
	self.cutoff = cutoff
        self.build_neighborlist()

    def build_neighborlist(self):
	"""
	Very primitive implementation of neighbor list not yet
	taking into account periodic images.
	"""
	# Initialization
	distance_matrix = self.frame.get_distance_matrix()
	chem_species = self.frame.get_chemical_species()
	num_atoms = len(distance_matrix)
	assert num_atoms == len(chem_species)
	num_species = len(self.species_dict)
	self.neighborlist = []

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
	    self.neighborlist.append(neighbors_i)
