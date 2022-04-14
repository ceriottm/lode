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


class KvectorGenerator():
    """
    Class for generating k-vectors.

    Initialization with all the parameters defining the mathematical
    problem.

    Parameters
    ----------
    cell : numpy.ndarray
        3x3 matrix containing the cell vectors in the format:
        cell[0] = basis vector 1, cell[1] = basis vector 2, ...
    rcut : FLOAT
        Cutoff radius in reciprocal space units
    is_reciprocal_cell : BOOL, optional
        DESCRIPTION. The default is False.
    need_origin : BOOL, optional
        If set to true, the vector (0,0,0) will be included in the
        returned k-vectors.

    Attributes
    ----------
    kvectors : numpy.ndarray
        the k vectors
    kvector_norms : numpy.ndarray
        norms of the k vctors
    kvector_number : int
        Number of k vectors
    """
    def __init__(self, cell, rcut, is_reciprocal_cell=False,
                 need_origin = False):
        # Store the values of the quantities defining the mathematical problem
        if is_reciprocal_cell:
            self.cell = cell
        else:
            self.cell = 2 * np.pi * np.linalg.inv(cell.T)
        self.cutoff = rcut
        self.need_origin = need_origin

    def compute(self):
        cutoff_squared = self.cutoff * self.cutoff
        kcut = self.cutoff
        b1 = self.cell[0]
        b2 = self.cell[1]
        b3 = self.cell[2]

        # Define boundaries of optimal search box
        M = self.cell @ self.cell.T
        kvol = np.linalg.det(self.cell)
        n1max = int(np.floor(np.sqrt(M[1,1]*M[2,2] - M[1,2]**2) / kvol * kcut))
        n2max = int(np.floor(np.sqrt(M[2,2]*M[0,0] - M[2,0]**2) / kvol * kcut))
        n3max = int(np.floor(np.sqrt(M[0,0]*M[1,1] - M[0,1]**2) / kvol * kcut))

        kvecs = []
        knorms = []

        # If desired (e.g. for SOAP), include k=(0,0,0)
        if self.need_origin:
            kvecs.append(np.array([0.,0.,0.]))
            knorms.append(0.)

        # Start main loops
        for n3 in range(1, n3max+1):
            k = b3 * n3
            norm_squared = np.dot(k,k)
            if norm_squared < cutoff_squared:
                kvecs.append(k)
                knorms.append(np.sqrt(norm_squared))

        for n2 in range(1,n2max+1):
            for n3 in range(-n3max, n3max+1):
                k = n2*b2 + n3*b3
                norm_squared = np.dot(k,k)
                if norm_squared < cutoff_squared:
                    kvecs.append(k)
                    knorms.append(np.sqrt(norm_squared))

        for n1 in range(1, n1max+1):
            for n2 in range(-n2max, n2max+1):
                for n3 in range(-n3max, n3max+1):
                    k = n1*b1+n2*b2+n3*b3
                    norm_squared = np.dot(k,k)
                    if norm_squared < cutoff_squared:
                        kvecs.append(k)
                        knorms.append(np.sqrt(norm_squared))

        self.kvectors = np.array(kvecs)
        self.kvector_norms = np.array(knorms)
        self.kvector_number = len(self.kvectors)
