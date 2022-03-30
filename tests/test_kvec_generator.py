# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:32:41 2022

@author: kevin
"""

import numpy as np
import pytest
import scipy

from pylode.lib.kvec_generator import KvectorGenerator
from numpy.testing import assert_allclose

np.random.seed(12419)


class TestKvecgen:
    """Test correct behavior of k-vector generation.

    The tests used here are essentially the same the first two tests 
    implemented in librascal."""

    eps = 1e-2
    cutoffs = [
        1 + eps,
        np.sqrt(2) - eps,
        np.sqrt(2) + eps,
        np.sqrt(3) - eps,
        np.sqrt(3) + eps, 2 - eps,
        np.sqrt(5) - eps
    ]
    num_vectors_correct = [3, 3, 9, 9, 13, 13, 16, 16]

    cell_cubic = np.eye(3)
    cell_triclinic = np.array([[1, 0, 0], [5, 1, 0], [3, 4, 1]])

    Q = scipy.linalg.qr(np.random.normal(0, 1, (3, 3)))[0]
    cell_rotated = np.array([[1, 0, 0], [5, 1, 0], [3, 4, 1]]) @ Q

    @pytest.mark.parametrize("cell",
                             [cell_cubic, cell_triclinic, cell_rotated])
    def test_kvecgen(self, cell):
        """Use simple cell for which the solution is known exactly."""

        for ik, kcut in enumerate(self.cutoffs):
            # Generate k vectors
            kvecgen = KvectorGenerator(cell, kcut, is_reciprocal_cell=True)
            kvecgen.compute()
            kvectors = kvecgen.kvectors
            kvecnorms = kvecgen.kvector_norms

            # Check whether number of obtained vectors agrees with exact result
            assert len(kvectors) == kvecgen.kvector_number
            assert kvecgen.kvector_number == self.num_vectors_correct[ik]

            # Check that the obtained normes are indeed the norms of the
            # corresponding k-vectors and that they lie in the cutoff ball
            assert_allclose(kvecnorms, np.linalg.norm(kvectors, axis=1))
            assert (kvecnorms < kcut).all()
