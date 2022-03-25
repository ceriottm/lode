# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:32:41 2022

@author: kevin
"""

import numpy as np

from pylode.lib.kvec_generator import KvectorGenerator


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

    def test_kvecgen_cubic(self):
        """Use simple cubic cell for which the solution is known exactly."""
        cell = np.eye(3)

        for ik, kcut in enumerate(self.cutoffs):
            # Generate k vectors
            kvecgen = KvectorGenerator(cell, kcut, is_reciprocal_cell=True)
            kvecgen.compute()
            kvectors = kvecgen.kvectors
            kvecnorms = kvecgen.kvector_norms

            # Check whether number of obtained vectors agrees with exact result
            assert len(kvectors) == kvecgen.num_kvecs
            assert kvecgen.num_kvecs == self.num_vectors_correct[ik]

            # Check that the obtained normes are indeed the norms of the
            # corresponding k-vectors and that they lie in the cutoff ball
            assert (kvecnorms == np.linalg.norm(kvectors, axis=1)).all()
            assert (kvecnorms < kcut).all()

    def test_kvecgen_triclinic(self):
        """Use triclinic input vectors (describing the same cell as above)."""
        cell = np.array([[1, 0, 0], [5, 1, 0], [3, 4, 1]])
        for ik, kcut in enumerate(self.cutoffs):
            # Generate k vectors
            kvecgen = KvectorGenerator(cell, kcut, is_reciprocal_cell=True)
            kvecgen.compute()
            kvectors = kvecgen.kvectors
            kvecnorms = kvecgen.kvector_norms

            # Check whether number of obtained vectors agrees with exact result
            assert len(kvectors) == kvecgen.num_kvecs
            assert kvecgen.num_kvecs == self.num_vectors_correct[ik]

            # Check that the obtained normes are indeed the norms of the
            # corresponding k-vectors and that they lie in the cutoff ball
            assert (kvecnorms == np.linalg.norm(kvectors, axis=1)).all()
            assert (kvecnorms < kcut).all()

    def test_kvecgen_rotated(self):
        """Use rotated cell"""
        # Generate random rotation matrix
        np.random.seed(12419)
        import scipy
        Q, R = scipy.linalg.qr(np.random.normal(0, 1, (3, 3)))
        assert np.linalg.norm(Q.T @ Q - np.eye(3)) < 1e-13
        cell = np.array([[1, 0, 0], [5, 1, 0], [3, 4, 1]]) @ Q
        for ik, kcut in enumerate(self.cutoffs):
            # Generate k vectors
            kvecgen = KvectorGenerator(cell, kcut, is_reciprocal_cell=True)
            kvecgen.comput()
            kvectors = kvecgen.kvectors()
            kvecnorms = kvecgen.kvector_norms()

            # Check whether number of obtained vectors agrees with exact result
            assert len(kvectors) == kvecgen.num_kvecs
            assert kvecgen.num_kvecs == self.num_vectors_correct[ik]

            # Check that the obtained normes are indeed the norms of the
            # corresponding k-vectors and that they lie in the cutoff ball
            assert (kvecnorms < kcut).all()
