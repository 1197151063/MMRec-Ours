import sys
from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from probe_collaborative_neighbors import similarity, neighbor_votes


class NeighborProbeTest(unittest.TestCase):
    def test_cosine_and_votes_and_joint_relabeling(self):
        r = sp.csr_matrix([[1., 1., 0., 0.], [1., 0., 1., 0.], [0., 0., 1., 1.], [0., 1., 0., 1.]])
        ties = np.array([.2, .6, .1, .8])
        s = similarity(r)
        dense = r.toarray()
        expected = dense @ dense.T / 2
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(s.toarray(), expected, atol=1e-7)
        v = neighbor_votes(r, s, k=2, ties=ties)
        p = np.array([2, 0, 3, 1]); q = np.array([3, 1, 0, 2])
        for kind in ('cosine', 'rare_cosine', 'jaccard'):
            original = neighbor_votes(r, similarity(r, kind), k=2, ties=ties)
            remapped = r[p][:, q]
            other = neighbor_votes(remapped, similarity(remapped, kind), k=2, ties=ties[p])
            np.testing.assert_allclose(other.toarray(), original.toarray()[p][:, q])
        self.assertTrue(np.isfinite(v.data).all())


if __name__ == '__main__':
    unittest.main()
