from pathlib import Path
import sys
import unittest
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from probe_untruncated_paths import item_gram, untruncated_paths, pair_support_votes


class UntruncatedPathsTest(unittest.TestCase):
    def setUp(self):
        self.r = np.array([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0],
                           [0, 1, 1, 1, 0], [0, 0, 1, 0, 1],
                           [0, 0, 0, 0, 0]], dtype=np.float32)
        self.train = sp.csr_matrix(self.r)

    def test_sparse_association_equals_dense_eight_variants(self):
        users = np.array([3, 0, 4])
        di = np.maximum(self.r.sum(0), 1.)
        du = np.maximum(self.r.sum(1), 1.)
        for a in (0., 1.):
            for b in (0., 1.):
                for c in (0., .5):
                    dense = (self.r @ np.diag(di**-a) @ self.r.T @
                             np.diag(du**-b) @ self.r @ np.diag(di**-c))
                    sparse = untruncated_paths(self.train, users, a, b, c,
                                               item_gram(self.train, b))
                    np.testing.assert_allclose(sparse, dense[users], atol=1e-6)

    def test_common_pair_matches_dense_and_is_not_single_shared_item(self):
        users = np.arange(5)
        overlap = self.r @ self.r.T
        pairs = overlap * (overlap - 1.) * .5
        self.assertEqual(pairs[0, 2], 0.)
        self.assertEqual(pairs[0, 1], 1.)
        for weight in (None, .1, 1., 10.):
            expected = pairs @ self.r
            if weight is not None:
                expected = overlap @ self.r + weight * expected
            actual = pair_support_votes(self.train, users, weight)
            np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_user_item_permutations_only_permute_scores(self):
        up = np.array([2, 4, 1, 0, 3]); ip = np.array([4, 2, 0, 3, 1])
        other = self.train[up][:, ip]
        for score in (lambda r: untruncated_paths(r, np.arange(5), 1., 1., .5),
                      lambda r: pair_support_votes(r, np.arange(5), 1.)):
            np.testing.assert_allclose(score(other), score(self.train)[up][:, ip], atol=1e-6)

if __name__ == '__main__':
    unittest.main()
