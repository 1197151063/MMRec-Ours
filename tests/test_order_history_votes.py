import sys
from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from common.order_history_votes import build_order_history_votes
from probe_order_history_votes import evaluate_votes


class HistoryVotesTest(unittest.TestCase):
    def test_sparse_votes_match_dense_explicit_distance_kernel(self):
        r = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float32)
        distance = np.abs(np.arange(4)[:, None] - np.arange(4))
        kernel = np.exp(-distance / 2.) * ((distance > 0) & (distance <= 2))
        for power in (0., 1.):
            expected = kernel @ (r / r.sum(1)[:, None] ** power)
            actual = build_order_history_votes(sp.csr_matrix(r), 2., radius=2, user_degree_power=power)
            np.testing.assert_allclose(actual.toarray(), expected, atol=1e-6)
        single = sp.csr_matrix(([1.], ([0], [1])), shape=(4, 3))
        self.assertEqual(float(build_order_history_votes(single, 1.)[0].sum()), 0.)

    def test_recommend_shared_neighbor_item_and_mask_own_history(self):
        r = sp.csr_matrix(([1., 1., 1.], ([0, 1, 2], [0, 2, 3])), shape=(3, 22))
        targets = sp.csr_matrix(([1.], ([0], [2])), shape=(3, 22))
        votes = build_order_history_votes(r, 1., radius=1)
        self.assertEqual(evaluate_votes(r, targets, votes)['ndcg@10'], 1.)
        votes[0, 0] = 100.
        self.assertEqual(evaluate_votes(r, targets, votes)['ndcg@10'], 1.)


if __name__ == '__main__':
    unittest.main()
