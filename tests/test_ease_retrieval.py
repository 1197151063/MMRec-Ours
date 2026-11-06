import sys
from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from probe_ease_retrieval import fit_ease, evaluate


class EaseRetrievalTest(unittest.TestCase):
    def test_closed_form_matches_each_constrained_ridge_solution(self):
        x = np.random.default_rng(5).integers(0, 2, (18, 7)).astype(np.float64)
        b = fit_ease(sp.csr_matrix(x), 3.)
        expected = np.zeros_like(b)
        for column in range(7):
            keep = np.arange(7) != column
            design = x[:, keep]
            expected[keep, column] = np.linalg.solve(design.T @ design + 3 * np.eye(6), design.T @ x[:, column])
        np.testing.assert_allclose(b, expected, atol=1e-6)

    def test_masking_and_metrics_and_permutation(self):
        train = sp.csr_matrix(([1., 1.], ([0, 1], [0, 1])), shape=(2, 22))
        target = sp.csr_matrix(([1., 1.], ([0, 1], [2, 3])), shape=(2, 22))
        b = np.zeros((22, 22), dtype=np.float32)
        b[0, 0] = b[1, 1] = 100  # Must be excluded by the TRAIN mask.
        b[0, 2] = b[1, 3] = 10
        result = evaluate(train, target, b)
        self.assertEqual(result['recall@20'], 1.)
        self.assertEqual(result['ndcg@10'], 1.)
        perm = np.random.default_rng(7).permutation(22)
        self.assertEqual(evaluate(train[:, perm], target[:, perm], b[perm][:, perm]), result)


if __name__ == '__main__':
    unittest.main()
