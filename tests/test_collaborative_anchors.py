"""Tests for TRAIN-only conditional anchor scores, including index relabeling."""
from pathlib import Path
import sys
import unittest
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from probe_collaborative_anchors import item_evidence, pool_evidence, bayes_evidence


class CollaborativeAnchorProbeTest(unittest.TestCase):
    def setUp(self):
        self.train = sp.csr_matrix(np.array([
            [1, 1, 0, 0, 0], [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0], [0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0]], dtype=np.float32))

    def test_scores_equivariant_under_user_and_item_relabeling(self):
        up = np.array([2, 4, 1, 0, 3]); ip = np.array([4, 2, 0, 3, 1])
        other = self.train[up][:, ip]
        for mode in ('cosine', 'conditional', 'lift'):
            first, _ = item_evidence(self.train, mode, shrink=2.)
            second, _ = item_evidence(other, mode, shrink=2.)
            np.testing.assert_allclose(second.toarray(), first.toarray()[ip][:, ip], atol=1e-6)
            self.assertTrue(np.all(first.diagonal() == 0))
            for pool in ('sum', 'max', 'top2', 'top3', 'geomean2', 'and2', 'noisy_or'):
                a = pool_evidence(self.train, first, np.arange(5), pool)
                b = pool_evidence(other, second, np.arange(5), pool)
                np.testing.assert_allclose(b, a[up][:, ip], atol=1e-6)
                self.assertTrue(np.isfinite(b).all())

    def test_bayes_matches_explicit_probability(self):
        alpha = 10.
        evidence, correction = bayes_evidence(self.train, alpha)
        degrees = np.asarray(self.train.sum(0)).ravel()
        prob = degrees / degrees.sum()
        co = (self.train.T @ self.train).toarray()
        np.fill_diagonal(co, 0.)
        explicit = np.log((co + alpha * prob[:, None]) /
                          (degrees[None, :] + alpha)) - np.log(prob[:, None])
        compact = evidence.toarray() - correction[None, :]
        np.testing.assert_allclose(compact, explicit, atol=1e-6)
        self.assertTrue(np.isfinite(compact).all())

    def test_unconsumed_candidate_cannot_receive_self_interaction(self):
        # A query's own co-occurrence contribution only joins consumed items;
        # after masking history it cannot trivially leak an unseen edge.
        evidence, _ = item_evidence(self.train, 'conditional', power=0.)
        changed = self.train.copy().tolil(); changed[0] = 0.; changed = changed.tocsr()
        other, _ = item_evidence(changed, 'conditional', power=0.)
        for candidate in (2, 3, 4):
            before = evidence[[0, 1], candidate].toarray().ravel()
            after = other[[0, 1], candidate].toarray().ravel()
            # Counts unchanged, source degree changes; compare recovered counts.
            before *= np.asarray(self.train.sum(0)).ravel()[[0, 1]]
            after *= np.asarray(changed.sum(0)).ravel()[[0, 1]]
            np.testing.assert_allclose(before, after)

if __name__ == '__main__':
    unittest.main()
