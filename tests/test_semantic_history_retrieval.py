import sys
from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from probe_semantic_history_retrieval import (
    semantic_vectors, semantic_neighbors, collaborative_edges, edge_weights,
    history_scores, row_max_normalize,
)


class SemanticHistoryRetrievalTest(unittest.TestCase):
    def test_max_retains_individual_interest_instead_of_averaging(self):
        train = sp.csr_matrix([[1., 1., 0., 0.], [0., 0., 1., 0.]])
        edges = sp.csr_matrix([[0., 0., .9, .3], [0., 0., .2, .8], [0., .5, 0., 0.], [0., 0., 0., 0.]])
        expected_max = np.array([[0., 0., .9, .8], [0., .5, 0., 0.]])
        np.testing.assert_allclose(history_scores(train, edges, 'max').toarray(), expected_max)
        np.testing.assert_allclose(history_scores(train, edges).toarray(), train.toarray() @ edges.toarray())
        np.testing.assert_allclose(row_max_normalize(sp.csr_matrix([[0., 0.], [2., 4.]])).toarray(), [[0., 0.], [.5, 1.]])

    def test_simultaneous_entity_relabeling_preserves_scores(self):
        rng = np.random.default_rng(5)
        train = sp.csr_matrix((rng.random((7, 9)) > .65).astype(np.float32))
        visual, text = rng.normal(size=(9, 6)), rng.normal(size=(9, 4))
        ties = rng.random(9)
        user_perm, item_perm = rng.permutation(7), rng.permutation(9)
        def build(r, v, t, keys):
            graph = semantic_neighbors(semantic_vectors(v, t, True), k=3, block=2, ties=keys)
            evidence, degree = collaborative_edges(r)
            return history_scores(r, edge_weights(graph, evidence, degree, True), 'max').toarray()
        original = build(train, visual, text, ties)
        permuted = build(train[user_perm][:, item_perm], visual[item_perm], text[item_perm], ties[item_perm])
        np.testing.assert_allclose(permuted, original[user_perm][:, item_perm], atol=2e-7)

    def test_self_edges_excluded_and_finite_for_zero_features(self):
        features = np.array([[0., 0.], [1., 0.], [1., 0.], [0., 1.]])
        graph = semantic_neighbors(semantic_vectors(features, features), k=2)
        self.assertTrue(np.isfinite(graph.data).all())
        self.assertTrue(np.all(graph.diagonal() == 0))
        self.assertEqual(graph[0].nnz, 0)


if __name__ == '__main__':
    unittest.main()
