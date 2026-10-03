import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from research_user_cohorts import (ResidualScores, binary, build_neighbor_rows,
                                  collaborative_features, load_train, mask_training,
                                  matched_pairs, normalize, vote_metrics)


class UserCohortsTest(unittest.TestCase):
    def test_remove_all_common_and_empty_residual(self):
        r = binary(sp.csr_matrix([[1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 0, 0]]))
        # Common items have huge same-direction vectors; they must not dominate.
        features = np.array([[100, 100], [100, 100], [1, 0], [-1, 0]], dtype=np.float32)
        scores, counts = ResidualScores(r, {'visual': features}).compare(0, [1, 2])
        np.testing.assert_array_equal(counts, [2, 2])
        self.assertAlmostEqual(scores['visual'][0], -1)
        self.assertTrue(np.isnan(scores['visual'][1]))

    def test_exact_matched_controls(self):
        r = binary(sp.csr_matrix([[1, 1, 1, 0, 0], [1, 0, 0, 1, 1],
                                  [0, 1, 0, 1, 0], [1, 0, 0, 1, 1],
                                  [1, 1, 0, 1, 0], [1, 0, 0, 1, 1]]))
        pairs, stats = matched_pairs(r, 1, 100, 2, np.random.default_rng(5))
        self.assertGreater(len(pairs), 0)
        self.assertEqual(stats['sampled_shared_pairs'], len(pairs) + stats['unmatched_pairs'])
        degree = np.diff(r.indptr)
        for u, v, w, anchor, overlap in pairs:
            self.assertEqual(v - u, 1)
            self.assertGreater(abs(w - u), 2)
            self.assertEqual(degree[v], degree[w])
            self.assertTrue(r[u, anchor] and r[v, anchor] and r[w, anchor])
            self.assertEqual(r[u].multiply(r[v]).sum(), overlap)
            self.assertEqual(r[u].multiply(r[w]).sum(), overlap)

    def test_mask_and_collaborative_features(self):
        r = binary(sp.csr_matrix([[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 0]]))
        fit, targets = mask_training(r, 2026)
        self.assertEqual(r.nnz - fit.nnz, 2)
        for u in np.flatnonzero(targets >= 0):
            self.assertEqual(fit[u, targets[u]], 0)
            self.assertEqual(r[u, targets[u]], 1)
        self.assertEqual(targets[2], -1)
        probes = np.random.default_rng(2).normal(size=(3, 4))
        expected = normalize(fit.toarray().T @ (probes / np.sqrt(np.maximum(np.diff(fit.indptr), 1))[:, None]))
        np.testing.assert_allclose(collaborative_features(fit, 4, probes), expected)

    def test_graph_relabeling_equivariance(self):
        rng = np.random.default_rng(7)
        r = binary(sp.csr_matrix((rng.random((12, 15)) < .4).astype(float)))
        content = {name: normalize(rng.normal(size=(15, 6))) for name in ('visual', 'text')}
        probes = rng.normal(size=(12, 6)); ties = rng.random(12)
        queries = np.array([0, 3, 7])
        graphs, _ = build_neighbor_rows(r, content, probes, queries, [3], ties, 6)
        permutation = rng.permutation(12)
        inverse = np.argsort(permutation)
        other, _ = build_neighbor_rows(r[permutation], content, probes[permutation], inverse[queries],
                                       [3], ties[permutation], 6)
        for key in graphs:
            np.testing.assert_allclose(graphs[key].toarray(), other[key][:, inverse].toarray())

    def test_votes_do_not_fill_zero_scores_or_recommend_history(self):
        r = binary(sp.csr_matrix([[1, 0, 0], [1, 1, 0], [1, 0, 0]]))
        graph = sp.csr_matrix([[0, 1, 0], [1, 0, 0]], dtype=float)
        metrics = vote_metrics(graph, r, np.array([0, 2]), np.array([1, -1, 2]), np.array([.1, .2, .3]))
        np.testing.assert_array_equal(metrics['recall@20'], [1, 0])
        np.testing.assert_array_equal(metrics['ndcg@20'], [1, 0])

    def test_cli_and_heldout_rows_do_not_enter_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root / 'data/baby'; folder.mkdir(parents=True)
            train = 'userID\titemID\tx_label\n' + ''.join(
                f'{u}\t{(u + j) % 12}\t0\n' for u in range(16) for j in range(4))
            inter = folder / 'baby.inter'
            inter.write_text(train + '100\t100\t1\n200\t200\t2\n')
            for name in ('image_feat.npy', 'text_feat.npy'):
                np.save(folder / name, np.random.default_rng(8).normal(size=(12, 9)).astype(np.float32))
            before, _, _ = load_train(folder, 'baby')
            inter.write_text(train + '999\t999\t2\n')
            after, _, _ = load_train(folder, 'baby')
            self.assertEqual((before != after).nnz, 0)
            result = subprocess.run([sys.executable, str(ROOT / 'experiments/research_user_cohorts.py'),
                                     '--data-path', str(folder.parent), '--output', str(root / 'out'),
                                     '--offsets', '1', '2', '--far-distance', '3', '--pairs', '30',
                                     '--queries', '10', '--dim', '6', '--ks', '2', '4',
                                     '--mask-seeds', '2026', '2027'], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / 'out/report.json').read_text())
            self.assertEqual(report['training_edges'], 64)
            self.assertEqual(len(report['prediction']), 2)
            self.assertEqual(len(report['prediction'][0]['methods']), 10)
            self.assertEqual(json.loads((root / 'out/status.json').read_text())['state'], 'complete')
            self.assertFalse(list((root / 'out').rglob('*.pt')))


if __name__ == '__main__':
    unittest.main()
