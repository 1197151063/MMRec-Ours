import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('history_mlp_probe', ROOT / 'experiments/probe_history_mlp.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class HistoryMLPTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        self.history = sp.csr_matrix(np.array([[1, 0, 1, 0, 1], [0, 1, 1, 0, 0],
                                               [1, 0, 0, 1, 0]], dtype=np.float32))

    def test_dense_equivalence_and_gradients_without_user_table(self):
        model = probe.HistoryMLP(5, 3, dropout=0.)
        items, offsets, values, rows, _ = probe.sparse_batch(self.history, np.arange(3))
        logits = model(items, offsets, values)
        dense = torch.from_numpy(self.history.toarray())
        dense = torch.nn.functional.normalize(dense, dim=1)
        expected = model.decoder(torch.relu(dense @ model.encoder.weight + model.encoder_bias))
        torch.testing.assert_close(logits, expected)
        loss = probe.multinomial_loss(logits, items, rows)
        loss.backward()
        self.assertEqual(set(dict(model.named_parameters())),
                         {'encoder.weight', 'encoder_bias', 'decoder.weight', 'decoder.bias'})
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_user_and_item_permutation_equivariance(self):
        model = probe.HistoryMLP(5, 4, dropout=.6).eval()
        items, offsets, values, _, _ = probe.sparse_batch(self.history, np.arange(3))
        original = model(items, offsets, values)
        pu, pi = np.array([2, 0, 1]), np.array([3, 1, 4, 0, 2])
        remapped = copy.deepcopy(model)
        with torch.no_grad():
            remapped.encoder.weight.copy_(model.encoder.weight[pi])
            remapped.decoder.weight.copy_(model.decoder.weight[pi])
            remapped.decoder.bias.copy_(model.decoder.bias[pi])
        r = self.history[pu][:, pi]
        items, offsets, values, _, _ = probe.sparse_batch(r, np.arange(3))
        torch.testing.assert_close(remapped(items, offsets, values), original[pu][:, pi])

    def test_test_edges_not_loaded_into_train_or_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny.inter'
            path.write_text('userID\titemID\tx_label\n0\t0\t0\n0\t1\t1\n0\t2\t2\n')
            train, valid = probe.load_train_valid(path)
            self.assertEqual(train.nnz, 1); self.assertEqual(valid.nnz, 1)
            self.assertEqual(train[0, 2], 0); self.assertEqual(valid[0, 2], 0)
            self.assertEqual(train[0, 0], 1); self.assertEqual(valid[0, 1], 1)

    def test_empty_corrupted_history_stays_finite(self):
        model = probe.HistoryMLP(5, 3, dropout=.99).train()
        r = sp.csr_matrix((2, 5), dtype=np.float32)
        items, offsets, values, _, _ = probe.sparse_batch(r, np.arange(2))
        self.assertTrue(torch.isfinite(model(items, offsets, values)).all())

    def test_early_stopping_uses_validation_only(self):
        metrics = {'recall@20': .1, 'ndcg@20': .05}
        with patch.object(probe, 'evaluate', return_value=metrics) as evaluator:
            result = probe.train_configuration(self.history, self.history, dropout=.3,
                                               hidden=3, epochs=20, batch=3,
                                               validate_every=1, patience=2)
        self.assertEqual(evaluator.call_count, 3)
        self.assertEqual(result['completed_epochs'], 3)
        self.assertEqual(result['selected']['epoch'], 1)
        self.assertEqual(result['stopping_reason'], 'VALID_Recall20_patience')

    def test_budget_before_first_validation_preserves_null_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'tiny'; data.mkdir()
            (data / 'tiny.inter').write_text('userID\titemID\tx_label\n'
                '0\t0\t0\n0\t1\t0\n1\t2\t0\n1\t3\t0\n0\t20\t1\n1\t21\t1\n')
            output = root / 'result.json'
            result = subprocess.run([sys.executable, str(ROOT / 'experiments/probe_history_mlp.py'),
                '--data-path', str(root), '--dataset', 'tiny', '--output', str(output),
                '--hidden', '3', '--epochs', '20', '--dropout', '.3', '--max-seconds', '.001'],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertIsNone(report['selected'])
            self.assertFalse(report['has_validated_winner'])


if __name__ == '__main__':
    unittest.main()
