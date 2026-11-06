import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

import numpy as np
import scipy.sparse as sp
import torch
import yaml

import test_simmrec as fixtures
from common.user_motifs import ItemGroupPeers, interest_distance, build_bridges
from models.freedom import FREEDOM
from models.freedommotif import FreedomMotif

ROOT = Path(__file__).resolve().parents[1]


class MotifOperatorsTest(unittest.TestCase):
    def test_peer_sampling_is_other_consumer_in_training_group(self):
        r = sp.csr_matrix([[1, 1], [1, 0], [0, 0]], dtype=np.float32)
        sampler = ItemGroupPeers(r)
        u = torch.tensor([0, 1, 0]); i = torch.tensor([0, 0, 1])
        for _ in range(5):
            peer, valid, degree = sampler.sample(u, i)
            self.assertEqual(peer.tolist(), [1, 0, 0])
            self.assertEqual(valid.tolist(), [True, True, False])
        with self.assertRaises(ValueError):
            sampler.sample(torch.tensor([2]), torch.tensor([0]))

    def test_shared_interest_loss_preserves_orthogonal_preferences(self):
        users = torch.tensor([[1., 1., 0.]], requires_grad=True)
        peers = torch.tensor([[1., -1., 0.]], requires_grad=True)
        direction = torch.tensor([[1., 0., 0.]], requires_grad=True)
        loss = interest_distance(users, peers, [direction])
        torch.testing.assert_close(loss, torch.zeros_like(loss))
        self.assertGreater(float(interest_distance(users, peers)), 0)
        loss.sum().backward()
        self.assertIsNone(direction.grad)

    def test_bridge_is_nonoverlap_fourhop_and_relabels(self):
        r = sp.csr_matrix([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32)
        ties = np.array([.2, .4, .1, .3])
        ids, weights = build_bridges(r, k=2, ties=ties)
        self.assertEqual(ids[0, 0], 1)
        self.assertEqual(ids[1, 0], 0)
        self.assertEqual(float(weights[0, 0]), 1.)
        self.assertEqual(float(weights[3].sum()), 0.)
        for u in range(4):
            for v in ids[u][weights[u] > 0]:
                self.assertNotEqual(u, v)
                self.assertEqual(r[u].multiply(r[v]).nnz, 0)
        perm = np.array([3, 1, 0, 2])
        pids, pw = build_bridges(r[perm], k=2, ties=ties[perm])
        a = sp.csr_matrix((weights.ravel(), (np.repeat(np.arange(4), 2), ids.ravel())), shape=(4, 4))
        b = sp.csr_matrix((pw.ravel(), (np.repeat(np.arange(4), 2), pids.ravel())), shape=(4, 4))
        np.testing.assert_allclose(a[perm][:, perm].toarray(), b.toarray())


class FreedomMotifTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        c = yaml.safe_load((ROOT / 'src/configs/model/FreedomMotif.yaml').read_text())
        c.update(embedding_size=6, feat_embed_dim=6, knn_k=4, dropout=0., cf_model='lightgcn')
        self.config.update(c)
        self.loader.inter_matrix = lambda form: sp.coo_matrix(
            (np.ones(6), ([0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 1, 3])), shape=(3, 24))

    def test_disabled_loss_exact_and_active_loss_backpropagates(self):
        batch = torch.tensor([[0, 1, 2], [0, 0, 1], [10, 11, 12]])
        torch.manual_seed(19)
        base = FREEDOM(self.config, self.loader)
        torch.manual_seed(19)
        disabled = FreedomMotif(dict(self.config, star_weight=0., bridge_weight=0.), self.loader)
        base.pre_epoch_processing(); disabled.pre_epoch_processing()
        for a, b in zip(base.calculate_loss(batch), disabled.calculate_loss(batch)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for mode in ('semantic', 'full'):
            model = FreedomMotif(dict(self.config, star_mode=mode, bridge_weight=.01), self.loader)
            model.pre_epoch_processing()
            loss = model.calculate_loss(batch)[0]
            self.assertGreater(model.last_motif_terms['star'], 0.)
            self.assertGreater(model.last_motif_terms['bridge'], 0.)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in base.parameters()))
            self.assertTrue(torch.isfinite(model.user_embedding.weight.grad).all())
            u, i = model.forward(model.norm_adj)
            torch.testing.assert_close(model.full_sort_predict([batch[0]]), u[batch[0]] @ i.T)

    def test_queue_entry_runs_and_does_not_save_model(self):
        workspace = Path(self.folder.name)
        data = workspace / 'data/baby'; data.mkdir(parents=True)
        for name in ('image_feat.npy', 'text_feat.npy'):
            shutil.copy(workspace / 'tiny' / name, data / name)
        lines = ['userID\titemID\tx_label']
        for u in range(3):
            lines.extend(f'{u}\t{i}\t0' for i in {0, *range(u * 7 + 1, u * 7 + 8)})
            lines.extend([f'{u}\t22\t1', f'{u}\t23\t2'])
        (data / 'baby.inter').write_text('\n'.join(lines) + '\n')
        output = workspace / 'runs'
        command = [sys.executable, str(ROOT / 'experiments/run_night.py'), '--suite', 'freedom_motif',
                   '--data-path', str(data.parent), '--cpu', '--epochs', '1', '--hours', '.1',
                   '--limit', '1', '--output', str(output)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        rows = json.loads((output / 'summary.json').read_text())
        self.assertEqual(rows[0]['state'], 'complete', rows)
        self.assertFalse(list(output.rglob('*.pth')))


if __name__ == '__main__':
    unittest.main()
