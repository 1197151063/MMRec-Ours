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
from common.spectral_user_groups import build_spectral_user_groups
from models.freedom import FREEDOM
from models.freedomspectralgroup import FreedomSpectralGroup

ROOT = Path(__file__).resolve().parents[1]


class SpectralGroupsTest(unittest.TestCase):
    def test_recovers_disconnected_groups_and_handles_isolated_user(self):
        r = sp.block_diag([np.ones((6, 6)), np.ones((6, 6))], format='csr', dtype=np.float32)
        r = sp.vstack((r, sp.csr_matrix((1, 12))), format='csr')
        labels = build_spectral_user_groups(r, groups=2, rank=2)
        self.assertTrue(np.all(labels[:6] == labels[0]))
        self.assertTrue(np.all(labels[6:12] == labels[6]))
        self.assertNotEqual(labels[0], labels[6])
        self.assertEqual(labels[-1], -1)
        permutation = np.random.default_rng(3).permutation(13)
        other = build_spectral_user_groups(r[permutation], groups=2, rank=2)
        inverse = np.argsort(permutation)
        self.assertTrue(np.array_equal(labels[:, None] == labels,
                                       other[inverse, None] == other[inverse]))


class SpectralGroupModelTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        self.config.update(yaml.safe_load((ROOT / 'src/configs/model/FreedomSpectralGroup.yaml').read_text()))
        self.config.update(embedding_size=6, feat_embed_dim=6, knn_k=4, num_groups=2,
                           spectral_rank=2, dropout=0., cf_model='lightgcn')

    def test_zero_switch_and_group_gradients_and_personal_regularizer(self):
        batch = torch.tensor([[0, 1, 2], [0, 2, 4], [10, 11, 12]])
        torch.manual_seed(5)
        base = FREEDOM(self.config, self.loader)
        torch.manual_seed(5)
        control = FreedomSpectralGroup(dict(self.config, group_strength=0.), self.loader)
        base.pre_epoch_processing(); control.pre_epoch_processing()
        for a, b in zip(base.calculate_loss(batch), control.calculate_loss(batch)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        model = FreedomSpectralGroup(dict(self.config, personal_reg_weight=.1), self.loader)
        model.pre_epoch_processing()
        unregularized = FREEDOM.calculate_loss(model, batch)[0]
        expected = .1 * model.user_embedding.weight.square().sum(-1).mean()
        total = model.calculate_loss(batch)[0]
        torch.testing.assert_close(total - unregularized, expected)
        total.backward()
        self.assertGreater(float(model.group_embedding.weight.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(model.group_embedding.weight.grad).all())
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in base.parameters()) + 2 * 6)
        u, i = model.forward(model.norm_adj)
        torch.testing.assert_close(model.full_sort_predict([batch[0]]), u[batch[0]] @ i.T)

    def test_queue_has_no_baseline_and_completes_one_cpu_epoch(self):
        workspace = Path(self.folder.name)
        data = workspace / 'data/baby'; data.mkdir(parents=True)
        for name in ('image_feat.npy', 'text_feat.npy'):
            shutil.copy(workspace / 'tiny' / name, data / name)
        lines = ['userID\titemID\tx_label']
        for u in range(32):
            lines.extend(f'{u}\t{i}\t0' for i in {0, *((u + j) % 21 + 1 for j in range(7))})
            lines.extend([f'{u}\t22\t1', f'{u}\t23\t2'])
        (data / 'baby.inter').write_text('\n'.join(lines) + '\n')
        output = workspace / 'runs'
        command = [sys.executable, str(ROOT / 'experiments/run_night.py'), '--suite', 'freedom_spectral_group',
                   '--data-path', str(data.parent), '--cpu', '--epochs', '1', '--hours', '.1', '--limit', '1',
                   '--output', str(output)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = json.loads((output / 'summary.json').read_text())
        self.assertEqual(rows[0]['state'], 'complete', rows)
        self.assertFalse(list(output.rglob('*.pth')))


if __name__ == '__main__':
    unittest.main()
