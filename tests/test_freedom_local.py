import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

import torch
import yaml

import test_simmrec as fixtures
from models.freedom import FREEDOM
from models.freedomlocal import FreedomLocal

ROOT = Path(__file__).resolve().parents[1]


class FreedomLocalTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        self.config.update(yaml.safe_load((ROOT / 'src/configs/model/FreedomLocal.yaml').read_text()))
        self.config.update(embedding_size=6, feat_embed_dim=6, knn_k=4, dropout=0., cf_model='lightgcn')

    def test_forward_unchanged_zero_weight_and_exact_objective(self):
        batch = torch.tensor([[0, 1, 2], [0, 2, 4], [10, 11, 12]])
        torch.manual_seed(5); base = FREEDOM(self.config, self.loader)
        torch.manual_seed(5); model = FreedomLocal(self.config, self.loader)
        base.pre_epoch_processing(); model.pre_epoch_processing()
        torch.testing.assert_close(base.full_sort_predict([batch[0]]), model.full_sort_predict([batch[0]]), rtol=0, atol=0)
        base_loss = base.calculate_loss(batch)
        total, bpr, modal = model.calculate_loss(batch)
        torch.testing.assert_close(total, base_loss[0] + model.local_weight * model.local_loss())
        torch.testing.assert_close(bpr, base_loss[1], rtol=0, atol=0)
        torch.testing.assert_close(modal, base_loss[2], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in base.parameters()))
        model.local_weight = 0.
        for actual, expected in zip(model.calculate_loss(batch), base_loss):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_formula_gradient_descent_and_empty_edges(self):
        model = FreedomLocal(self.config, self.loader)
        p = model.user_embedding.weight
        expected = ((p[1] - model.local_rho * p[0]).square().mean() +
                    (p[2] - model.local_rho * p[1]).square().mean()) / 2 / model.local_scale
        loss = model.local_loss()
        torch.testing.assert_close(loss, expected)
        fixed_scale = model.local_scale.clone()
        before = float(loss.detach())
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        with torch.no_grad():
            p -= .05 * p.grad
        self.assertLess(float(model.local_loss().detach()), before)
        torch.testing.assert_close(model.local_scale, fixed_scale, rtol=0, atol=0)
        model.local_pair_mask.zero_()
        self.assertEqual(float(model.local_loss().detach()), 0.)

    def test_three_job_queue_without_checkpoints(self):
        workspace = Path(self.folder.name)
        data = workspace / 'data/baby'; data.mkdir(parents=True)
        for name in ('image_feat.npy', 'text_feat.npy'):
            shutil.copy(workspace / 'tiny' / name, data / name)
        rows = ['userID\titemID\tx_label']
        for u in range(3):
            rows.extend(f'{u}\t{i}\t0' for i in range(u * 7, u * 7 + 7))
            rows.extend([f'{u}\t22\t1', f'{u}\t23\t2'])
        (data / 'baby.inter').write_text('\n'.join(rows) + '\n')
        output = workspace / 'runs'
        command = [sys.executable, str(ROOT / 'experiments/run_night.py'), '--suite', 'freedom_local',
                   '--data-path', str(data.parent), '--output', str(output), '--cpu', '--epochs', '1', '--hours', '.1']
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual([row['state'] for row in summary], ['complete'] * 3, summary)
        self.assertFalse(list(output.rglob('*.pth')))
        jobs = json.loads((output / 'manifest.json').read_text())['jobs']
        self.assertEqual([j['overrides']['local_weight'] for j in jobs], [.01, .1, 1.])
        self.assertTrue(all(j['overrides']['seed'] == 999 for j in jobs))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_gpu_constructor_and_local_gradient(self):
        device = torch.device('cuda:0')
        model = FreedomLocal(dict(self.config, device=device), self.loader).to(device)
        model.pre_epoch_processing()
        batch = torch.tensor([[0, 1, 2], [0, 2, 4], [10, 11, 12]], device=device)
        model.calculate_loss(batch)[0].backward()
        self.assertEqual(model.local_pair_mask.device, device)
        self.assertTrue(torch.isfinite(model.user_embedding.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
