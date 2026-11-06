import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
import yaml

import test_simmrec as fixtures
from common.user_order_prior import UserOrderPrior
from models.freedom import FREEDOM
from models.freedomorderprior import FreedomOrderPrior

ROOT = Path(__file__).resolve().parents[1]


class OrderSamplerTest(unittest.TestCase):
    def test_near_and_far_support_boundary_and_inactive_masks(self):
        active = torch.ones(20, dtype=torch.bool); active[[2, 6, 17, 18, 19]] = False
        prior = UserOrderPrior(active, bandwidth=2., radius=3, samples=128, far_gap=7)
        anchors = torch.tensor([0, 5, 16])
        near, valid = prior.sample_near(anchors)
        self.assertTrue(valid.all()); self.assertTrue(active[near].all())
        distance = (near - anchors[:, None]).abs()
        self.assertTrue(((distance > 0) & (distance <= 3)).all())
        far, valid = prior.sample_far(anchors)
        self.assertTrue(valid.all()); self.assertTrue(active[far].all())
        self.assertTrue(((far - anchors[:, None]).abs() >= 7).all())
        isolated = UserOrderPrior([True, False, False, False, True], radius=1)
        _, valid = isolated.sample_near(torch.tensor([0, 4]))
        self.assertFalse(valid.any())

    def test_losses_and_gradients_match_explicit_pairs(self):
        prior = UserOrderPrior([True] * 4, radius=1, samples=1, far_gap=2)
        embeddings = torch.tensor([[1., 0.], [.8, .2], [-.5, 1.], [-1., 0.]], requires_grad=True)
        anchors = torch.tensor([0, 1])
        neighbors = torch.tensor([[1], [2]])
        fars = torch.tensor([[3], [3]])
        valid = torch.tensor([True, True])
        with patch.object(prior, 'sample_near', return_value=(neighbors, valid)):
            expected_near = F.cosine_similarity(embeddings[anchors], embeddings[neighbors[:, 0]])
            torch.testing.assert_close(prior(embeddings, anchors), (1 - expected_near).mean())
            with patch.object(prior, 'sample_far', return_value=(fars, valid)):
                expected_far = F.cosine_similarity(embeddings[anchors], embeddings[fars[:, 0]])
                loss = prior(embeddings, anchors, mode='local_rank', margin=.1, temperature=.1)
                torch.testing.assert_close(loss, (.1 * F.softplus((expected_far - expected_near + .1) / .1)).mean())
                loss.backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertGreater(float(embeddings.grad.abs().sum()), 0.)
        isolated = UserOrderPrior([True, False, False, True], radius=1)
        self.assertEqual(float(isolated(embeddings, torch.tensor([0, 3]))), 0.)


class OrderPriorModelTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        self.config.update(yaml.safe_load((ROOT / 'src/configs/model/FreedomOrderPrior.yaml').read_text()))
        self.config.update(embedding_size=6, feat_embed_dim=6, knn_k=4, dropout=0.,
                           cf_model='lightgcn', order_radius=1, order_far_gap=2)

    def test_zero_weight_regression_and_exact_regularized_objective(self):
        batch = torch.tensor([[0, 1, 2], [0, 2, 4], [10, 11, 12]])
        torch.manual_seed(5); base = FREEDOM(self.config, self.loader)
        torch.manual_seed(5); control = FreedomOrderPrior(dict(self.config, order_weight=0.), self.loader)
        base.pre_epoch_processing(); control.pre_epoch_processing()
        for actual, expected in zip(control.calculate_loss(batch), base.calculate_loss(batch)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for mode in ('attract', 'local_rank'):
            model = FreedomOrderPrior(dict(self.config, order_mode=mode), self.loader)
            model.pre_epoch_processing()
            with patch.object(model.order_prior, 'forward', return_value=torch.tensor(.7)):
                total, bpr, modal = model.calculate_loss(batch)
                torch.testing.assert_close(total, bpr + model.reg_weight * modal + .07)
            model.calculate_loss(batch)[0].backward()
            self.assertTrue(torch.isfinite(model.user_embedding.weight.grad).all())
            self.assertGreater(float(model.user_embedding.weight.grad.abs().sum()), 0.)
            self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in base.parameters()))
            u, i = model.forward(model.norm_adj)
            torch.testing.assert_close(model.full_sort_predict([batch[0]]), u[batch[0]] @ i.T)

    def test_all_eight_queue_jobs_complete_without_checkpoints(self):
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
        command = [sys.executable, str(ROOT / 'experiments/run_night.py'), '--suite', 'freedom_order_prior',
                   '--data-path', str(data.parent), '--output', str(output), '--cpu', '--epochs', '1', '--hours', '.1']
        run = subprocess.run(command, capture_output=True, text=True, timeout=180)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual([row['state'] for row in summary], ['complete'] * 8, summary)
        self.assertFalse(list(output.rglob('*.pth')))
        self.assertTrue(all('EXPLICIT' in row['hypothesis'] for row in summary))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_constructor_and_order_gradients(self):
        device = torch.device('cuda:0')
        for mode in ('attract', 'local_rank'):
            model = FreedomOrderPrior(dict(self.config, device=device, order_mode=mode), self.loader).to(device)
            model.pre_epoch_processing()
            batch = torch.tensor([[0, 1, 2], [0, 2, 4], [10, 11, 12]], device=device)
            model.calculate_loss(batch)[0].backward()
            self.assertTrue(torch.isfinite(model.user_embedding.weight.grad).all())
            self.assertEqual(model.order_prior.offsets.device, device)


if __name__ == '__main__':
    unittest.main()
