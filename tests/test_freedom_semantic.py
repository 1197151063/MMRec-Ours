import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

import torch
from torch.nn import functional as F
import yaml

import test_simmrec as fixtures
from models.freedomalign import FreedomAlign
from models.freedomalignsemantic import FreedomAlignSemantic

ROOT = Path(__file__).resolve().parents[1]


class SemanticModelTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        options = yaml.safe_load((ROOT / 'src/configs/model/FreedomAlignSemantic.yaml').read_text())
        options.pop('hyper_parameters'); self.config.update(options)
        self.batch = torch.tensor([[0, 1, 2], [0, 2, 4], [9, 10, 11]])

    def test_exact_baseline_parameters_loss_gradients_and_rng(self):
        torch.manual_seed(3); base = FreedomAlign(self.config, self.loader)
        state = torch.get_rng_state().clone()
        torch.manual_seed(3)
        model = FreedomAlignSemantic(dict(self.config, semantic_beta=0.), self.loader)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for a, b in zip(base.parameters(), model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        loss_a, loss_b = base.calculate_loss(self.batch), model.calculate_loss(self.batch)
        torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
        loss_a.backward(); loss_b.backward()
        for a, b in zip(base.parameters(), model.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        base.eval(); model.eval()
        torch.testing.assert_close(base.full_sort_predict(self.batch[0]), model.full_sort_predict(self.batch[0]), rtol=0, atol=0)

    def test_bpr_uses_inference_score_and_unchanged_auxiliary_terms(self):
        model = FreedomAlignSemantic(self.config, self.loader)
        scores = model.full_sort_predict(self.batch[0])
        rows = torch.arange(len(self.batch[0]))
        expected = F.softplus(scores[rows, self.batch[2]] - scores[rows, self.batch[1]]).mean()
        u, i = model.forward(); users, pos, neg = self.batch
        for embedding, projector in ((model.image_embedding, model.image_trs), (model.text_embedding, model.text_trs)):
            z = projector(embedding.weight)
            expected = expected + .01 * model.modality_ssm(u[users], z[pos], z[neg], .1)
        expected = expected + .0005 * (.1 * model.item_alignment(pos, model.image_knn_idx, model.image_knn_weight)
                                      + .9 * model.item_alignment(pos, model.text_knn_idx, model.text_knn_weight))
        torch.testing.assert_close(model.calculate_loss(self.batch), expected)
        model.eval()
        torch.testing.assert_close(scores, model.full_sort_predict(self.batch[0]))
        self.assertIsNotNone(model._eval_semantic)
        model.train()
        self.assertIsNone(model._eval_semantic); self.assertIsNone(model._eval_embeddings)

    def test_freezing_preserves_features_but_trains_projectors(self):
        for beta in (0., .1):
            model = FreedomAlignSemantic(dict(self.config, freeze_modal_features=True, semantic_beta=beta), self.loader)
            before_v = model.image_embedding.weight.detach().clone()
            before_t = model.text_embedding.weight.detach().clone()
            before_proj = model.image_trs.weight.detach().clone()
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            model.calculate_loss(self.batch).backward()
            self.assertIsNone(model.image_embedding.weight.grad)
            self.assertIsNone(model.text_embedding.weight.grad)
            self.assertGreater(model.image_trs.weight.grad.abs().sum().item(), 0.)
            self.assertGreater(model.text_trs.weight.grad.abs().sum().item(), 0.)
            optimizer.step()
            torch.testing.assert_close(model.image_embedding.weight, before_v, rtol=0, atol=0)
            torch.testing.assert_close(model.text_embedding.weight, before_t, rtol=0, atol=0)
            self.assertFalse(torch.equal(before_proj, model.image_trs.weight))
            self.assertEqual(model.feature_drift()['visual_relative_l2'], 0.)
            self.assertEqual(model.feature_drift()['text_relative_l2'], 0.)
        trainable = FreedomAlignSemantic(self.config, self.loader)
        before = trainable.image_embedding.weight.detach().clone()
        optimizer = torch.optim.Adam(trainable.parameters(), lr=.001)
        trainable.calculate_loss(self.batch).backward(); optimizer.step()
        self.assertFalse(torch.equal(before, trainable.image_embedding.weight))
        self.assertGreater(trainable.feature_drift()['visual_relative_l2'], 0.)

    def test_bpr_alone_reaches_semantic_projectors(self):
        model = FreedomAlignSemantic(dict(self.config, modal_weight=0., ii_weight=0.), self.loader)
        loss = model.calculate_loss(self.batch)
        loss.backward()
        for parameter in (model.image_trs.weight, model.text_trs.weight,
                          model.user_embedding.weight, model.item_id_embedding.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_four_job_queue_no_checkpoints_with_source_inventory(self):
        workspace = Path(self.folder.name); data = workspace / 'data/baby'; data.mkdir(parents=True)
        for name in ('image_feat.npy', 'text_feat.npy'):
            shutil.copy(workspace / 'tiny' / name, data / name)
        lines = ['userID\titemID\tx_label']
        for u in range(3):
            lines.extend(f'{u}\t{i}\t0' for i in range(u * 8, (u + 1) * 8))
            lines.extend([f'{u}\t{((u + 1) * 8) % 24}\t1', f'{u}\t{((u + 1) * 8 + 1) % 24}\t2'])
        (data / 'baby.inter').write_text('\n'.join(lines) + '\n')
        output = workspace / 'runs'
        command = [sys.executable, str(ROOT / 'experiments/run_night.py'), '--suite', 'freedom_semantic',
                   '--data-path', str(data.parent), '--cpu', '--epochs', '1', '--hours', '.2', '--output', str(output)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=180)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        rows = json.loads((output / 'summary.json').read_text())
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row['state'] == 'complete' for row in rows), rows)
        self.assertFalse(list(output.rglob('*.pt')) + list(output.rglob('*.pth')))
        manifest = json.loads((output / 'manifest.json').read_text())
        name = 'src/models/freedomalignsemantic.py'
        self.assertEqual(manifest['source_files'][name], hashlib.sha256((ROOT / name).read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
