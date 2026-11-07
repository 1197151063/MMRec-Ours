import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest import mock

import torch
from torch.nn import functional as F
import yaml

import test_simmrec as fixtures
from models.freedomlocalssm import FreedomLocalSSM

ROOT = Path(__file__).resolve().parents[1]


class FreedomLocalSSMTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        self.config.update(yaml.safe_load((ROOT / 'src/configs/model/FreedomLocalSSM.yaml').read_text()))
        self.config.update(embedding_size=6, feat_embed_dim=6, knn_k=4, dropout=0.)
        self.batch = torch.tensor([[0, 1, 2], [0, 2, 4], [10, 11, 12]])

    def test_id_only_forward_and_no_item_graph(self):
        model = FreedomLocalSSM(self.config, self.loader)
        self.assertFalse(hasattr(model, 'mm_adj'))
        torch.testing.assert_close(model.item_representations(), model.item_id_embedding.weight)
        x = torch.cat((model.user_embedding.weight, model.item_id_embedding.weight))
        ax = torch.sparse.mm(model.norm_adj, x)
        expected = (x + ax + torch.sparse.mm(model.norm_adj, ax)) / 3
        torch.testing.assert_close(torch.cat(model.forward()), expected)
        before = model.full_sort_predict(self.batch[0])
        # Auxiliary modality projectors have no direct path into forward scores.
        with torch.no_grad():
            model.image_trs.weight.mul_(100)
            model.text_trs.bias.add_(50)
        torch.testing.assert_close(model.full_sort_predict(self.batch[0]), before, rtol=0, atol=0)
        torch.testing.assert_close(before, expected[:3][self.batch[0]] @ expected[3:].T)

    def test_exact_bpr_two_ssms_ii_local_and_gradients(self):
        model = FreedomLocalSSM(self.config, self.loader)
        model.pre_epoch_processing()
        users, pos, neg = self.batch
        u, i = model.forward(model.masked_adj)
        bpr = F.softplus((u[users] * (i[neg] - i[pos])).sum(-1)).mean()
        visual = model.image_trs(model.image_embedding.weight)
        textual = model.text_trs(model.text_embedding.weight)
        def expected_ssm(user, item):
            scores = F.normalize(user, dim=-1) @ F.normalize(item, dim=-1).t() / model.tau
            off_diagonal = ~torch.eye(len(user), dtype=torch.bool)
            return (torch.logsumexp(scores[off_diagonal].reshape(len(user), -1), dim=1) - scores.diagonal()).mean()
        visual_ssm = expected_ssm(u[users], visual[pos])
        textual_ssm = expected_ssm(u[users], textual[pos])
        # Verify [B,K] edge weights and raw item-ID anchors, not propagated items.
        raw = model.item_id_embedding.weight
        ii = bpr.new_zeros(())
        for alpha, features, ids in ((model.ii_image_weight, model.v_feat, model.image_knn_idx),
                                     (1-model.ii_image_weight, model.t_feat, model.text_knn_idx)):
            normalized = F.normalize(features, dim=-1)
            weights = (normalized[pos, None] * normalized[ids[pos]]).sum(-1)
            scores = (raw[pos, None] * raw[ids[pos]]).sum(-1)
            ii = ii + alpha * (weights * F.softplus(-scores)).sum()
        expected = bpr + model.ssm_weight * (model.alpha * textual_ssm + (1-model.alpha) * visual_ssm) + model.ii_weight * ii + model.local_weight * model.local_loss()
        # No hidden SSM random sampler; BPR uses the supplied batch negatives.
        with mock.patch('models.freedomlocalssm.torch.randint', side_effect=AssertionError('SSM sampled negatives')):
            actual = model.calculate_loss(self.batch)
        torch.testing.assert_close(actual, expected)
        for key, value in (('bpr', bpr), ('visual_ssm', visual_ssm), ('textual_ssm', textual_ssm), ('ii', ii)):
            self.assertAlmostEqual(model.last_loss_terms[key], value.detach().item(), places=4)
        actual.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)

    def test_constraint_targets_raw_item_ids_and_disabled_aux_is_pure_bpr(self):
        model = FreedomLocalSSM(self.config, self.loader)
        ii = model.item_alignment(self.batch[1], model.image_knn_idx,
                                  model.image_knn_weight, model.item_id_embedding.weight)
        ii.backward()
        self.assertGreater(float(model.item_id_embedding.weight.grad.abs().sum()), 0.)
        for name, parameter in model.named_parameters():
            if name != 'item_id_embedding.weight':
                self.assertIsNone(parameter.grad, name)
        model.zero_grad(set_to_none=True)
        model.ssm_weight = model.ii_weight = model.local_weight = 0.
        users, pos, neg = self.batch
        u, i = model.forward()
        expected = F.softplus((u[users] * (i[neg] - i[pos])).sum(-1)).mean()
        actual = model.calculate_loss(self.batch)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNone(model.image_trs.weight.grad)
        self.assertIsNone(model.text_embedding.weight.grad)
        model.eval()
        model.full_sort_predict([users])
        self.assertIsNotNone(model._eval_embeddings)
        model.train()
        self.assertIsNone(model._eval_embeddings)
        model.dropout = .5
        model.pre_epoch_processing()
        self.assertEqual(model.masked_adj._nnz(), int(model.edge_values.numel() * .5) * 2)

    def test_ssm_off_diagonal_duplicates_singleton_and_alpha_endpoints(self):
        model = FreedomLocalSSM(self.config, self.loader)
        user = torch.randn(3, 6, requires_grad=True)
        item = torch.randn(3, 6, requires_grad=True)
        with torch.no_grad():
            item[2].copy_(item[0])  # Duplicates are not specially masked.
        scores = F.normalize(user, dim=-1) @ F.normalize(item, dim=-1).t() / model.tau
        expected = torch.stack([torch.logsumexp(torch.cat((scores[r, :r], scores[r, r+1:])), 0)
                                - scores[r, r] for r in range(3)]).mean()
        loss = model.ssm_loss(user, item)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(user.grad).all())
        self.assertTrue(torch.isfinite(item.grad).all())
        singleton = model.ssm_loss(user[:1], item[:1])
        self.assertEqual(singleton.item(), 0.)
        singleton.backward()
        # Alpha weights TEXT SSM only; ii keeps its independent image mix.
        for alpha in (0., 1.):
            model.alpha = alpha
            actual = model.calculate_loss(self.batch)
            terms = model.last_loss_terms
            selected = terms['textual_ssm'] if alpha else terms['visual_ssm']
            expected = terms['bpr'] + model.ssm_weight * selected + model.ii_weight * terms['ii'] + model.local_weight * terms['local']
            self.assertAlmostEqual(actual.item(), expected, places=5)
        self.assertFalse(hasattr(model, 'num_negatives'))

    def test_three_job_queue_no_checkpoints(self):
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
        run = subprocess.run([sys.executable, str(ROOT / 'experiments/run_night.py'),
                              '--suite', 'freedom_local_ssm', '--data-path', str(data.parent),
                              '--output', str(output), '--cpu', '--epochs', '1', '--hours', '.1'],
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual([row['state'] for row in summary], ['complete'] * 3, summary)
        self.assertFalse(list(output.rglob('*.pth')))
        jobs = json.loads((output / 'manifest.json').read_text())['jobs']
        self.assertEqual([j['overrides']['local_weight'] for j in jobs], [.1, .01, .001])
        self.assertTrue(all(j['overrides']['seed'] == 999 and j['model'] == 'FreedomLocalSSM' for j in jobs))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_construction_and_loss(self):
        device = torch.device('cuda:0')
        model = FreedomLocalSSM(dict(self.config, device=device), self.loader).to(device)
        model.pre_epoch_processing()
        model.calculate_loss(self.batch.to(device)).backward()
        for name, buffer in model.named_buffers():
            self.assertEqual(buffer.device, device, name)
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == '__main__':
    unittest.main()
