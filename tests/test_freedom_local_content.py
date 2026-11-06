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
from models.freedomlocalcontent import FreedomLocalContent

ROOT = Path(__file__).resolve().parents[1]


class FreedomLocalContentTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        self.config.update(yaml.safe_load((ROOT / 'src/configs/model/FreedomLocalContent.yaml').read_text()))
        self.config.update(embedding_size=6, feat_embed_dim=6, knn_k=4, dropout=0.)

    def test_fusion_and_ui_polynomial_no_item_id_or_ii_propagation(self):
        model = FreedomLocalContent(self.config, self.loader)
        self.assertFalse(hasattr(model, 'item_id_embedding'))
        self.assertFalse(hasattr(model, 'mm_adj'))
        self.assertEqual(set(dict(model.named_parameters())), {
            'user_embedding.weight', 'image_embedding.weight', 'text_embedding.weight',
            'image_trs.weight', 'image_trs.bias', 'text_trs.weight', 'text_trs.bias'})
        for alpha in (0., .1, 1.):
            model.alpha = alpha
            q = alpha * model.image_trs(model.image_embedding.weight) + (1-alpha) * model.text_trs(model.text_embedding.weight)
            torch.testing.assert_close(model.item_representations(), q)
            x = torch.cat((model.user_embedding.weight, q))
            ax = torch.sparse.mm(model.norm_adj, x)
            expected = (x + ax + torch.sparse.mm(model.norm_adj, ax)) / 3
            actual = torch.cat(model.forward())
            torch.testing.assert_close(actual, expected)
        model.n_ui_layers = 0
        u, i = model.forward()
        torch.testing.assert_close(u, model.user_embedding.weight)
        torch.testing.assert_close(i, model.item_representations())

    def test_ssm_and_alignment_match_requested_formulas(self):
        model = FreedomLocalContent(self.config, self.loader)
        u, p, n = torch.randn(3, 6), torch.randn(3, 6), torch.randn(3, 32, 6)
        logits = (F.normalize(u, dim=-1)[:, None] *
                  F.normalize(torch.cat((p[:, None], n), dim=1), dim=-1)).sum(-1) / .04
        expected = (-torch.log(logits[:, 0].exp() / logits[:, 1:].exp().sum(1))).mean()
        torch.testing.assert_close(model.ssm_loss(u, p, n, .04), expected)
        # Even extreme temperatures stay finite through logsumexp.
        self.assertTrue(torch.isfinite(model.ssm_loss(u, p, n, 1e-6)))
        content = model.item_representations()
        anchors = torch.tensor([0, 2, 0])  # Repeated anchors intentionally retained.
        dense = F.normalize(model.v_feat, dim=-1) @ F.normalize(model.v_feat, dim=-1).T
        neighbors = model.image_knn_idx[anchors]
        expected_weights = dense[anchors[:, None], neighbors]
        torch.testing.assert_close(model.image_knn_weight[anchors], expected_weights)
        score = (content[anchors, None] * content[neighbors]).sum(-1)
        expected = -(expected_weights * score.sigmoid().log()).sum()
        torch.testing.assert_close(model.item_alignment(anchors, model.image_knn_idx,
                                                       model.image_knn_weight, content), expected)
        large = content.detach().clone() * 1e6
        self.assertTrue(torch.isfinite(model.item_alignment(anchors, model.image_knn_idx,
                                                           model.image_knn_weight, large)))

    def test_total_gradient_pruning_and_prediction_cache(self):
        model = FreedomLocalContent(self.config, self.loader)
        batch = torch.tensor([[0, 1, 2], [0, 2, 4], [10, 11, 12]])
        model.pre_epoch_processing()
        # Reconstruct the exact uniform negatives without changing the model RNG.
        state = torch.get_rng_state()
        neg = torch.randint(model.n_items, (3, model.num_negatives))
        torch.set_rng_state(state)
        q = model.item_representations()
        u, i = model.forward(model.masked_adj, q)
        ssm = model.ssm_loss(u[batch[0]], i[batch[1]], i[neg], model.temperature)
        ii = sum(weight * model.item_alignment(batch[1], ids, weights, q) for weight, ids, weights in (
            (model.alpha, model.image_knn_idx, model.image_knn_weight),
            (1-model.alpha, model.text_knn_idx, model.text_knn_weight)))
        loss = model.calculate_loss(batch)
        torch.testing.assert_close(loss, ssm + model.ii_weight * ii + model.local_weight * model.local_loss())
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
        model.dropout = .5
        model.pre_epoch_processing()
        self.assertEqual(model.masked_adj._nnz(), 2 * int(model.edge_values.numel() * .5))
        model.eval()
        scores = model.full_sort_predict([batch[0]])
        torch.testing.assert_close(scores, u[batch[0]] @ i.T)
        cached = model._eval_embeddings
        model.full_sort_predict(batch[0])
        self.assertIs(model._eval_embeddings, cached)
        model.train()
        self.assertIsNone(model._eval_embeddings)

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
                              '--suite', 'freedom_local_content', '--data-path', str(data.parent),
                              '--output', str(output), '--cpu', '--epochs', '1', '--hours', '.1'],
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual([row['state'] for row in summary], ['complete'] * 3, summary)
        self.assertFalse(list(output.rglob('*.pth')))
        jobs = json.loads((output / 'manifest.json').read_text())['jobs']
        self.assertEqual([j['overrides']['local_weight'] for j in jobs], [.01, .1, 1.])
        self.assertTrue(all(j['overrides']['seed'] == 999 for j in jobs))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_construction_before_outer_to_and_gradient(self):
        device = torch.device('cuda:0')
        model = FreedomLocalContent(dict(self.config, device=device), self.loader).to(device)
        model.pre_epoch_processing()
        batch = torch.tensor([[0, 1, 2], [0, 2, 4], [10, 11, 12]], device=device)
        model.calculate_loss(batch).backward()
        for name, buffer in model.named_buffers():
            self.assertEqual(buffer.device, device, name)
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == '__main__':
    unittest.main()
