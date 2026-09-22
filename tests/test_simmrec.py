"""Run from the repository root: python -m unittest discover -s tests -v."""
import copy
import inspect
import sys
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from models.simmrec import SIMMRec
from models.lightmrecnope import LightMRecNoPE


class SIMMRecTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        path = Path(self.folder.name) / 'tiny'
        path.mkdir()
        np.save(path / 'image_feat.npy', np.random.default_rng(7).normal(size=(24, 8)))
        np.save(path / 'text_feat.npy', np.random.default_rng(8).normal(size=(24, 10)))
        self.config = dict(USER_ID_FIELD='userID', ITEM_ID_FIELD='itemID', NEG_PREFIX='neg_',
                           train_batch_size=3, device=torch.device('cpu'), end2end=False,
                           is_multimodal_model=True, data_path=self.folder.name + '/', dataset='tiny',
                           vision_feature_file='image_feat.npy', text_feature_file='text_feat.npy',
                           embedding_size=6, alpha=0.5, item_id_weight=0.1, temperature=0.1,
                           num_negatives=32, reg_weight=1e-4)
        self.rows = np.array([0, 0, 1, 1, 2, 2])
        self.cols = np.array([0, 1, 2, 3, 4, 5])
        self.loader = SimpleNamespace(
            dataset=SimpleNamespace(get_user_num=lambda: 3, get_item_num=lambda: 24),
            inter_matrix=lambda form: sp.coo_matrix((np.ones(6), (self.rows, self.cols)), shape=(3, 24)))
        self.model = SIMMRec(self.config, self.loader)

    def test_negatives_exclude_all_training_positives(self):
        users = torch.tensor([0, 1, 2])
        for _ in range(5):
            negatives = self.model.sample_negatives(users)
            for u in users.tolist():
                self.assertFalse(set(negatives[u].tolist()) & set(self.cols[self.rows == u]))

    def test_frozen_content_and_trainable_residual(self):
        loss = self.model.calculate_loss(torch.tensor([[0, 1, 2], [0, 2, 4]]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(self.model.image_embedding.weight.grad)
        self.assertIsNone(self.model.text_embedding.weight.grad)
        for parameter in (self.model.user_embedding.weight, self.model.item_embedding.weight,
                          self.model.image_trs.weight, self.model.text_trs.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_content_only_and_prediction_contract(self):
        model = SIMMRec(dict(self.config, item_id_weight=0), self.loader).eval()
        self.assertIsNone(model.item_embedding)
        users, items = model.forward()
        scores = model.full_sort_predict([torch.arange(3)])
        torch.testing.assert_close(scores, users @ items.T)
        torch.testing.assert_close(scores, model.full_sort_predict(torch.arange(3)))
        self.assertEqual(scores.shape, (3, 24))
        model.train()
        self.assertIsNone(model._eval_embeddings)

    def test_consistent_id_remapping_preserves_scores(self):
        # Remap every entity-specific row, including its initialized residual.
        model = self.model.eval()
        original = model.full_sort_predict(torch.arange(3))
        remapped = copy.deepcopy(model)
        users, items = torch.randperm(3), torch.randperm(24)
        with torch.no_grad():
            remapped.user_embedding.weight.copy_(model.user_embedding.weight[users])
            for name in ('image_embedding', 'text_embedding', 'item_embedding'):
                getattr(remapped, name).weight.copy_(getattr(model, name).weight[items])
        remapped.eval()
        actual = remapped.full_sort_predict(torch.arange(3))
        torch.testing.assert_close(actual, original[users][:, items])


    def test_id_only_needs_no_features_and_trains(self):
        config = dict(self.config, representation='id', is_multimodal_model=False)
        model = SIMMRec(config, self.loader)
        self.assertFalse(hasattr(model, 'image_embedding'))
        loss = model.calculate_loss(torch.tensor([[0, 1, 2], [0, 2, 4]]))
        loss.backward()
        self.assertGreater(model.item_embedding.weight.grad.abs().sum().item(), 0)
        model.eval()
        expected = torch.nn.functional.normalize(model.user_embedding.weight, dim=-1) @ torch.nn.functional.normalize(model.item_embedding.weight, dim=-1).T
        torch.testing.assert_close(model.full_sort_predict(torch.arange(3)), expected)

    def test_original_without_pe_keeps_loss_and_raw_scoring(self):
        model = LightMRecNoPE(dict(self.config, temperature=0.04, num_negatives=32), self.loader).eval()
        batch = torch.tensor([[0, 1, 2], [0, 2, 4]])
        torch.manual_seed(42)
        actual = model.calculate_loss(batch)
        torch.manual_seed(42)
        users, items = model.forward()
        negatives = torch.randint(24, (3, 32))
        candidates = torch.cat((batch[1, :, None], negatives), dim=1)
        scores = (torch.nn.functional.normalize(items[candidates], dim=-1) *
                  torch.nn.functional.normalize(users, dim=-1)[:, None]).sum(-1) / 0.04
        expected = -torch.log(scores[:, 0].exp() / scores[:, 1:].exp().sum(-1)).mean()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(model.full_sort_predict(torch.arange(3)), users @ items.T)
        model.train()
        model.calculate_loss(batch).backward()
        self.assertIsNotNone(model.image_embedding.weight.grad)
        self.assertIsNotNone(model.text_embedding.weight.grad)
        self.assertFalse(any('pe' in name for name, _ in model.named_buffers()))

    def test_server_entrypoint_runs_both_ablation_configs_and_saves(self):
        root = Path(__file__).resolve().parents[1]
        workspace = Path(self.folder.name)
        shutil.copytree(root / 'src/configs', workspace / 'configs')
        data = workspace / 'data' / 'baby'
        data.mkdir(parents=True)
        for name in ('image_feat.npy', 'text_feat.npy'):
            shutil.copy(workspace / 'tiny' / name, data / name)
        rows = ['userID\titemID\tx_label']
        for user in range(3):
            rows.extend(f'{user}\t{item}\t0' for item in range(user * 8, (user + 1) * 8))
            rows.append(f'{user}\t{((user + 1) * 8) % 24}\t1')
            rows.append(f'{user}\t{((user + 1) * 8 + 1) % 24}\t2')
        (data / 'baby.inter').write_text('\n'.join(rows) + '\n')
        (workspace / 'overrides.yaml').write_text(
            'use_gpu: false\nitem_id_weight: [0.0, 0.1]\nnum_negatives: [4]\n'
            'train_batch_size: 8\neval_batch_size: 2\n')
        result = subprocess.run(
            [sys.executable, str(root / 'src/main.py'), '--model', 'SIMMRec',
             '--dataset', 'baby', '--data-path', str(workspace / 'data'),
             '--config', str(workspace / 'overrides.yaml'), '--epochs', '2', '--save-model'],
            cwd=workspace, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        checkpoints = list((workspace / 'saved').glob('*.pth'))
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(len(list((workspace / 'csv').glob('*.csv'))), 2)
        for checkpoint in checkpoints:
            options = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
            state = torch.load(checkpoint, map_location='cpu', **options)
            self.assertIn('recall@20', state['valid_result'])
            self.assertIn('image_trs.weight', state['state_dict'])
        # Exercise the exact server diagnostic configurations, with short CPU runs.
        for model_name, diagnostic, expected_runs in [
            ('LightMRecNoPE', None, 1),
            ('LightMRecOrder', 'order-audit-none.yaml', 1),
            ('LightMRecOrder', None, 12),
            ('SIMMRec', 'simmrec-id-diagnostic.yaml', 3),
            ('SIMMRec', 'simmrec-content-diagnostic.yaml', 9),
        ]:
            before = len(list((workspace / 'saved').glob('*.pth')))
            csv_before = len(list((workspace / 'csv').glob('*.csv')))
            import yaml
            overrides = yaml.safe_load((root / 'src/configs' / diagnostic).read_text()) if diagnostic else {}
            overrides.update(use_gpu=False, train_batch_size=8, eval_batch_size=2)
            (workspace / 'diagnostic.yaml').write_text(yaml.safe_dump(overrides))
            result = subprocess.run(
                [sys.executable, str(root / 'src/main.py'), '--model', model_name,
                 '--dataset', 'baby', '--data-path', str(workspace / 'data'),
                 '--config', str(workspace / 'diagnostic.yaml'), '--epochs', '2'],
                cwd=workspace, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(list((workspace / 'saved').glob('*.pth'))), before)
            self.assertEqual(len(list((workspace / 'csv').glob('*.csv'))) - csv_before, expected_runs)


        # Night queue: real subprocesses, structured summaries and no-save resume.
        night = workspace / 'night'
        command = [sys.executable, str(root / 'experiments/run_night.py'),
                   '--data-path', str(workspace / 'data'), '--dataset', 'baby', '--cpu',
                   '--epochs', '1', '--limit', '2', '--hours', '0.1', '--output', str(night)]
        import json
        for repeat in range(2):
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((night / 'summary.json').read_text())
            self.assertEqual([row['state'] for row in summary], ['complete', 'complete'])
            self.assertTrue(all('valid_recall@20' in row for row in summary))
            if repeat == 0:
                first_status = [(night / row['name'] / 'status.json').stat().st_mtime_ns for row in summary]
            else:
                self.assertEqual(first_status, [(night / row['name'] / 'status.json').stat().st_mtime_ns for row in summary])
        self.assertFalse(list(night.rglob('*.pth')))
        init_night = workspace / 'init-night'
        command = [sys.executable, str(root / 'experiments/run_night.py'),
                   '--suite', 'init', '--data-path', str(workspace / 'data'), '--cpu',
                   '--epochs', '1', '--hours', '0.1', '--output', str(init_night)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((init_night / 'summary.json').read_text())
        self.assertEqual(len(summary), 11)
        self.assertTrue(all(row['state'] == 'complete' for row in summary), summary)
        self.assertFalse(list(init_night.rglob('*.pth')))
        profile_night = workspace / 'profile-night'
        command = [sys.executable, str(root / 'experiments/run_night.py'),
                   '--suite', 'profile', '--data-path', str(workspace / 'data'), '--cpu',
                   '--epochs', '1', '--limit', '12', '--hours', '0.1', '--output', str(profile_night)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((profile_night / 'summary.json').read_text())
        self.assertEqual(len(summary), 12)
        self.assertTrue(all(row['state'] == 'complete' for row in summary), summary)
        self.assertFalse(list(profile_night.rglob('*.pth')))
        corr_night = workspace / 'corr-night'
        command = [sys.executable, str(root / 'experiments/run_night.py'),
                   '--suite', 'corr', '--data-path', str(workspace / 'data'), '--cpu',
                   '--epochs', '1', '--limit', '36', '--hours', '0.1', '--output', str(corr_night)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((corr_night / 'summary.json').read_text())
        self.assertEqual(len(summary), 36)
        self.assertTrue(all(row['state'] == 'complete' for row in summary), summary)
        self.assertFalse(list(corr_night.rglob('*.pth')))


        sys.path.insert(0, str(root / 'experiments'))
        from diagnose_user_neighbors import diagnose
        initial = diagnose(workspace / 'data', 'baby', pairs=100)
        self.assertEqual(initial['results']['adjacent']['mean_jaccard'], 0.)
        # Held-out rows cannot affect the training-only diagnostic.
        (data / 'baby.inter').write_text((data / 'baby.inter').read_text() + '999\t999\t2\n')
        self.assertEqual(initial, diagnose(workspace / 'data', 'baby', pairs=100))




if __name__ == '__main__':
    unittest.main()
