import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch
import yaml

import test_simmrec as fixtures
from common.spectral_coordinates import build_spectral_coordinates
from models.freedom import FREEDOM
from models.freedomspectralcode import FreedomSpectralCode

ROOT = Path(__file__).resolve().parents[1]


class CoordinatesTest(unittest.TestCase):
    def test_normalized_svd_geometry_and_permutations_and_cold_nodes(self):
        rng = np.random.default_rng(19)
        r = sp.csr_matrix(rng.random((14, 11)) < .4, dtype=np.float32)
        r = sp.vstack((r, sp.csr_matrix((1, 11))), format='csr')
        r = sp.hstack((r, sp.csr_matrix((15, 1))), format='csr')
        u, i, singular = build_spectral_coordinates(r, rank=4)
        self.assertTrue(np.all(u[-1] == 0)); self.assertTrue(np.all(i[-1] == 0))
        du = np.asarray(r.sum(1)).ravel(); di = np.asarray(r.sum(0)).ravel()
        x = r.toarray() / np.sqrt(np.maximum(du, 1))[:, None] / np.sqrt(np.maximum(di, 1))[None, :]
        eu, s, vt = np.linalg.svd(x, full_matrices=False)
        np.testing.assert_allclose(singular, s[:4], atol=2e-6)
        expected = np.vstack((eu[:, :4] * s[:4], vt[:4].T * s[:4]))
        expected /= np.sqrt(np.square(expected).sum() / (np.count_nonzero(du) + np.count_nonzero(di)) / 4)
        codes = np.vstack((u, i))
        np.testing.assert_allclose(codes @ codes.T, expected @ expected.T, atol=4e-5)
        pu = rng.permutation(15); pi = rng.permutation(12)
        ru, ri, _ = build_spectral_coordinates(r[pu][:, pi], rank=4)
        remapped = np.vstack((ru[np.argsort(pu)], ri[np.argsort(pi)]))
        np.testing.assert_allclose(codes @ codes.T, remapped @ remapped.T, atol=4e-5)
        self.assertEqual(build_spectral_coordinates(r[:2, :3], rank=64)[0].shape[1], 1)
        with self.assertRaises(ValueError):
            build_spectral_coordinates(sp.csr_matrix((3, 4)))


class SpectralCodeModelTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        self.config.update(yaml.safe_load((ROOT / 'src/configs/model/FreedomSpectralCode.yaml').read_text()))
        self.config.update(embedding_size=6, feat_embed_dim=6, knn_k=4,
                           spectral_rank=2, dropout=0., cf_model='lightgcn')

    def test_zero_switch_matches_freedom_and_shared_projection_gradients(self):
        batch = torch.tensor([[0, 1, 2], [0, 2, 4], [10, 11, 12]])
        config = dict(self.config, n_ui_layers=2)
        torch.manual_seed(5); base = FREEDOM(config, self.loader)
        torch.manual_seed(5); control = FreedomSpectralCode(dict(config, graph_strength=0.), self.loader)
        base.pre_epoch_processing(); control.pre_epoch_processing()
        for a, b in zip(base.calculate_loss(batch), control.calculate_loss(batch)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for sides, layers in [('user', 0), ('both', 0), ('both', 2)]:
            with self.subTest(sides=sides, layers=layers):
                model = FreedomSpectralCode(dict(config, graph_sides=sides, n_ui_layers=layers), self.loader)
                model.pre_epoch_processing()
                user, item = model.structural_inputs()
                torch.testing.assert_close(user, model.user_embedding.weight + model.code_scale * model.code_projector(model.user_codes))
                if sides == 'user':
                    torch.testing.assert_close(item, model.item_id_embedding.weight)
                else:
                    torch.testing.assert_close(item, model.item_id_embedding.weight + model.code_scale * model.code_projector(model.item_codes))
                with patch('torch.sparse.mm', wraps=torch.sparse.mm) as multiply:
                    u, i = model.forward(model.masked_adj)
                self.assertEqual(multiply.call_count, model.n_layers + layers)
                torch.testing.assert_close(model.full_sort_predict([batch[0]]), u[batch[0]] @ i.T)
                model.calculate_loss(batch)[0].backward()
                self.assertGreater(float(model.code_projector.weight.grad.abs().sum()), 0.)
                self.assertTrue(torch.isfinite(model.code_projector.weight.grad).all())
                self.assertIsNone(model.user_codes.grad); self.assertIsNone(model.item_codes.grad)
                self.assertEqual(sum(p.numel() for p in model.parameters()),
                                 sum(p.numel() for p in base.parameters()) + 2 * 6)
                # All observed user rows and item rows: common initial RMS calibration.
                projected = model.code_projector(torch.cat((model.user_codes, model.item_codes[:6])))
                ids = torch.cat((model.user_embedding.weight, model.item_id_embedding.weight[:6]))
                torch.testing.assert_close((projected * model.code_scale).square().mean(), ids.square().mean())

    def test_model_predictions_equivariant_when_entity_rows_are_remapped(self):
        model = FreedomSpectralCode(self.config, self.loader)
        original = model.full_sort_predict([torch.arange(3)])
        other = copy.deepcopy(model)
        pu, pi = torch.tensor([2, 0, 1]), torch.randperm(24)
        with torch.no_grad():
            other.user_embedding.weight.copy_(model.user_embedding.weight[pu])
            other.item_id_embedding.weight.copy_(model.item_id_embedding.weight[pi])
            other.user_codes = model.user_codes[pu]
            other.item_codes = model.item_codes[pi]
            other.mm_adj = model.mm_adj.to_dense()[pi][:, pi].to_sparse()
        torch.testing.assert_close(other.full_sort_predict([torch.arange(3)]), original[pu][:, pi])

    def test_three_job_cpu_queue_without_checkpoints_or_timestamps(self):
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
        command = [sys.executable, str(ROOT / 'experiments/run_night.py'), '--suite', 'freedom_spectral_code',
                   '--data-path', str(data.parent), '--output', str(output), '--cpu', '--epochs', '1', '--hours', '.1']
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual([row['state'] for row in summary], ['complete'] * 3, summary)
        self.assertTrue(all('valid_recall@20' in row for row in summary))
        self.assertFalse(list(output.rglob('*.pth')))
        jobs = json.loads((output / 'manifest.json').read_text())['jobs']
        self.assertEqual([(j['overrides']['graph_sides'], j['overrides']['n_ui_layers']) for j in jobs],
                         [('user', 0), ('both', 0), ('both', 2)])


if __name__ == '__main__':
    unittest.main()
