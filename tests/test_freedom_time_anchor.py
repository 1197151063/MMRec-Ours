import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch
import yaml

import test_simmrec as fixtures
from common.time_anchor_groups import train_item_times, build_time_anchor_groups
from models.freedom import FREEDOM
from models.freedomtimeanchor import FreedomTimeAnchor

ROOT = Path(__file__).resolve().parents[1]


class TimeAnchorRuleTest(unittest.TestCase):
    def test_global_item_time_anchor_and_tie_relabeling(self):
        r = sp.csr_matrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        times = np.array([100., 200., 50., 300.])
        labels, anchor = build_time_anchor_groups(r, times)
        np.testing.assert_array_equal(anchor, [0, 2, 2, -1])
        self.assertEqual(labels[1], labels[2]); self.assertNotEqual(labels[0], labels[1])
        self.assertEqual(labels[-1], -1)
        tied_times = np.ones(4); keys = np.array([.4, .2, .1, .3])
        _, anchor = build_time_anchor_groups(r, tied_times, ties=keys)
        perm = np.array([2, 0, 3, 1])
        _, other = build_time_anchor_groups(r[:, perm], tied_times[perm], ties=keys[perm])
        np.testing.assert_array_equal(anchor[:3], perm[other[:3]])

    def test_heldout_and_removed_train_edges_cannot_change_item_times(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tiny.inter'
            path.write_text('userID\titemID\ttimestamp\tx_label\n'
                            '0\t0\t100\t0\n0\t1\t200\t0\n'
                            '1\t0\t120\t0\n1\t1\t210\t0\n'
                            '0\t1\t1\t1\n1\t1\t-10\t2\n')
            r = sp.csr_matrix([[1, 1], [1, 1]])
            np.testing.assert_array_equal(train_item_times(path, r), [100, 200])
            subset = sp.csr_matrix([[0, 1], [1, 1]])
            np.testing.assert_array_equal(train_item_times(path, subset), [120, 200])


class TimeAnchorModelTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        self.config.update(yaml.safe_load((ROOT / 'src/configs/model/FreedomTimeAnchor.yaml').read_text()))
        self.config.update(embedding_size=6, feat_embed_dim=6, knn_k=4, dropout=0., cf_model='lightgcn',
                           inter_file_name='tiny.inter', inter_splitting_label='x_label', field_separator='\t')
        self.loader.inter_matrix = lambda form: sp.coo_matrix(
            (np.ones(6), ([0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 2, 3])), shape=(3, 24))
        p = Path(self.folder.name) / 'tiny/tiny.inter'
        p.write_text('userID\titemID\ttimestamp\tx_label\n'
                     '0\t0\t100\t0\n0\t1\t200\t0\n1\t0\t110\t0\n'
                     '1\t2\t50\t0\n2\t2\t60\t0\n2\t3\t400\t0\n'
                     '0\t1\t-1\t1\n')

    def test_group_gradients_personal_penalty_and_prediction(self):
        batch = torch.tensor([[0, 1, 2], [0, 2, 3], [10, 11, 12]])
        torch.manual_seed(5); base = FREEDOM(self.config, self.loader)
        torch.manual_seed(5); control = FreedomTimeAnchor(dict(self.config, group_strength=0.), self.loader)
        base.pre_epoch_processing(); control.pre_epoch_processing()
        for a, b in zip(base.calculate_loss(batch), control.calculate_loss(batch)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        model = FreedomTimeAnchor(dict(self.config, personal_reg_weight=.1), self.loader)
        np.testing.assert_array_equal(model.user_anchor_items.cpu().numpy(), [0, 2, 2])
        self.assertEqual(model.group_embedding.num_embeddings, 2)
        model.pre_epoch_processing()
        original = FREEDOM.calculate_loss(model, batch)[0]
        loss = model.calculate_loss(batch)[0]
        torch.testing.assert_close(loss - original, .1 * model.user_embedding.weight.square().sum(-1).mean())
        loss.backward()
        self.assertGreater(float(model.group_embedding.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(model.group_embedding.weight.grad).all())
        u, i = model.forward(model.norm_adj)
        torch.testing.assert_close(model.full_sort_predict([batch[0]]), u[batch[0]] @ i.T)

    def test_one_epoch_queue_without_checkpoints(self):
        workspace = Path(self.folder.name)
        data = workspace / 'data/baby'; data.mkdir(parents=True)
        for name in ('image_feat.npy', 'text_feat.npy'):
            shutil.copy(workspace / 'tiny' / name, data / name)
        rows = ['userID\titemID\ttimestamp\tx_label']
        for u in range(3):
            rows.extend(f'{u}\t{i}\t{100+i}\t0' for i in {0, *range(u * 7 + 1, u * 7 + 8)})
            rows.extend([f'{u}\t22\t-2\t1', f'{u}\t23\t-3\t2'])
        (data / 'baby.inter').write_text('\n'.join(rows) + '\n')
        output = workspace / 'runs'
        command = [sys.executable, str(ROOT / 'experiments/run_night.py'), '--suite', 'freedom_time_anchor_no_ui',
                   '--data-path', str(data.parent), '--output', str(output), '--cpu', '--epochs', '1',
                   '--hours', '.1', '--limit', '1']
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(json.loads((output / 'summary.json').read_text())[0]['state'], 'complete')
        self.assertFalse(list(output.rglob('*.pth')))
        jobs = json.loads((output / 'manifest.json').read_text())['jobs']
        self.assertEqual(len(jobs), 1)  # --limit selects only the first no-UI job.
        self.assertEqual(jobs[0]['overrides']['n_ui_layers'], 0)
        sys.path.insert(0, str(ROOT / 'experiments'))
        from freedom_time_anchor_plan import build_no_ui_plan
        plan = build_no_ui_plan()
        self.assertEqual(len(plan), 3)
        self.assertEqual([j['overrides']['group_strength'] for j in plan], [.1, .3, 1.])
        self.assertTrue(all(j['overrides']['n_ui_layers'] == 0 and
                            j['overrides']['personal_reg_weight'] == 0. and
                            j['overrides']['seed'] == 999 for j in plan))

    def test_no_ui_uses_direct_user_sum_and_only_item_propagation(self):
        model = FreedomTimeAnchor(dict(self.config, n_ui_layers=0, dropout=.8,
                                       group_strength=1.), self.loader)
        batch = torch.tensor([[0, 1, 2], [0, 2, 3], [10, 11, 12]])
        with patch('torch.multinomial', side_effect=AssertionError('Unused UI edge sampling')):
            model.pre_epoch_processing()
        self.assertIsNone(model.masked_adj)
        with patch('torch.sparse.mm', wraps=torch.sparse.mm) as multiply:
            users, items = model.forward(None)
        self.assertEqual(multiply.call_count, model.n_layers)
        for call in multiply.call_args_list:
            self.assertIs(call.args[0], model.mm_adj)
        expected = model.user_embedding.weight + model.group_embedding(model.anchor_group_ids)
        torch.testing.assert_close(users, expected, rtol=0, atol=0)
        h = torch.sparse.mm(model.mm_adj, model.item_id_embedding.weight)
        torch.testing.assert_close(items, model.item_id_embedding.weight + h)
        torch.testing.assert_close(model.full_sort_predict([batch[0]]), users[batch[0]] @ items.T)
        loss = model.calculate_loss(batch)[0]
        loss.backward()
        for parameter in (model.user_embedding.weight, model.group_embedding.weight,
                          model.item_id_embedding.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
