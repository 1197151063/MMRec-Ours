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
from common.user_cohort_graph import build_cohort_graph, project_content
from models.freedomalign import FreedomAlign
from models.freedomaligncohort import FreedomAlignCohort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from research_user_cohorts import build_neighbor_rows


class CohortGraphTest(unittest.TestCase):
    def test_matches_diagnostic_graphs_and_relabeling(self):
        rng = np.random.default_rng(6)
        r = sp.csr_matrix((rng.random((15, 20)) < .3).astype(np.float32))
        visual, text = (rng.normal(size=(20, d)).astype(np.float32) for d in (8, 10))
        ties = rng.random(15)
        content = dict(visual=project_content(visual, 6, 2026), text=project_content(text, 6, 2027))
        reference, _ = build_neighbor_rows(r, content, rng.normal(size=(15, 6)), np.arange(15), [4], ties, 6)
        permutation = rng.permutation(15); inverse = np.argsort(permutation)
        for kind in ('shared_cosine', 'shared_random', 'residual_semantic'):
            graph = build_cohort_graph(r, visual, text, kind, 4, 6, ties=ties)
            np.testing.assert_allclose(graph.toarray(), reference[(kind, 4)].toarray())
            permuted = build_cohort_graph(r[permutation], visual, text, kind, 4, 6, ties=ties[permutation])
            np.testing.assert_allclose(graph.toarray(), permuted[inverse][:, inverse].toarray())
            self.assertFalse(graph.diagonal().any())
            self.assertTrue((np.diff(graph.indptr) <= 4).all())
            mass = np.asarray(graph.sum(1)).ravel()
            np.testing.assert_allclose(mass[mass > 0], 1.)

    def test_all_common_items_isolates_and_rng(self):
        r = sp.csr_matrix([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
        feat = np.ones((3, 4), dtype=np.float32)
        np_before = np.random.get_state(); torch_before = torch.get_rng_state().clone()
        g = build_cohort_graph(r, feat, feat, 'residual_semantic', 80, 6)
        self.assertEqual(g.nnz, 0)
        g = build_cohort_graph(r, None, None, 'shared_cosine', 80, 6)
        self.assertEqual(g.nnz, 2)
        np.testing.assert_array_equal(np.random.get_state()[1], np_before[1])
        self.assertEqual(np.random.get_state()[2:], np_before[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))


class CohortModelTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        options = yaml.safe_load((ROOT / 'src/configs/model/FreedomAlignCohort.yaml').read_text())
        options.pop('hyper_parameters'); self.config.update(options)
        self.loader.inter_matrix = lambda form: sp.csr_matrix(
            (np.ones(9), ([0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 3, 4, 0, 5, 6])), shape=(3, 24))

    def test_baseline_forward_loss_gradient_and_cache(self):
        batch = torch.tensor([[0, 1, 2], [1, 3, 5], [9, 10, 11]])
        torch.manual_seed(7); base = FreedomAlign(self.config, self.loader)
        for kind, beta in [('none', .1), ('residual_semantic', 0.)]:
            torch.manual_seed(7)
            model = FreedomAlignCohort(dict(self.config, cohort_kind=kind, cohort_beta=beta), self.loader)
            self.assertIsNone(model.cohort_adj)
            torch.testing.assert_close(model.calculate_loss(batch), base.calculate_loss(batch), rtol=0, atol=0)
        for kind in ('shared_cosine', 'shared_random', 'residual_semantic'):
            torch.manual_seed(7)
            model = FreedomAlignCohort(dict(self.config, cohort_kind=kind), self.loader)
            self.assertEqual(len(list(base.parameters())), len(list(model.parameters())))
            for a, b in zip(base.parameters(), model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            raw_u, raw_i = FreedomAlign.forward(model)
            u, i = model.forward()
            torch.testing.assert_close(u, raw_u + .1 * (model.cohort_adj.to_dense() @ raw_u))
            torch.testing.assert_close(i, raw_i)
            loss = model.calculate_loss(batch)
            self.assertEqual(set(model.last_loss_terms), {'bpr', 'modality', 'alignment', 'total'})
            loss.backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            model.eval()
            torch.testing.assert_close(model.full_sort_predict([batch[0]]), u[batch[0]] @ i.T)
            self.assertIsNotNone(model._eval_embeddings)
            model.train(); self.assertIsNone(model._eval_embeddings)

    def test_four_job_queue_and_resume_without_checkpoints(self):
        workspace = Path(self.folder.name); data = workspace / 'data/baby'; data.mkdir(parents=True)
        for name in ('image_feat.npy', 'text_feat.npy'):
            shutil.copy(workspace / 'tiny' / name, data / name)
        lines = ['userID\titemID\tx_label']
        for u in range(3):
            lines.extend(f'{u}\t{i}\t0' for i in {0, *range(u * 7 + 1, u * 7 + 8)})
            lines.extend([f'{u}\t22\t1', f'{u}\t23\t2'])
        (data / 'baby.inter').write_text('\n'.join(lines) + '\n')
        output = workspace / 'runs'
        command = [sys.executable, str(ROOT / 'experiments/run_night.py'), '--suite', 'freedom_cohort',
                   '--data-path', str(data.parent), '--cpu', '--epochs', '1', '--hours', '.2', '--output', str(output)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=180)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        rows = json.loads((output / 'summary.json').read_text())
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(x['state'] == 'complete' for x in rows), rows)
        self.assertFalse(list(output.rglob('*.pt')) + list(output.rglob('*.pth')))
        mtimes = {path: path.stat().st_mtime_ns for path in output.glob('*/result.json')}
        resumed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertEqual(mtimes, {path: path.stat().st_mtime_ns for path in mtimes})


if __name__ == '__main__':
    unittest.main()
