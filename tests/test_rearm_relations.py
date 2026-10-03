"""Relation partition, score consistency, gradients, and six-arm queue smoke test."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def model_checks():
    sys.path.insert(0, str(ROOT / 'third_party/rearm'))
    sys.path.insert(0, str(ROOT / 'experiments'))
    from types import SimpleNamespace
    import torch
    import scipy.sparse as sp
    from utils import helper
    from rearm_runtime import install
    install(helper, block_size=3)
    from rearm_relations import REARMRelations, semantic_knn, partition_semantics, row_stochastic
    from model import REARM

    def sparse(rows, cols, vals, n=4):
        return torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(vals), (n, n)).coalesce()

    visual = sparse([0, 0, 1], [1, 2, 2], [.8, .4, .6])
    text = sparse([0, 2], [1, 3], [.6, .9])
    agree, vonly, tonly = partition_semantics(visual, text)
    torch.testing.assert_close(agree.to_dense()[0, 1], torch.tensor(.7))
    assert agree._nnz() == 1 and vonly._nnz() == 2 and tonly._nnz() == 1
    for graph in (agree, vonly, tonly, sparse([], [], [])):
        normalized = row_stochastic(graph)
        torch.testing.assert_close(normalized.to_dense().sum(1), torch.ones(4))
    empty = sparse([], [], [])
    assert partition_semantics(empty, empty)[0]._nnz() == 0
    assert partition_semantics(visual, empty)[1]._nnz() == 3

    torch.manual_seed(45)
    x = torch.randn(12, 7)
    graph = semantic_knn(x, 3, block=4)
    permutation = torch.randperm(12)
    permuted = semantic_knn(x[permutation], 3, block=5).to_dense()
    torch.testing.assert_close(permuted, graph.to_dense()[permutation][:, permutation])
    assert torch.all(graph.indices()[0] != graph.indices()[1])
    assert torch.all(graph.values() > 0)

    nusers, nitems, dim = 8, 6, 4
    r = sp.coo_matrix((np.ones(16), (np.repeat(np.arange(8), 2),
                      np.array([[u % 6, (u + 1) % 6] for u in range(8)]).flatten())), shape=(8, 6))
    vf, tf = torch.randn(6, 7), torch.randn(6, 5)
    history = torch.tensor(r.toarray()).float() / 2
    dataset = SimpleNamespace(n_users=nusers, n_items=nitems, i_v_feat=vf, i_t_feat=tf,
        u_v_interest=history @ vf, u_t_interest=history @ tf,
        topK_users=list(range(8)), topK_items=list(range(6)),
        topK_users_counts=[2] * 8, topK_items_counts=[2] * 6,
        dict_user_co_occ_graph={u: [[(u + 1) % 8, (u + 2) % 8], [2., 1.]] for u in range(8)},
        dict_item_co_occ_graph={i: [[(i + 1) % 6, (i + 2) % 6], [2., 1.]] for i in range(6)},
        i_mm_adj=torch.eye(6).to_sparse(), u_mm_adj=torch.eye(8).to_sparse(),
        sparse_inter_matrix=lambda form: r)
    config = SimpleNamespace(embedding_dim=dim, reg_weight=.0005, device='cpu', cl_tmp=.6,
        cl_loss_weight=5e-6, diff_loss_weight=5e-5, n_layers=2, num_user_co=2, num_item_co=2,
        user_aggr_mode='softmax', n_ii_layers=1, n_uu_layers=1, rank=2,
        uu_co_weight=.4, ii_co_weight=.2, s_drop=0., m_drop=0., item_knn_k=2)
    models = {}
    for mode in ('merged_single', 'merged_matched', 'typed_fixed', 'typed_global', 'typed_user'):
        torch.manual_seed(99)
        models[mode] = REARMRelations(config, dataset, mode=mode, graph_block=3)
        models[mode].eval()
    users = torch.tensor([[0, 0], [1, 1], [2, 2]])
    items = torch.tensor([[8, 11], [9, 12], [10, 13]])
    base_params = dict(models['merged_single'].named_parameters())
    for name, parameter in base_params.items():
        for model in models.values():
            torch.testing.assert_close(dict(model.named_parameters())[name], parameter)
    # Repeated identical graphs must not gain score scale simply from four branches.
    single = models['merged_single'].full_sort_predict((torch.arange(8),))
    repeated = models['merged_matched'].full_sort_predict((torch.arange(8),))
    torch.testing.assert_close(single, repeated)
    # At uniform gate initialization, fixed, global and user arms are identical.
    fixed = models['typed_fixed'].full_sort_predict((torch.arange(8),))
    for mode in ('typed_global', 'typed_user'):
        torch.testing.assert_close(models[mode].full_sort_predict((torch.arange(8),)), fixed)
    for mode, model in models.items():
        model.train()
        losses = model.loss(users, items)
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in losses)
        model.eval()
        scores = model.full_sort_predict((torch.arange(8),))
        expected = torch.nn.functional.softplus(scores[users[:, 0], items[:, 1] - 8] -
                                               scores[users[:, 0], items[:, 0] - 8]).mean()
        torch.testing.assert_close(losses[1], expected)
        losses[0].backward()
        assert model.item_id_embedding.weight.grad.abs().sum() > 0
        if mode in ('typed_global', 'typed_user'):
            assert model.relation_gate.weight.grad.abs().sum() > 0
        model.train()
        assert model._eval_representation is None
    # Merged-single preserves all official losses, not just BPR.
    model = models['merged_single']
    graph_before = model.stre_ii_graph
    model.stre_ii_graph = model.relation_merged
    original_forward = model.forward
    model.forward = lambda: REARM.forward(model)
    expected = REARM.loss(model, users, items)
    model.forward = original_forward
    model.stre_ii_graph = graph_before
    actual = model.loss(users, items)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(torch.as_tensor(a), torch.as_tensor(b))
    # Global gate stays identical across users after nonzero weights; user gate can vary.
    for mode in ('typed_global', 'typed_user'):
        torch.nn.init.normal_(models[mode].relation_gate.weight)
    assert models['typed_global'].gate_weights().std(0).max() < 1e-6
    assert models['typed_user'].gate_weights().std(0).max() > 1e-4
    print('Partition, permutation, paired initialization, score, losses and gradient checks passed')


class REARMRelationsTest(unittest.TestCase):
    def test_model_invariants(self):
        run = subprocess.run([sys.executable, __file__, '--checks'], cwd=ROOT, env=ENV,
                             capture_output=True, text=True, timeout=90)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_six_arm_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data/baby'; data.mkdir(parents=True)
            rng = np.random.default_rng(123)
            with (data / 'baby.inter').open('w') as stream:
                stream.write('userID\titemID\tx_label\n')
                for user in range(48):
                    for offset in range(6):
                        stream.write(f'{user}\t{(user + offset) % 24}\t{0 if offset < 4 else offset - 3}\n')
            np.save(data / 'image_feat.npy', rng.normal(size=(24, 12)).astype('float32'))
            np.save(data / 'text_feat.npy', rng.normal(size=(24, 8)).astype('float32'))
            output = root / 'runs'
            command = [sys.executable, str(ROOT / 'experiments/run_rearm_relations.py'),
                       '--data-path', str(data.parent), '--output', str(output), '--cpu',
                       '--epochs', '2', '--num-workers', '0', '--graph-block', '13']
            run = subprocess.run(command, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=240)
            logs = '\n'.join(p.read_text() for p in output.glob('*/console.log'))
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr + logs)
            self.assertEqual(json.loads((output / 'status.json').read_text())['state'], 'complete')
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(len(summary['runs']), 6)
            self.assertTrue(all(r['state'] == 'complete' for r in summary['runs']))
            parameters = {}
            for folder in sorted(output.glob('*_s2025')):
                result = json.loads((folder / 'result.json').read_text())
                self.assertEqual(result['negative_scope'], 'train')
                self.assertIn(result['best_epoch'], (0, 1))
                parameters[result['relation_mode']] = result['parameters']
                if result['relation_mode'] != 'original':
                    self.assertTrue((folder / 'relation_diagnostics.jsonl').exists())
            self.assertEqual(parameters['merged_matched'], parameters['typed_user'])
            self.assertEqual(parameters['typed_global'], parameters['typed_user'])
            self.assertEqual(parameters['typed_fixed'], parameters['original'])
            self.assertFalse(list(output.rglob('*.pt')) + list(output.rglob('*.pth')))
            again = subprocess.run(command, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(again.returncode, 0)
            resume = subprocess.run(command + ['--resume'], cwd=ROOT, env=ENV,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(resume.returncode, 0, resume.stdout + resume.stderr)


if __name__ == '__main__':
    if '--checks' in sys.argv:
        model_checks()
    else:
        unittest.main()
