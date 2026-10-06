"""Shared-item user basis, norm bounds, relabeling and queue integration."""
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


def checks():
    sys.path.insert(0, str(ROOT/'third_party/rearm'))
    sys.path.insert(0, str(ROOT/'experiments'))
    from types import SimpleNamespace
    import scipy.sparse as sp
    import torch
    from utils import helper
    from rearm_runtime import install
    install(helper, 3)
    from rearm_cf import REARMCF
    from rearm_shared_users import history_weights, bounded_residual, diagnostic_pairs
    from rearm_shared_plan import experiments
    torch.set_num_threads(1)
    torch.manual_seed(123)
    rr = sp.coo_matrix((np.ones(16), (np.repeat(np.arange(8), 2),
                       np.array([[u % 6, (u+1) % 6] for u in range(8)]).ravel())), shape=(8, 6))
    vf, tf = torch.randn(6, 7), torch.randn(6, 5)
    history = torch.tensor(rr.toarray()).float()/2

    def dataset():
        return SimpleNamespace(n_users=8, n_items=6, i_v_feat=vf.clone(), i_t_feat=tf.clone(),
            u_v_interest=history@vf, u_t_interest=history@tf,
            topK_users=list(range(8)), topK_items=list(range(6)), topK_users_counts=[2]*8, topK_items_counts=[2]*6,
            dict_user_co_occ_graph={u: [[(u+1)%8, (u+2)%8], [2., 1.]] for u in range(8)},
            dict_item_co_occ_graph={i: [[(i+1)%6, (i+2)%6], [2., 1.]] for i in range(6)},
            i_mm_adj=torch.eye(6).to_sparse(), u_mm_adj=torch.eye(8).to_sparse(),
            sparse_inter_matrix=lambda form: rr.asformat(form))

    config = dict(embedding_dim=4, reg_weight=.0005, device='cpu', cl_tmp=.6,
        cl_loss_weight=5e-6, diff_loss_weight=5e-5, n_layers=2, num_user_co=2, num_item_co=2,
        user_aggr_mode='softmax', n_ii_layers=1, n_uu_layers=1, rank=2,
        uu_co_weight=.4, ii_co_weight=.2, s_drop=0., m_drop=0.)
    users = torch.tensor([[0, 0], [1, 1], [2, 2]])
    items = torch.tensor([[8, 11], [9, 12], [10, 13]])
    # Relabeling users/items is only a diagnostic oracle, not a new server experiment.
    pu, pi = np.array([5, 1, 6, 0, 4, 3, 7, 2]), np.array([3, 1, 5, 0, 4, 2])
    z = np.random.default_rng(5).normal(size=(6, 4))
    for beta in (0., .5):
        a = history_weights(rr, beta)
        b = history_weights(rr.tocsr()[pu][:, pi], beta)
        np.testing.assert_allclose(b @ z[pi], (a @ z)[pu], atol=1e-6)
        np.testing.assert_allclose(np.asarray(a.sum(1)), 1., atol=1e-7)
    isolated = sp.vstack([rr, sp.csr_matrix((1,6))]).tocsr()
    assert not history_weights(isolated).getrow(8).nnz
    base, delta = torch.randn(10, 4), torch.randn(10, 4)*1e5
    changed = bounded_residual(base, delta, torch.full((10,1), .1))
    assert torch.all((changed-base).norm(dim=1) <= .100001*base.norm(dim=1))
    for spec in experiments(extended=True)[1:]:
        torch.manual_seed(99)
        cfg = dict(config, **spec['config']['overrides'])
        c = spec['config']['options']
        m = REARMCF(SimpleNamespace(**cfg), dataset(), c).eval()
        m.forward()
        p = torch.tensor(history_weights(rr, c['shared_pop_power']).toarray())
        expected = p @ m.image_u_trs.encoder(m.image_embedding.weight)
        torch.testing.assert_close(m._projected['image_u_trs'], expected)
        # Identical training histories produce identical shared preference inputs.
        torch.testing.assert_close(expected[0], expected[6])
        if c['shared_users'] == 'all':
            torch.testing.assert_close(m.user_id_embedding.weight, p @ m.item_id_embedding.weight)
        if not c['shared_residual']:
            assert 'user_v_prefer' not in dict(m.named_parameters())
        # The *current* item table affects all consumers, not only initialization.
        before = m.image_u_trs(m.user_v_prefer).detach().clone()
        with torch.no_grad():
            m.image_embedding.weight[0].add_(.2)
        after = m.image_u_trs(m.user_v_prefer).detach()
        assert not torch.allclose(before[0], after[0])
        torch.testing.assert_close(after[0]-before[0], after[6]-before[6])
        m.zero_grad(set_to_none=True)
        # An isolated user-input loss must backpropagate to its consumed items.
        m.image_u_trs(m.user_v_prefer)[0].square().sum().backward()
        assert m.image_embedding.weight.grad[:2].abs().sum() > 0
        assert m.image_embedding.weight.grad[2:].abs().sum() == 0
        m.zero_grad(set_to_none=True)
        losses = m.loss(users, items)
        losses[0].backward()
        assert torch.isfinite(losses[0])
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        if c['shared_residual']:
            assert m.user_v_prefer.grad.abs().sum() > 0
            assert m.user_id_embedding.residual.grad.abs().sum() > 0
        out = m.forward().detach().clone()
        torch_state, np_state = torch.get_rng_state().clone(), np.random.get_state()
        diag = m.diagnostics()['user_geometry']
        assert 'final_user' in diag and diag['pair_counts']['shared_item'] > 0
        torch.testing.assert_close(torch.get_rng_state(), torch_state)
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        m.geometry_pairs = {}  # Remove every ID-distance diagnostic: predictions identical.
        torch.testing.assert_close(m.forward(), out)
        with torch.no_grad():
            scores = m.full_sort_predict((users[:,0],))
            expected_loss = torch.nn.functional.softplus(scores[torch.arange(3), items[:,1]-8]-scores[torch.arange(3), items[:,0]-8]).mean()
            torch.testing.assert_close(losses[1], expected_loss)
        optimizer = torch.optim.AdamW(m.parameters(), lr=.001)
        optimizer.step()
        assert sum(v['total'] for v in m.parameter_breakdown().values()) == sum(p.numel() for p in m.parameters())
    # Pair controls use the same anchor, exact target degree and far distance.
    rng = np.random.default_rng(42)
    r = sp.csr_matrix((np.ones(1600), (np.repeat(np.arange(800),2), rng.integers(0,80,1600))), shape=(800,80))
    r.sum_duplicates(); r.data[:] = 1
    pairs = diagnostic_pairs(r, count=64)
    degrees = np.diff(r.indptr)
    for offset in (1,4,16,64):
        near, far = pairs[f'near_{offset}'], pairs[f'degree_far_{offset}']
        assert len(near)
        np.testing.assert_array_equal(near[:,0], far[:,0])
        np.testing.assert_array_equal(degrees[near[:,1]], degrees[far[:,1]])
        assert np.all(np.abs(far[:,0]-far[:,1])>256)
    print('Shared-user aggregation, live gradients, residual bounds, relabeling and diagnostic isolation passed')


class SharedUserTests(unittest.TestCase):
    def test_oracles(self):
        r = subprocess.run([sys.executable, __file__, '--checks'], cwd=ROOT, env=ENV,
                           capture_output=True, text=True, timeout=90)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)

    def test_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data/baby'; data.mkdir(parents=True)
            rng=np.random.default_rng(123)
            with (data/'baby.inter').open('w') as f:
                f.write('userID\titemID\tx_label\n')
                for u in range(48):
                    for offset in range(6):
                        f.write(f'{u}\t{(u+offset)%24}\t{0 if offset<4 else offset-3}\n')
            np.save(data/'image_feat.npy',rng.normal(size=(24,12)).astype('float32'))
            np.save(data/'text_feat.npy',rng.normal(size=(24,8)).astype('float32'))
            out=root/'runs'
            r=subprocess.run([sys.executable,str(ROOT/'experiments/run_rearm_cf.py'),
                '--data-path',str(data.parent),'--output',str(out),'--suite','shared_users_extended',
                '--epochs','2','--cpu','--num-workers','0','--graph-block','13'],
                cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=180)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            self.assertEqual(json.loads((out/'status.json').read_text())['state'],'complete')
            rows=json.loads((out/'summary.json').read_text())['runs']
            self.assertEqual(len(rows),7)
            for row in rows:
                result=json.loads((out/row['name']/'result.json').read_text())
                self.assertIn('user_geometry', result['cf_diagnostics'])
                self.assertEqual(result['cf']['options']['aux'],'official')
            self.assertFalse(list(out.rglob('*.pt'))+list(out.rglob('*.pth')))


if __name__ == '__main__':
    if '--checks' in sys.argv: checks()
    else: unittest.main()
