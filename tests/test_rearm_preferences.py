"""Preference reparameterization oracles and actual training/prediction consistency."""
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
    from rearm_preferences import InterestBank
    from rearm_preferences_plan import experiments
    torch.set_num_threads(1)
    rr = sp.coo_matrix((np.ones(16), (np.repeat(np.arange(8), 2),
                       np.array([[u % 6, (u+1) % 6] for u in range(8)]).ravel())), shape=(8, 6))
    torch.manual_seed(7)
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

    cfg = SimpleNamespace(embedding_dim=4, reg_weight=.0005, device='cpu', cl_tmp=.6,
        cl_loss_weight=5e-6, diff_loss_weight=5e-5, n_layers=2, num_user_co=2, num_item_co=2,
        user_aggr_mode='softmax', n_ii_layers=1, n_uu_layers=1, rank=2,
        uu_co_weight=.4, ii_co_weight=.2, s_drop=0., m_drop=0.)
    users = torch.tensor([[0, 0], [1, 1], [2, 2]])
    items = torch.tensor([[8, 11], [9, 12], [10, 13]])
    torch.manual_seed(99)
    original = REARMCF(cfg, dataset(), dict(aux='official')).eval()
    original_out = original.forward().detach()
    for spec in experiments(extended=True):
        torch.manual_seed(99)
        options = dict(spec['config']['options'], interest_chunk=2)
        m = REARMCF(cfg, dataset(), options).eval()
        out = m.forward()
        if spec['name'] in ('cap_original', 'pref_projected64', 'pref_history_residual64', 'pref_freeze_users', 'pref_freeze_items', 'pref_freeze_both'):
            torch.testing.assert_close(out, original_out)
        if options['preference_mode'] in ('projected', 'residual'):
            assert m.user_v_prefer.shape == (8, 4)
        if options['preference_mode'] == 'projected':
            assert not list(m.image_u_trs.parameters())
        breakdown = m.parameter_breakdown()
        assert sum(x['total'] for x in breakdown.values()) == sum(p.numel() for p in m.parameters())
        assert breakdown['interests']['total'] == 2*8*4*options['interest_heads']
        loss = m.loss(users, items)
        # Independently verify both training BPR and full-sort prediction include
        # exactly the same interest correction, across item chunks/cache reuse.
        with torch.no_grad():
            scores = m.full_sort_predict((users[:, 0],))
            torch.testing.assert_close(scores, m.full_sort_predict((users[:, 0],)))
            positive = scores[torch.arange(3), items[:, 0]-8]
            negative = scores[torch.arange(3), items[:, 1]-8]
            torch.testing.assert_close(loss[1], torch.nn.functional.softplus(negative-positive).mean())
            torch.testing.assert_close(m.pair_score(m.cf_out, users[:, 0], items[:, 0]-8), positive)
        loss[0].backward()
        assert torch.isfinite(loss[0])
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        if options['preference_mode'] == 'frozen':
            assert m.user_v_prefer.grad is None
        else:
            assert m.user_v_prefer.grad.abs().sum() > 0
        if options['freeze_item_features']:
            assert m.image_embedding.weight.grad is None
        if options['interest_heads']:
            assert m.interest_bank.visual_codes.grad.abs().sum() > 0
            assert m.interest_bank.text_codes.grad.abs().sum() > 0
        optimizer = torch.optim.AdamW(m.parameters(), lr=.001)
        optimizer.step()
        m.train()
        assert m._eval_out is None

    one = InterestBank(2, 4, 1, .1, 'cpu')
    many = InterestBank(2, 4, 4, .1, 'cpu')
    torch.testing.assert_close(one.visual_codes[:, 0], many.visual_codes[:, 0])
    x = torch.tensor([[.2], [-.3]])
    torch.testing.assert_close(one.pool(x, 1), many.pool(x.expand(-1, 4), 1))
    # Candidate-dependent matching differs from averaging the head embeddings.
    context = [(torch.tensor([[[1., 0.], [0., 1.]]]), torch.eye(2))]*2
    bank = InterestBank(1, 2, 2, .1, 'cpu')
    actual = bank.block(context, torch.tensor([0]), 0, 2)
    expected = .1 * (torch.logsumexp(torch.tensor([10., 0.]), 0) - np.log(2))
    torch.testing.assert_close(actual, expected.expand(1, 2))
    assert not torch.allclose(actual, torch.full_like(actual, .5))
    print('Preference forward-equivalence, gradients, parameter counts, multi-interest score and cache checks passed')


class PreferenceTests(unittest.TestCase):
    def test_oracles(self):
        r = subprocess.run([sys.executable, __file__, '--checks'], cwd=ROOT, env=ENV,
                           capture_output=True, text=True, timeout=90)
        self.assertEqual(r.returncode, 0, r.stdout+r.stderr)

    def test_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root/'data/baby'; data.mkdir(parents=True)
            rng = np.random.default_rng(123)
            with (data/'baby.inter').open('w') as f:
                f.write('userID\titemID\tx_label\n')
                for u in range(48):
                    for offset in range(6):
                        f.write(f'{u}\t{(u+offset)%24}\t{0 if offset<4 else offset-3}\n')
            np.save(data/'image_feat.npy', rng.normal(size=(24,12)).astype('float32'))
            np.save(data/'text_feat.npy', rng.normal(size=(24,8)).astype('float32'))
            out = root/'runs'
            r = subprocess.run([sys.executable, str(ROOT/'experiments/run_rearm_cf.py'),
                '--data-path', str(data.parent), '--output', str(out), '--suite', 'preferences_extended',
                '--epochs', '2', '--cpu', '--num-workers', '0', '--graph-block', '13'],
                cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=240)
            self.assertEqual(r.returncode, 0, r.stdout+r.stderr)
            self.assertEqual(json.loads((out/'status.json').read_text())['state'], 'complete')
            rows = json.loads((out/'summary.json').read_text())['runs']
            self.assertEqual(len(rows), 15)
            self.assertNotIn('cap_original', [r['variant'] for r in rows])
            self.assertTrue(all(r['state']=='complete' for r in rows))
            self.assertFalse(list(out.rglob('*.pt'))+list(out.rglob('*.pth')))


if __name__ == '__main__':
    if '--checks' in sys.argv:
        checks()
    else:
        unittest.main()
