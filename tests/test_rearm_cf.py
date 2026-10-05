"""Formula oracles, all CF variants' backward passes, and queue integration."""
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
    from rearm_cf import REARMCF, sampled_loss
    from rearm_cf_plan import experiments
    from rearm_capacity_plan import experiments as capacity_experiments
    from rearm_capacity import ATTENTION, META, PROJECTORS
    from rearm_cf_graphs import ultra_neighbors, cir_direction, cagcn, normalize, graphda
    from model import REARM
    torch.set_num_threads(1)
    # Independent dense oracles for equations and neighbor indexing.
    dense = np.array([[1,1,0,0], [1,0,1,0], [1,1,1,0], [0,1,0,1]], np.float32)
    r = sp.csr_matrix(dense)
    g = dense.T @ dense
    degree = g.sum(1)
    omega = g / (degree - g.diagonal())[:, None] * np.sqrt(degree[:,None]/degree[None,:])
    np.fill_diagonal(omega, 0)
    ids, weights = ultra_neighbors(r, 3, 2)
    for i in range(4):
        nonzero = weights[i] > 0
        np.testing.assert_allclose(weights[i, nonzero], omega[i, ids[i, nonzero]], atol=1e-7)
        np.testing.assert_allclose(np.sort(weights[i]), np.sort(omega[i])[-3:], atol=1e-7)
    for sim in ('jc', 'lhn'):
        d = dense.sum(0)
        simmat = g/(d[:,None]+d[None,:]-g if sim == 'jc' else d[:,None]*d[None,:])
        expected = dense * (dense @ simmat) / dense.sum(1)[:, None]
        np.testing.assert_allclose(cir_direction(r, sim, 2).toarray(), expected, atol=1e-7)
        base = normalize(sp.bmat([[None, r], [r.T, None]], format='csr'))
        changed = cagcn(r, sim, .7, 2)
        np.testing.assert_allclose(changed.sum(1), base.sum(1), atol=1e-6)
        np.testing.assert_array_equal(changed.toarray() != 0, base.toarray() != 0)
    empty_item = sp.hstack([r, sp.csr_matrix((4,1))]).tocsr()
    assert not ultra_neighbors(empty_item, 2)[1][-1].any()
    a,p,n = torch.randn(3,4),torch.randn(3,4),torch.randn(3,5,4)
    an,pn,nn = [torch.nn.functional.normalize(x,dim=-1) for x in (a,p,n)]
    pos, neg = (an*pn).sum(-1)/.1,(an[:,None]*nn).sum(-1)/.1
    torch.testing.assert_close(sampled_loss(a,p,n,.1), torch.logsumexp(neg,1)-pos)
    torch.testing.assert_close(sampled_loss(a,p,n,.1,True), torch.nn.functional.softplus(sampled_loss(a,p,n,.1)))

    nusers,nitems,dim = 8,6,4
    rr = sp.coo_matrix((np.ones(16),(np.repeat(np.arange(8),2),
                       np.array([[u%6,(u+1)%6] for u in range(8)]).ravel())),shape=(8,6))
    vf,tf = torch.randn(6,7),torch.randn(6,5)
    history = torch.tensor(rr.toarray()).float()/2
    def dataset():
        # Fresh dictionaries: upstream topk_sample mutates short neighbor lists.
        return SimpleNamespace(n_users=nusers,n_items=nitems,i_v_feat=vf.clone(),i_t_feat=tf.clone(),
            u_v_interest=history@vf,u_t_interest=history@tf,
            topK_users=list(range(8)),topK_items=list(range(6)),topK_users_counts=[2]*8,topK_items_counts=[2]*6,
            dict_user_co_occ_graph={u:[[(u+1)%8,(u+2)%8],[2.,1.]] for u in range(8)},
            dict_item_co_occ_graph={i:[[(i+1)%6,(i+2)%6],[2.,1.]] for i in range(6)},
            i_mm_adj=torch.eye(6).to_sparse(),u_mm_adj=torch.eye(8).to_sparse(),sparse_inter_matrix=lambda form:rr.asformat(form))
    config = dict(embedding_dim=dim,reg_weight=.0005,device='cpu',cl_tmp=.6,cl_loss_weight=5e-6,
        diff_loss_weight=5e-5,n_layers=2,num_user_co=2,num_item_co=2,user_aggr_mode='softmax',
        n_ii_layers=1,n_uu_layers=1,rank=2,uu_co_weight=.4,ii_co_weight=.2,s_drop=0.,m_drop=0.,item_knn_k=2)
    users = torch.tensor([[0,0],[1,1],[2,2]])
    items = torch.tensor([[8,11],[9,12],[10,13]])
    torch.manual_seed(99)
    original = REARM(SimpleNamespace(**config),dataset()).eval()
    torch.manual_seed(99)
    reference = REARMCF(SimpleNamespace(**config),dataset(),dict(aux='official')).eval()
    reference_rng = torch.get_rng_state().clone()
    for expected,actual in zip(original.loss(users,items),reference.loss(users,items)):
        torch.testing.assert_close(torch.as_tensor(actual).float(),torch.as_tensor(expected).float())
    # Exact exclusion works in both directions, only TRAIN positives are forbidden.
    sample = reference.sample(torch.arange(8),100)
    assert not reference.contains(torch.arange(8)[:,None],sample).any()
    reverse = reference.sample(torch.arange(6),100,reverse=True)
    assert not reference.contains(reverse,torch.arange(6)[:,None]).any()
    reference.forward()
    parts = reference.id_parts()
    torch.testing.assert_close(parts[0]+parts[1], reference.cf_id)
    negatives = reference.sample(users[:,0],32)
    expected = sampled_loss(reference.cf_id[users[:,0]], reference.cf_id[items[:,0]],
                            reference.cf_id[negatives+8],.1,True).mean()
    torch.testing.assert_close(reference.nt_loss(users[:,0],items[:,0]-8,negatives),expected)
    # Each planned model (including removal of UI propagation) has finite gradients.
    for spec in experiments():
        cfg = dict(config)
        for k,v in spec['config']['overrides'].items():
            if k == 'num_layer': cfg['n_layers'] = v
            elif k in cfg: cfg[k] = v
        model = REARMCF(SimpleNamespace(**cfg),dataset(),spec['config']['options'],graph_block=3)
        model.before_epoch(200)  # activate denoising, not just its warmup control
        losses = model.loss(users,items)
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in losses),spec['name']
        losses[0].backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients),spec['name']
        assert model.item_id_embedding.weight.grad.abs().sum() > 0,spec['name']
        if spec['config']['options']['freeze_features']:
            assert model.image_embedding.weight.grad is None
        assert 'total' in model.diagnostics()['losses']
    # Architecture ablations retain common initialization, prediction dimension,
    # and original CL/diff. Removed modules must have no optimizer parameters.
    reference_parameters = dict(reference.named_parameters())
    for spec in capacity_experiments(extended=True):
        torch.manual_seed(99)
        model = REARMCF(SimpleNamespace(**config),dataset(),spec['config']['options']).eval()
        torch.testing.assert_close(torch.get_rng_state(),reference_rng)
        opts=spec['config']['options']
        for name,p in model.named_parameters():
            if name in reference_parameters:
                torch.testing.assert_close(p,reference_parameters[name])
        count=model.parameter_breakdown()
        assert sum(g['total'] for g in count.values()) == sum(p.numel() for p in model.parameters())
        assert count['attention']['total'] == (0 if opts['attention']=='none' else 32)
        if opts['meta']=='none':
            assert count['meta']['total']==0 and all(not hasattr(model,n) for n in META)
        if opts['projector_hidden']:
            names=PROJECTORS if opts['projector_scope']=='all' else PROJECTORS[:2]
            for name in names:
                assert isinstance(getattr(model,name),torch.nn.Sequential)
                assert getattr(model,name)[0].out_features==256
                assert getattr(model,name)[2].out_features==dim
        out=model.forward()
        assert out.shape==(14,3*dim)
        if spec['name']=='cap_no_attention':
            with torch.no_grad():
                for name in ATTENTION:
                    for p in getattr(original,name).parameters(): p.zero_()
            torch.testing.assert_close(out,original.forward())
        losses=model.loss(users,items)
        losses[0].backward()
        assert torch.isfinite(losses[0])
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if opts['projector_hidden']:
            assert model.image_i_trs[0].weight.grad.abs().sum()>0
            assert model.image_i_trs[2].weight.grad.abs().sum()>0
    # GraphDA dense oracle verifies union, symmetric homogeneous edges and no self loops.
    emb = torch.randn(14,4)
    adj,stats = graphda(emb,8,rr,k=2,homogeneous_k=1,block=3)
    u,i = emb[:8],emb[8:]
    selected = torch.zeros(8,6)
    selected.scatter_(1,(u@i.T).topk(2,dim=1).indices,1.)
    reverse = torch.zeros(6,8).scatter_(1,(i@u.T).topk(2,dim=1).indices,1.)
    selected = torch.maximum(selected,reverse.T).numpy()
    assert stats['ui_edges'] == int(selected.sum())
    np.testing.assert_array_equal(adj[:8,8:].toarray()!=0,selected!=0)
    np.testing.assert_allclose(adj.toarray(),adj.toarray().T,atol=1e-7)
    assert not adj.diagonal().any()
    from rearm_cf_training import prepare_teacher
    student = REARMCF(SimpleNamespace(**config), dataset(), dict(graphda_epochs=1, graphda_k=2), graph_block=3)
    net = SimpleNamespace(model=student, train_data=[(users,items)], learning_rate=.001, reg_weight=.0005,
                          config=SimpleNamespace(learning_rate_scheduler=[1.,50]), logger=SimpleNamespace(info=lambda *args:None))
    net.optimizer=torch.optim.AdamW(student.parameters(),lr=net.learning_rate,weight_decay=net.reg_weight)
    net.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(net.optimizer,lambda epoch:1.)
    before={k:p.detach().clone() for k,p in student.named_parameters()}
    graph_before=student.norm_adj.to_dense().clone()
    prepare_teacher(net)
    for k,p in student.named_parameters():
        torch.testing.assert_close(p,before[k],rtol=0,atol=0)
    assert not net.optimizer.state and not student.edge_seen.any()
    assert not torch.allclose(graph_before,student.norm_adj.to_dense())
    print('72 CF + 8 capacity variants: finite backward, shared initialization, loss/graph oracles passed')


class REARMCFTest(unittest.TestCase):
    def test_default_queues_skip_reference(self):
        for suite, count in [('capacity',3),('capacity_extended',7),('quick',13),('all',71)]:
            command=[sys.executable,str(ROOT/'experiments/run_rearm_cf.py'),'--data-path','/unused',
                     '--output','/unused-output','--suite',suite,'--dry-run']
            run=subprocess.run(command,cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=20)
            self.assertEqual(run.returncode,0,run.stderr)
            jobs=json.loads(run.stdout)['jobs']
            self.assertEqual(len(jobs),count)
            self.assertTrue(all(j['variant'] not in ('reference','cap_original') for j in jobs))
        for extra in (['--include-reference'],['--variants','cap_original']):
            run=subprocess.run(command[:command.index('--suite')]+['--suite','capacity','--dry-run']+extra,
                               cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=20)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertIn('cap_original',[j['variant'] for j in json.loads(run.stdout)['jobs']])

    def test_formulas_and_all_models(self):
        run = subprocess.run([sys.executable,__file__,'--checks'],cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=120)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)

    def test_queue_and_teacher(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data=root/'data/baby'; data.mkdir(parents=True)
            rng=np.random.default_rng(123)
            with (data/'baby.inter').open('w') as f:
                f.write('userID\titemID\tx_label\n')
                for user in range(48):
                    for offset in range(6):
                        f.write(f'{user}\t{(user+offset)%24}\t{0 if offset<4 else offset-3}\n')
            np.save(data/'image_feat.npy',rng.normal(size=(24,12)).astype('float32'))
            np.save(data/'text_feat.npy',rng.normal(size=(24,8)).astype('float32'))
            output=root/'runs'
            command=[sys.executable,str(ROOT/'experiments/run_rearm_cf.py'),'--data-path',str(data.parent),
                     '--output',str(output),'--suite','quick','--include-reference','--epochs','2','--cpu','--num-workers','0','--graph-block','13']
            run=subprocess.run(command,cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=300)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            status=json.loads((output/'status.json').read_text())
            self.assertEqual(status['state'],'complete',run.stdout+run.stderr)
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(len(summary['runs']),14)
            self.assertTrue(all(r['state']=='complete' for r in summary['runs']))
            for job in summary['runs']:
                folder=output/job['name']
                result=json.loads((folder/'result.json').read_text())
                self.assertIn(result['best_epoch'],(0,1))
                self.assertEqual(result['negative_scope'],'train')
                self.assertEqual(len((folder/'cf_diagnostics.jsonl').read_text().splitlines()),2)
            manifest=json.loads((output/'graphda_residual_s2025/manifest.json').read_text())
            teacher=manifest['cf_graphs']['graphda']
            self.assertTrue(teacher['student_reinitialized'])
            self.assertEqual(teacher['teacher_epochs'],120)
            self.assertFalse(list(output.rglob('*.pt'))+list(output.rglob('*.pth')))
            again=subprocess.run(command,cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=20)
            self.assertNotEqual(again.returncode,0)
            resume=subprocess.run(command+['--resume'],cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=30)
            self.assertEqual(resume.returncode,0,resume.stdout+resume.stderr)
            # Explicit retry preserves a failed attempt, then replaces only its new run.
            failed=output/'without_cl_s2025/status.json'
            failed.write_text(json.dumps(dict(state='interrupted',error='simulated interruption')))
            retry=subprocess.run(command+['--resume','--retry-failed'],cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=60)
            self.assertEqual(retry.returncode,0,retry.stdout+retry.stderr)
            self.assertEqual(len(list((output/'attempts').iterdir())),1)
            self.assertEqual(json.loads((output/'status.json').read_text())['state'],'complete')

    def test_capacity_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data=root/'data/baby'; data.mkdir(parents=True)
            rng=np.random.default_rng(123)
            with (data/'baby.inter').open('w') as f:
                f.write('userID\titemID\tx_label\n')
                for user in range(48):
                    for offset in range(6):
                        f.write(f'{user}\t{(user+offset)%24}\t{0 if offset<4 else offset-3}\n')
            np.save(data/'image_feat.npy',rng.normal(size=(24,12)).astype('float32'))
            np.save(data/'text_feat.npy',rng.normal(size=(24,8)).astype('float32'))
            output=root/'runs'
            command=[sys.executable,str(ROOT/'experiments/run_rearm_cf.py'),'--data-path',str(data.parent),
                     '--output',str(output),'--suite','capacity_extended','--include-reference','--epochs','2','--cpu','--num-workers','0','--graph-block','13']
            run=subprocess.run(command,cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=180)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertEqual(json.loads((output/'status.json').read_text())['state'],'complete',run.stdout+run.stderr)
            rows=json.loads((output/'summary.json').read_text())['runs']
            self.assertEqual(len(rows),8)
            base=next(r for r in rows if r['variant']=='cap_original')
            plain=next(r for r in rows if r['variant']=='cap_no_attention')
            self.assertEqual(base['parameters']-plain['parameters'],32)
            for row in rows:
                self.assertIsNotNone(row['train_seconds_at_best'])
                self.assertGreater(row['train_seconds_at_best'],0)
                result=json.loads((output/row['name']/'result.json').read_text())
                self.assertEqual(result['cf']['options']['aux'],'official')
                self.assertEqual(row['parameters'],sum(v['total'] for v in result['parameter_breakdown'].values()))
            self.assertFalse(list(output.rglob('*.pt'))+list(output.rglob('*.pth')))


if __name__=='__main__':
    if '--checks' in sys.argv: checks()
    else: unittest.main()
