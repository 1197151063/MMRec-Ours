import sys
import copy
import json
import shutil
import subprocess
import unittest
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import yaml
import test_simmrec as fixtures
from models.relationrec import RelationRec
from common.relation_neighbors import user_neighbors,feature_neighbors
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from relation_plan import build_plan


class RelationTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        root=Path(__file__).resolve().parents[1]
        conf=yaml.safe_load((root/'src/configs/model/RelationRec.yaml').read_text())
        self.config.update({k:v[0] if isinstance(v,list) else v for k,v in conf.items() if k!='hyper_parameters'})
        self.r=sp.coo_matrix((np.ones(8),([0,0,1,1,1,1,2,2],[0,1,0,1,2,3,4,5])),shape=(3,24))
        self.loader.inter_matrix=lambda form:self.r

    def test_neighbors_and_consistent_relabeling(self):
        r=sp.csr_matrix([[1,1,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,1]],dtype=np.float32)
        ties=np.array([.1,.4,.8,.6])
        ids,w=user_neighbors(r,2,tie_keys=ties)
        self.assertEqual(w[3].sum(),0)
        self.assertFalse(any(ids[u][w[u]>0].eq(u).any() for u in range(4)))
        up=np.array([2,0,3,1]);inv=np.argsort(up)
        actual,aw=user_neighbors(r[up],2,tie_keys=ties[up])
        np.testing.assert_array_equal(up[actual.numpy()][aw.numpy()>0],ids[up].numpy()[aw.numpy()>0])
        torch.testing.assert_close(aw,w[up])
        x=torch.tensor([[1.,0.],[.9,.1],[0.,1.],[-1.,0.]])
        fi,fw=feature_neighbors(x,2,tie_keys=ties)
        ai,aw=feature_neighbors(x[up],2,tie_keys=ties[up])
        np.testing.assert_array_equal(up[ai.numpy()][aw.numpy()>0],fi[up].numpy()[aw.numpy()>0])
        torch.testing.assert_close(aw,fw[up])

    def test_loss_formula_and_gradient(self):
        a=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True)
        n=torch.tensor([[[1.,0.],[0.,1.]],[[1.,1.],[2.,1.]]],requires_grad=True)
        w=torch.tensor([[.25,.75],[0.,0.]])
        actual=RelationRec.neighbor_loss(a,n,w)
        expected=(.25*torch.nn.functional.softplus(torch.tensor(-1.))+.75*torch.nn.functional.softplus(torch.tensor(-2.)))/2
        torch.testing.assert_close(actual,expected)
        actual.backward()
        self.assertEqual(a.grad[1].abs().sum(),0)
        self.assertEqual(n.grad[1].abs().sum(),0)

    def test_all_variants_and_filtered_negatives(self):
        jobs=build_plan();self.assertEqual(len(jobs),12)
        batch=torch.tensor([[0,1,2],[0,2,4]])
        for j in jobs:
            model=RelationRec(dict(self.config,**j['overrides']),self.loader)
            self.assertFalse(any('group' in key or 'pe'==key for key in model.state_dict()))
            self.assertEqual(hasattr(model,'ui_adj'),j['overrides']['ui_layers']>0)
            negatives=model.sample_negatives(batch[0])
            for u in range(3):
                self.assertFalse(set(negatives[u].tolist())&set(self.r.tocsr()[u].indices))
            loss=model.calculate_loss(batch);loss.backward()
            self.assertTrue(torch.isfinite(loss),j['name'])
            for p in model.parameters():
                if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),j['name'])
            self.assertGreater(model.item_embedding.weight.grad.abs().sum(),0)
            model.eval();u,i=model.forward()
            torch.testing.assert_close(model.full_sort_predict(torch.arange(3)),u@i.T)
            model.train();self.assertIsNone(model._eval_embeddings)

    def test_neighbor_building_does_not_change_training_rng(self):
        torch.manual_seed(10);a=RelationRec(dict(self.config,ii_weight=0.,uu_weight=0.),self.loader)
        rng=torch.get_rng_state().clone()
        torch.manual_seed(10);b=RelationRec(self.config,self.loader)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q)

    def test_zero_layer_score_and_graph_polynomial(self):
        for layers in (0,2):
            m=RelationRec(dict(self.config,ui_layers=layers),self.loader)
            x=torch.cat((m.user_embedding.weight,m.item_embedding.weight))
            if layers:
                b=self.r.toarray();adj=np.block([[np.zeros((3,3)),b],[b.T,np.zeros((24,24))]])
                d=np.maximum(adj.sum(1),1)**-.5
                a=torch.from_numpy(d[:,None]*adj*d[None,:]).float()
                expected=(x+a@x+a@a@x)/3
            else:expected=x
            actual=torch.cat(m.forward())
            torch.testing.assert_close(actual,expected)

    def test_server_all_twelve_no_save(self):
        root=Path(__file__).resolve().parents[1];workspace=Path(self.folder.name)
        data=workspace/'data'/'baby';data.mkdir(parents=True)
        for name in ('image_feat.npy','text_feat.npy'):shutil.copy(workspace/'tiny'/name,data/name)
        lines=['userID\titemID\tx_label']
        for u in range(3):
            lines.extend(f'{u}\t{i}\t0' for i in range(u*8,(u+1)*8))
            lines.extend([f'{u}\t{((u+1)*8)%24}\t1',f'{u}\t{((u+1)*8+1)%24}\t2'])
        (data/'baby.inter').write_text('\n'.join(lines)+'\n')
        output=workspace/'relation'
        result=subprocess.run([sys.executable,str(root/'experiments/run_night.py'),'--suite','relation',
            '--data-path',str(data.parent),'--cpu','--epochs','1','--hours','.2','--output',str(output)],
            capture_output=True,text=True,timeout=240)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        rows=json.loads((output/'summary.json').read_text())
        self.assertEqual(len(rows),12)
        self.assertTrue(all(r['state']=='complete' for r in rows),rows)
        self.assertFalse(list(output.rglob('*.pth')))

    def test_model_and_loss_equivariant_with_entity_state(self):
        model=RelationRec(dict(self.config,ui_layers=2,link_mix=.5),self.loader)
        changed=copy.deepcopy(model)
        up=torch.tensor([2,0,1]);ip=torch.randperm(24)
        ui,ii=torch.argsort(up),torch.argsort(ip)
        with torch.no_grad():
            changed.user_embedding.weight.copy_(model.user_embedding.weight[up])
            for name in ('item_embedding','image_embedding','text_embedding'):
                getattr(changed,name).weight.copy_(getattr(model,name).weight[ip])
            for prefix in ('visual_nn','text_nn'):
                getattr(changed,prefix+'_ids').copy_(ii[getattr(model,prefix+'_ids')[ip]])
                getattr(changed,prefix+'_weights').copy_(getattr(model,prefix+'_weights')[ip])
            changed.user_nn_ids.copy_(ui[model.user_nn_ids[up]])
            changed.user_nn_weights.copy_(model.user_nn_weights[up])
            keys=model.train_positive_keys
            new_keys=ui[keys//24]*24+ii[keys%24]
            order=torch.argsort(new_keys)
            changed.train_positive_keys=new_keys[order]
            changed.link_weights=model.link_weights[order]
            node_inverse=torch.argsort(torch.cat((up,ip+3)))
            changed.ui_adj=torch.sparse_coo_tensor(node_inverse[model.ui_adj.indices()],model.ui_adj.values(),(27,27)).coalesce()
        model.eval();changed.eval()
        scores=model.full_sort_predict(torch.arange(3))
        torch.testing.assert_close(changed.full_sort_predict(torch.arange(3)),scores[up][:,ip])
        negatives=torch.tensor([[7,8,9],[10,11,12],[13,14,15]])
        model.sample_negatives=lambda users:negatives
        changed.sample_negatives=lambda users:ii[negatives]
        batch=torch.tensor([[0,1,2],[0,2,4]])
        mapped=torch.stack((ui[batch[0]],ii[batch[1]]))
        torch.testing.assert_close(changed.calculate_loss(mapped),model.calculate_loss(batch),atol=2e-5,rtol=1e-5)
