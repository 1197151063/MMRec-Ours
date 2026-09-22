import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml
import test_simmrec as fixtures
from models.corrrec import CorrRec, correlated_normal, ppr_items
from models.lightmrecnope import LightMRecNoPE
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from corr_plan import build_plan


class CorrTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        config=yaml.safe_load((Path(__file__).resolve().parents[1]/'src/configs/model/CorrRec.yaml').read_text())
        self.config.update({k:v[0] if isinstance(v,list) else v for k,v in config.items() if k!='hyper_parameters'})
        self.config['embedding_size']=6
        self.loader.dataset_bk=SimpleNamespace(df=pd.DataFrame({'userID':self.rows,'itemID':self.cols}))

    def test_stationary_covariance_and_independent_rng(self):
        before=torch.get_rng_state().clone()
        x=correlated_normal(40000,16,.75,.9,.3,123).double()
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertAlmostEqual(x.square().mean().item(),.09,delta=.004)
        self.assertAlmostEqual((x[1:]*x[:-1]).mean().item(),.09*.75*.9,delta=.004)
        a=correlated_normal(20,8,0.,.9,.3,123)
        b=correlated_normal(20,8,0.,.5,.3,123)
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_disabled_components_match_legacy(self):
        torch.manual_seed(55); a=LightMRecNoPE(self.config,self.loader)
        torch.manual_seed(55); b=CorrRec(dict(self.config,corr_init=False),self.loader)
        batch=torch.tensor([[0,1,2],[0,2,4]])
        torch.manual_seed(88); la=a.calculate_loss(batch)
        torch.manual_seed(88); lb=b.calculate_loss(batch)
        torch.testing.assert_close(la,lb,atol=0,rtol=0)
        la.backward();lb.backward()
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)

    def test_ppr_blocks_match_dense_reference(self):
        b=sp.csr_matrix(np.array([[1,1,0,0],[0,1,1,0],[1,0,1,1]],dtype=np.float32))
        p=np.diag(1/np.asarray(b.sum(0)).ravel())@b.T.toarray()@np.diag(1/np.asarray(b.sum(1)).ravel())@b.toarray()
        np.fill_diagonal(p,0)
        x=.85*np.eye(4)
        for _ in range(4):x=.15*p@x+.85*np.eye(4)
        np.fill_diagonal(x,0)
        actual=ppr_items(b,3,iterations=4,block=2).toarray()
        np.testing.assert_allclose(actual,x,rtol=1e-5,atol=1e-7)

    def test_all_component_configs_train(self):
        jobs=build_plan()
        self.assertEqual(len({j['name'] for j in jobs}),len(jobs))
        batch=torch.tensor([[0,1,2],[0,2,4]])
        for job in jobs[:36]:
            model=CorrRec(dict(self.config,**job['overrides']),self.loader)
            loss=model.calculate_loss(batch)
            self.assertTrue(torch.isfinite(loss),job['name'])
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),job['name'])
            model.eval()
            self.assertEqual(model.full_sort_predict(torch.arange(3)).shape,(3,24))

    def test_random_order_only_reassigns_correlated_vectors(self):
        torch.manual_seed(123)
        a=CorrRec(self.config,self.loader)
        torch.manual_seed(123)
        b=CorrRec(dict(self.config,order_source='random_user'),self.loader)
        from models.lightmrecorder import make_positions
        positions,_=make_positions(self.loader.dataset_bk.df,'userID','itemID',3,24,'random_user',2026)
        torch.testing.assert_close(b.user_embedding.weight,a.user_embedding.weight[positions])
        torch.testing.assert_close(b.image_trs.mlp[0].weight,a.image_trs.mlp[0].weight)

    def test_ui_graph_uses_binary_train_edges(self):
        model=CorrRec(dict(self.config,ui_layers=1),self.loader)
        expected=torch.zeros(27,27)
        for u,i in zip(self.rows,self.cols):
            expected[u,3+i]=expected[3+i,u]=1/(2**.5)
        torch.testing.assert_close(model.ui_adj.to_dense(),expected)
        self.assertEqual(model.ui_adj._nnz(),12)
