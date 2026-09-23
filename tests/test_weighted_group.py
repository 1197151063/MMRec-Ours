import unittest
import numpy as np
import scipy.sparse as sp
import torch
import test_group as fixtures
from models.grouprec import GroupRec
from models.weightedgrouprec import WeightedGroupRec,positive_support


class WeightedTest(unittest.TestCase):
    def setUp(self):
        fixtures.GroupTest.setUp(self)
        self.config.update(wbpr_mix=.5,link_beta=.5,link_gamma=.5,link_delta=.5)
        self.matrix=sp.coo_matrix((np.ones(8),([0,0,1,1,1,1,2,2],[0,1,0,1,2,3,4,5])),shape=(3,24))
        self.loader.inter_matrix=lambda form:self.matrix

    def test_support_matches_enumerated_nonbacktracking_paths(self):
        r=self.matrix.toarray();du=r.sum(1).clip(1);di=r.sum(0).clip(1)
        for exponents in ((0.,0.,0.),(.5,.7,.2)):
            b,g,d=exponents
            keys,actual=positive_support(self.matrix,b,g,d,block=1)
            expected=[]
            for key in keys:
                u,i=divmod(int(key),24);score=0.
                for ix in range(24):
                    for v in range(3):
                        if ix!=i and v!=u and r[u,ix] and r[v,ix] and r[v,i]:
                            score+=di[ix]**-b*du[v]**-g*di[i]**-d
                expected.append(score)
            np.testing.assert_allclose(actual,expected,atol=1e-12)
        # Duplicate interactions do not alter binary support.
        duplicate=sp.coo_matrix((np.r_[self.matrix.data,1.],
                                (np.r_[self.matrix.row,0],np.r_[self.matrix.col,0])),shape=(3,24))
        np.testing.assert_allclose(positive_support(duplicate)[1],positive_support(self.matrix)[1])

    def test_zero_weight_recovers_baseline_and_ssm_is_unchanged(self):
        batch=torch.tensor([[0,1,2],[0,2,4]])
        torch.manual_seed(42);base=GroupRec(self.config,self.loader)
        state=torch.get_rng_state().clone()
        torch.manual_seed(42);zero=WeightedGroupRec(dict(self.config,wbpr_mix=0.),self.loader)
        torch.manual_seed(42);weighted=WeightedGroupRec(self.config,self.loader)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertFalse(weighted.support_weights.requires_grad)
        self.assertAlmostEqual(weighted.support_weights.mean().item(),1.,places=6)
        self.assertGreater(weighted.support_weights.max()-weighted.support_weights.min(),0)
        torch.manual_seed(99);a=base.calculate_loss(batch)
        torch.manual_seed(99);b=zero.calculate_loss(batch)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        a.backward();b.backward()
        for p,q in zip(base.parameters(),zero.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
        torch.manual_seed(99);c=weighted.calculate_loss(batch)
        torch.manual_seed(99);neg=torch.randint(24,(3,32))
        user,item=weighted.forward();u=user[batch[0]]
        pos=(u*item[batch[1]]).sum(-1);negative=(u[:,None]*item[neg]).sum(-1)
        plain=base.bpr_objective(batch[0],batch[1],pos,negative)
        weight=weighted.bpr_objective(batch[0],batch[1],pos,negative)
        torch.testing.assert_close(c-a,weight-plain,atol=2e-6,rtol=1e-4)
        c.backward()
        for p in weighted.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
        torch.testing.assert_close(weighted.full_sort_predict(torch.arange(3)),base.full_sort_predict(torch.arange(3)))

    def test_server_entrypoint(self):
        self.suite='wbpr'
        fixtures.GroupTest.test_server_queue_runs_all_configs_without_saving(self)
