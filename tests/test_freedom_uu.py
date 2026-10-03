import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import yaml
import test_simmrec as fixtures
from models.freedomalign import FreedomAlign
from models.freedomalignuu import FreedomAlignUU
from common.user_walks import UserWalks,RowSampler
ROOT=Path(__file__).resolve().parents[1]

class UserWalkTest(unittest.TestCase):
    def test_distributions_and_masked_loss(self):
        r=sp.csr_matrix([[1,1,0],[1,0,0],[0,0,1]],dtype=np.float32)
        s=sp.csr_matrix([[0,2,1],[2,0,0],[1,0,0]],dtype=np.float32)
        walk=UserWalks(r,s,seed=13)
        ui=r.toarray()/np.asarray(r.sum(1));iu=r.T.toarray()/np.asarray(r.sum(0)).T
        sem=s.toarray()/np.asarray(s.sum(1))
        for bridge,expected in [(False,ui@iu),(True,ui@sem@iu)]:
            expected=expected.copy();np.fill_diagonal(expected,0)
            neighbors,valid=walk.sample(np.arange(3),60000,bridge)
            actual=np.array([[np.mean((neighbors[u]==v)&valid[u]) for v in range(3)] for u in range(3)])
            np.testing.assert_allclose(actual,expected,atol=.006)
        e=torch.tensor([[1.,0.],[0.,2.],[1.,1.]],requires_grad=True)
        users=torch.tensor([0,1]);neighbors=torch.tensor([[0,1],[0,2]]);valid=torch.tensor([[False,True],[True,True]])
        value=FreedomAlignUU.sampled_constraint(e,users,neighbors,valid)
        target=(2*torch.nn.functional.softplus(torch.tensor(0.))+torch.nn.functional.softplus(torch.tensor(-2.)))/4
        torch.testing.assert_close(value,target);value.backward();self.assertTrue(torch.isfinite(e.grad).all())
        empty=RowSampler(sp.csr_matrix((3,3)),True)
        _,valid=empty.draw(np.array([0,2]),np.random.default_rng(1));self.assertFalse(valid.any())

    def test_streams_separate_and_no_global_rng(self):
        r=sp.csr_matrix([[1,1],[1,0],[0,1]],dtype=np.float32);s=sp.csr_matrix([[0,1],[1,0]])
        torch_state=torch.get_rng_state().clone();np_state=np.random.get_state()
        a=UserWalks(r,s,seed=2);b=UserWalks(r,s,seed=2)
        a.sample(np.array([0,1]),8,True)
        for x,y in zip(a.sample(np.array([0,1]),8),b.sample(np.array([0,1]),8)):np.testing.assert_array_equal(x,y)
        self.assertTrue(torch.equal(torch.get_rng_state(),torch_state))
        np.testing.assert_array_equal(np.random.get_state()[1],np_state[1])

class ModelUUTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        options=yaml.safe_load((ROOT/'src/configs/model/FreedomAlignUU.yaml').read_text());options.pop('hyper_parameters')
        self.config.update(options)

    def test_zero_weight_equivalence_and_gradients(self):
        batch=torch.tensor([[0,1,2],[0,2,4],[9,10,11]])
        torch.manual_seed(3);a=FreedomAlign(self.config,self.loader)
        torch.manual_seed(3);b=FreedomAlignUU(dict(self.config,uu_weight=0.),self.loader)
        torch.testing.assert_close(a.calculate_loss(batch),b.calculate_loss(batch))
        for beta in (0.,.1):
            torch.manual_seed(3);m=FreedomAlignUU(dict(self.config,uu_bridge_weight=beta),self.loader)
            for p,q in zip(a.parameters(),m.parameters()):torch.testing.assert_close(p,q)
            loss=m.calculate_loss(batch);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(m.user_embedding.weight.grad).all())
            m.eval();u,i=m.forward();torch.testing.assert_close(m.full_sort_predict([batch[0]]),u[batch[0]]@i.T)

    def test_seven_run_queue(self):
        workspace=Path(self.folder.name);data=workspace/'data/baby';data.mkdir(parents=True)
        for name in ('image_feat.npy','text_feat.npy'):shutil.copy(workspace/'tiny'/name,data/name)
        lines=['userID\titemID\tx_label']
        for u in range(3):
            lines.extend(f'{u}\t{i}\t0' for i in range(u*8,(u+1)*8))
            lines.extend([f'{u}\t{((u+1)*8)%24}\t1',f'{u}\t{((u+1)*8+1)%24}\t2'])
        (data/'baby.inter').write_text('\n'.join(lines)+'\n')
        output=workspace/'runs'
        run=subprocess.run([sys.executable,str(ROOT/'experiments/run_night.py'),'--suite','freedom_uu',
            '--data-path',str(data.parent),'--cpu','--epochs','1','--hours','.2','--output',str(output)],
            capture_output=True,text=True,timeout=240)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        rows=json.loads((output/'summary.json').read_text())
        self.assertEqual(len(rows),7);self.assertTrue(all(x['state']=='complete' for x in rows),rows)
        self.assertFalse(list(output.rglob('*.pt'))+list(output.rglob('*.pth')))

if __name__=='__main__':unittest.main()
