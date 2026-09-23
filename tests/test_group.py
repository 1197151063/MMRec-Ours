import sys
import unittest
import json
import shutil
import subprocess
from pathlib import Path
import torch
import test_simmrec as fixtures
from models.grouprec import GroupRec, user_groups
from models.lightmrecnope import LightMRecNoPE
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from group_plan import build_plan


class GroupTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        self.config.update(embedding_size=64,group_size=2,group_strength=1.,group_mode='contiguous',group_init='normal',
                           group_std=.125,group_trainable=True,use_personal=True,group_seed=2026)

    def test_group_sizes_and_rng(self):
        rng=torch.get_rng_state().clone()
        a=user_groups(11,3,'contiguous',20)
        b=user_groups(11,3,'random',20)
        torch.testing.assert_close(torch.bincount(a),torch.tensor([3,3,3,2]))
        torch.testing.assert_close(torch.bincount(a),torch.bincount(b))
        torch.testing.assert_close(b,user_groups(11,3,'random',20))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_three_losses_and_id_inference(self):
        batch=torch.tensor([[0,1,2],[0,2,4]])
        for alpha in (0.,.25,1.):
            model=GroupRec(dict(self.config,alpha=alpha,group_strength=0.),self.loader)
            torch.manual_seed(99); actual=model.calculate_loss(batch)
            torch.manual_seed(99)
            negatives=torch.randint(24,(3,32))
            ids=torch.cat((batch[1,:,None],negatives),dim=1)
            users=torch.nn.functional.normalize(model.user_embedding(batch[0]),dim=-1)
            def reference(table):
                logits=(torch.nn.functional.normalize(table[ids],dim=-1)*users[:,None]).sum(-1)/model.temperature
                return (torch.logsumexp(logits[:,1:],dim=-1)-logits[:,0]).mean()
            expected=reference(model.item_embedding.weight)
            expected=expected+alpha*reference(model.image_trs(model.image_embedding.weight))
            expected=expected+(1-alpha)*reference(model.text_trs(model.text_embedding.weight))
            torch.testing.assert_close(actual,expected)
            actual.backward()
            self.assertGreater(model.item_embedding.weight.grad.abs().sum(),0)
            self.assertIsNone(model.group_embedding.weight.grad)
            if alpha>0:self.assertGreater(model.image_trs.weight.grad.abs().sum(),0)
            if alpha<1:self.assertGreater(model.text_trs.weight.grad.abs().sum(),0)
            model.eval()
            expected_scores=model.user_embedding.weight@model.item_embedding.weight.T
            torch.testing.assert_close(model.full_sort_predict(torch.arange(3)),expected_scores)

    def test_group_gradient_is_sum_of_member_gradients(self):
        model=GroupRec(dict(self.config,group_init='zero'),self.loader).eval()
        user,_=model.forward()
        coefficients=torch.arange(192).reshape(3,64).float()
        (user*coefficients).sum().backward()
        torch.testing.assert_close(model.group_embedding.weight.grad[0],coefficients[:2].sum(0))
        torch.testing.assert_close(model.group_embedding.weight.grad[1],coefficients[2])

    def test_variants_and_plan(self):
        model=GroupRec(self.config,self.loader)
        self.assertIsInstance(model.image_trs,torch.nn.Linear)
        self.assertIsInstance(model.text_trs,torch.nn.Linear)
        self.assertEqual(model.image_trs.out_features,64)
        self.assertEqual(model.text_trs.out_features,64)
        self.assertFalse(any(isinstance(m,(torch.nn.BatchNorm1d,torch.nn.Dropout)) for m in model.modules()))
        plan=build_plan()
        self.assertEqual(len(plan),5)
        self.assertEqual(len({j['name'] for j in plan}),5)
        self.assertTrue(all(not any(k.startswith('group_init') or k == 'group_std' for k in j['overrides']) for j in plan))
        batch=torch.tensor([[0,1,2],[0,2,4]])
        for j in plan:
            if j['model']!='GroupRec':continue
            model=GroupRec(dict(self.config,**j['overrides']),self.loader)
            loss=model.calculate_loss(batch);loss.backward()
            self.assertTrue(torch.isfinite(loss),j['name'])
            for p in model.parameters():
                if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),j['name'])

    def test_server_queue_runs_all_configs_without_saving(self):
        root=Path(__file__).resolve().parents[1]
        workspace=Path(self.folder.name)
        data=workspace/'data'/'baby';data.mkdir(parents=True)
        for name in ('image_feat.npy','text_feat.npy'):
            shutil.copy(workspace/'tiny'/name,data/name)
        lines=['userID\titemID\tx_label']
        for u in range(3):
            lines.extend(f'{u}\t{i}\t0' for i in range(u*8,(u+1)*8))
            lines.extend([f'{u}\t{((u+1)*8)%24}\t1',f'{u}\t{((u+1)*8+1)%24}\t2'])
        (data/'baby.inter').write_text('\n'.join(lines)+'\n')
        output=workspace/'group-queue'
        result=subprocess.run([sys.executable,str(root/'experiments/run_night.py'),'--suite','group',
            '--data-path',str(data.parent),'--cpu','--epochs','1','--hours','.2','--output',str(output)],
            capture_output=True,text=True,timeout=240)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        summary=json.loads((output/'summary.json').read_text())
        self.assertEqual(len(summary),5)
        self.assertTrue(all(r['state']=='complete' for r in summary),summary)
        self.assertFalse(list(output.rglob('*.pth')))
