import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
import test_simmrec as fixtures
from models.freedomalign import FreedomAlign

ROOT = Path(__file__).resolve().parents[1]


class FreedomAlignTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        options = yaml.safe_load((ROOT/'src/configs/model/FreedomAlign.yaml').read_text())
        options.pop('hyper_parameters')
        self.config.update(options)
        self.m = FreedomAlign(self.config, self.loader)

    def test_corrected_formula_and_gradients(self):
        batch = torch.tensor([[0,1,2],[0,2,4],[9,10,11]])
        u, i = self.m.forward()
        us, pos, neg = batch
        expected = -F.logsigmoid((u[us]*i[pos]).sum(-1)-(u[us]*i[neg]).sum(-1)).mean()
        for embedding, projector in ((self.m.image_embedding,self.m.image_trs),(self.m.text_embedding,self.m.text_trs)):
            x = F.normalize(projector(embedding.weight),dim=1)
            user = F.normalize(u[us],dim=1)
            numerator = torch.exp((user*x[pos]).sum(-1)/.1)
            denominator = torch.exp(user@x[neg].T/.1).sum(-1)
            expected += .01*(-torch.log(numerator/denominator)).mean()
        for feature, ids, mix in ((self.m.v_feat,self.m.image_knn_idx,.1),(self.m.t_feat,self.m.text_knn_idx,.9)):
            x=F.normalize(feature,dim=1);sim=x@x.T
            neighbors=ids[pos]
            # Dense reference: pairwise gather, not cross-batch broadcasting.
            weights=sim[pos[:,None],neighbors]
            emb=self.m.item_id_embedding.weight
            dots=(emb[pos,None]*emb[neighbors]).sum(-1)
            expected += .0005*mix*(-weights*F.logsigmoid(dots)).sum()
            self.assertEqual(sim[pos][:,neighbors].shape,(3,3,20))
        actual=self.m.calculate_loss(batch)
        torch.testing.assert_close(actual,expected)
        actual.backward()
        for p in self.m.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(self.m.image_trs.weight.grad.abs().sum(),0)

    def test_ui_polynomial_and_knn(self):
        e=torch.cat((self.m.user_embedding.weight,self.m.item_id_embedding.weight))
        a=self.m.norm_adj.to_dense()
        torch.testing.assert_close(torch.cat(self.m.forward()),(e+a@e+a@a@e)/3)
        x=F.normalize(self.m.v_feat,dim=-1);sim=x@x.T
        ids,w=FreedomAlign.content_knn(self.m.v_feat,20,block=5)
        torch.testing.assert_close(w,sim.gather(1,ids))
        torch.testing.assert_close(w.sort(1).values,sim.topk(20,dim=1).values.sort(1).values)
        ids,w=FreedomAlign.content_knn(self.m.v_feat,20,block=7,keep_self=False)
        self.assertFalse((ids==torch.arange(24)[:,None]).any())
        self.m.eval()
        u,i=self.m.forward()
        torch.testing.assert_close(self.m.full_sort_predict([torch.arange(3)]),u@i.T)
        self.m.train();self.assertIsNone(self.m._eval_embeddings)

    def test_sum_keeps_repeated_anchors(self):
        one=torch.tensor([0]);two=torch.tensor([0,0])
        fn=lambda x:self.m.item_alignment(x,self.m.image_knn_idx,self.m.image_knn_weight)
        torch.testing.assert_close(fn(two),2*fn(one))

    def test_four_job_queue_no_checkpoints(self):
        workspace=Path(self.folder.name);data=workspace/'data'/'baby';data.mkdir(parents=True)
        for name in ('image_feat.npy','text_feat.npy'):shutil.copy(workspace/'tiny'/name,data/name)
        lines=['userID\titemID\tx_label']
        for u in range(3):
            lines.extend(f'{u}\t{i}\t0' for i in range(u*8,(u+1)*8))
            lines.extend([f'{u}\t{((u+1)*8)%24}\t1',f'{u}\t{((u+1)*8+1)%24}\t2'])
        (data/'baby.inter').write_text('\n'.join(lines)+'\n')
        output=workspace/'runs'
        run=subprocess.run([sys.executable,str(ROOT/'experiments/run_night.py'),'--suite','freedom_align',
            '--data-path',str(data.parent),'--cpu','--epochs','1','--hours','.2','--output',str(output)],
            capture_output=True,text=True,timeout=180)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        rows=json.loads((output/'summary.json').read_text())
        self.assertEqual(len(rows),4)
        self.assertTrue(all(row['state']=='complete' for row in rows),rows)
        self.assertFalse(list(output.rglob('*.pth'))+list(output.rglob('*.pt')))


if __name__=='__main__':unittest.main()
