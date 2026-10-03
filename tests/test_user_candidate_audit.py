import sys
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
import scipy.sparse as sp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from audit_user_candidates import item_balanced,concentration,rank_info,summarize_pairs,audit
from build_semantic_user_graphs import normalize

class CandidateAuditTest(unittest.TestCase):
    def test_ranks_ties_and_denominators(self):
        ids=np.array([3,1,2]);scores=np.array([.9,.5,.5])
        self.assertIsNone(rank_info(ids,scores,0))
        x=rank_info(ids,scores,2)
        self.assertEqual(x,dict(rank=3,low=2,high=3,ties=2))
        records=[None,dict(rank=5,low=4,high=6,ties=3)]
        s=summarize_pairs(records)
        self.assertEqual(s['candidate_coverage'],.5)
        self.assertEqual(s['topk']['5']['fraction_of_all_pairs_selected'],.5)
        self.assertAlmostEqual(s['topk']['5']['expected_candidate_recall_random_ties'],2/3)

    def test_balanced_diversity_and_graph_integrity(self):
        r=sp.csr_matrix([[1,1],[1,0],[1,0],[1,0],[0,1]],dtype=np.float32)
        ranked=np.array([1,2,3,4]);ties=np.array([.2,.9])
        chosen=item_balanced(0,ranked,r,ties,2)
        np.testing.assert_array_equal(chosen,[1,4])
        d=np.asarray(r.sum(0)).ravel()
        self.assertEqual(concentration(0,ranked[:2],r,d)['max_anchor_share'],1.)
        self.assertEqual(concentration(0,chosen,r,d)['max_anchor_share'],.5)
        report,graphs=audit(r,normalize(np.array([[1.,0],[0.,1]],dtype=np.float32)),2,2)
        for name,g in graphs.items():
            self.assertTrue(np.all(g.diagonal()==0))
            self.assertTrue(np.all(np.diff(g.indptr)<=2))
            pairs=report['graphs'][name]['exact_offset_pair_recall']['1']
            direct=sum(g[u,v]>0 for u in range(5) for v in (u-1,u+1) if 0<=v<5)/8
            self.assertAlmostEqual(pairs,direct)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data/baby';data.mkdir(parents=True)
            lines=['userID\titemID\tx_label']
            for u in range(8):
                lines.extend(f'{u}\t{(u+j)%12}\t0' for j in range(3))
            (data/'baby.inter').write_text('\n'.join(lines)+'\n')
            for name in ('image_feat.npy','text_feat.npy'):
                np.save(data/name,np.random.default_rng(3).normal(size=(12,6)).astype(np.float32))
            run=subprocess.run([sys.executable,str(ROOT/'experiments/audit_user_candidates.py'),'--data-path',str(data.parent),
                                '--output',str(root/'out')],capture_output=True,text=True,timeout=30)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            report=json.loads((root/'out/report.json').read_text())
            self.assertEqual(len(report['graphs']),3)
            self.assertEqual(len(list((root/'out').glob('*.npz'))),6)
            self.assertIn('Finished:',run.stdout)

if __name__=='__main__':unittest.main()
