import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from research_user_connectivity import paths_from_anchors,rewire,paired_controls,evaluate

class ConnectivityTest(unittest.TestCase):
    def test_boolean_paths_against_shortest_path(self):
        r=sp.csr_matrix([[1,0,0],[1,1,0],[0,0,1]],dtype=bool)
        ii=sp.csr_matrix([[0,0,0],[0,0,1],[0,1,0]],dtype=bool)
        a=sp.bmat([[sp.csr_matrix((3,3),dtype=bool),r],[r.T,ii]],format='csr',dtype=bool)
        distances=shortest_path(a,directed=False)
        for start,hop,reached in paths_from_anchors(a,np.arange(3),3,block=2):
            expected=distances[start:start+reached.shape[0],:3]<=hop
            expected[np.arange(reached.shape[0]),np.arange(start,start+reached.shape[0])]=False
            np.testing.assert_array_equal(reached.toarray(),expected)
        pairs=paired_controls(np.array([1,2,1]),np.arange(3),2026)
        base=evaluate(r,sp.csr_matrix((3,3),dtype=bool),np.arange(3),pairs,2)
        expanded=evaluate(r,ii,np.arange(3),pairs,2)
        self.assertEqual(base['hops']['2'],expanded['hops']['2']|{'pairs':base['hops']['2']['pairs']})
        self.assertGreater(expanded['hops']['3']['mean_reachable_other_users'],base['hops']['3']['mean_reachable_other_users'])

    def test_rewire_degrees(self):
        rng=np.random.default_rng(5);dense=rng.random((20,20))<.2
        dense=np.triu(dense,1);dense=dense|dense.T
        g=sp.csr_matrix(dense);h,info=rewire(g,5)
        np.testing.assert_array_equal(np.diff(g.indptr),np.diff(h.indptr))
        self.assertEqual((h!=h.T).nnz,0)
        self.assertEqual(h.diagonal().sum(),0)
        self.assertGreater(info['successful_swaps'],0)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'data/baby';d.mkdir(parents=True)
            (d/'baby.inter').write_text('userID\titemID\tx_label\n'+''.join(f'{u}\t{(u+j)%12}\t0\n' for u in range(8) for j in range(2)))
            for name in ('image_feat.npy','text_feat.npy'):np.save(d/name,np.random.default_rng(8).normal(size=(12,6)).astype(np.float32))
            result=subprocess.run([sys.executable,str(ROOT/'experiments/research_user_connectivity.py'),
                '--data-path',str(d.parent),'--output',str(root/'out'),'--anchors','6','--item-k','2',
                '--null-repeats','1'],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report=json.loads((root/'out/report.json').read_text())
            self.assertEqual(len(report['variants']),3)
            self.assertEqual(json.loads((root/'out/status.json').read_text())['state'],'complete')

if __name__=='__main__':unittest.main()
