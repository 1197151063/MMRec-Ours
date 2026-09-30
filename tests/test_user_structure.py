import sys
from pathlib import Path
import unittest
import numpy as np
import scipy.sparse as sp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from diagnose_user_structure import build_graph,diagnose

class StructureTest(unittest.TestCase):
    def test_scores_and_no_id_distance(self):
        r=sp.csr_matrix([[1,1,0],[1,0,0],[0,0,1],[1,1,0]],dtype=np.float32)
        for metric in ('cosine','jaccard','resource_cosine'):
            g=build_graph(r,metric,3,np.random.default_rng(1)).toarray()
            self.assertTrue(np.all(np.diag(g)==0))
            self.assertEqual(g[0,2],0)
            self.assertAlmostEqual(g[0,3],1,places=5)
            order=np.array([2,0,3,1])
            remapped=build_graph(r[order],metric,3,np.random.default_rng(1)).toarray()
            np.testing.assert_allclose(remapped,g[order][:,order],atol=1e-6)
        result=diagnose(r,[1],10,3,2,5,2)
        self.assertEqual(result['training_pairs'],6)
        self.assertEqual(result['graphs']['cosine']['edges'],6)

if __name__=='__main__':unittest.main()
