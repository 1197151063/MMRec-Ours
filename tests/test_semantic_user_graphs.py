import sys
import unittest
from pathlib import Path
import numpy as np
import scipy.sparse as sp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from build_semantic_user_graphs import normalize,anchor_history,item_transition,select_graph

class SemanticGraphTest(unittest.TestCase):
    def test_leave_one_out_bridge_and_permutation(self):
        r=sp.csr_matrix([[1,1,0,0],[1,0,1,0],[0,0,0,1]],dtype=np.float32)
        f=normalize(np.array([[1.,0],[1,0],[0,1],[0,1]],dtype=np.float32))
        a=anchor_history(r,f)
        self.assertAlmostEqual(a[0,0],1)
        self.assertAlmostEqual(a[1,0],.5)
        self.assertAlmostEqual(a[2,3],.5) # no other interactions: neutral
        ties=np.array([.8,.1,.3]);it=np.array([.3,.7,.1,.5])
        s=item_transition(f,1,2,it)
        np.testing.assert_allclose(np.asarray(s.sum(1)).ravel(),1)
        u=np.array([2,0,1]);i=np.array([3,1,0,2])
        for x,y in ((r,r[u][:,i]),(a,anchor_history(r[u][:,i],f[i])),
                    ((r@s).tocsr(),(r[u][:,i]@item_transition(f[i],1,2,it[i])).tocsr())):
            for overlap in (False,):
                g=select_graph(x,2,2,ties,overlap).toarray()
                gp=select_graph(y,2,2,ties[u],overlap).toarray()
                np.testing.assert_allclose(gp,g[u][:,u],atol=1e-6)
                self.assertTrue(np.all(np.diag(g)==0))
        bridge=select_graph((r@s).tocsr(),2,2,ties).toarray()
        self.assertGreater(bridge[1,2],0) # shared semantics without shared item

if __name__=='__main__':unittest.main()
