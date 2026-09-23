"""Pinned-source integrity, graph equivalence, and a complete official CPU run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class REARMOfficialTest(unittest.TestCase):
    def test_pinned_sources_unchanged(self):
        root = ROOT / 'third_party/rearm'
        manifest = json.loads((root / 'UPSTREAM.json').read_text())
        for name, expected in manifest['files'].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected, name)

    def test_graph_equivalence(self):
        code = r'''
import sys, copy
sys.path.insert(0, 'third_party/rearm')
sys.path.insert(0, 'experiments')
import torch, numpy as np
from types import SimpleNamespace
from utils import helper
from rearm_runtime import install
# Exact shared-set counts, same full-width topk tie handling, zero diagonal.
old_counts, old_dict, old_knn = helper.creat_co_occur_matrix, helper.creat_dict_graph, helper.get_knn_adj_mat
edges=np.array([[0,5],[0,6],[1,5],[1,7],[2,6],[2,7],[3,8],[4,9]])
expected=[old_dict(old_counts(kind,edges,start,5),5) for kind,start in [('user',0),('item',5)]]
torch.manual_seed(7)
x=torch.randn(9,6)
expected_knn=old_knn(x,3,'cpu').to_dense()
install(helper, block_size=3)
for idx,(kind,start) in enumerate([('user',0),('item',5)]):
 actual=helper.creat_dict_graph(helper.creat_co_occur_matrix(kind,edges,start,5),5)
 assert actual==expected[idx],(actual,expected[idx])
torch.testing.assert_close(helper.get_knn_adj_mat(x,3,'cpu').to_dense(),expected_knn)
import scipy.sparse as sp
r=sp.coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1]-5)),shape=(5,5))
a=np.block([[np.zeros((5,5)),r.toarray()],[r.toarray().T,np.zeros((5,5))]])
d=(a.sum(1)+1e-7)**-.5
want=torch.tensor(d[:,None]*a*d[None,:]).float()
got=helper.get_norm_adj_mat(SimpleNamespace(n_users=5,n_items=5,device='cpu'),r)
torch.testing.assert_close(got.to_dense(),want)
# Official dictionary mutation includes held-out positives. Train mode is explicit.
from utils import data_loader
from rearm_runtime import patch_loader
for scope in ('official','train'):
 if scope=='train': patch_loader(data_loader,scope)
 ds=object.__new__(data_loader.BaseDataset)
 ds.n_users=2
 ds.train_data=np.array([[0,2],[1,3]])
 ds.valid_data=np.array([[0,4]])
 ds.test_data=np.array([[0,5]])
 ds.dict_user_items()
 assert ds.user_items_dict[0] == ({2,4,5} if scope=='official' else {2})
print('graph and protocol checks passed')
'''
        run = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_official_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / 'data/baby'; data.mkdir(parents=True)
            rng = np.random.default_rng(123)
            with (data / 'baby.inter').open('w') as f:
                f.write('userID\titemID\tx_label\n')
                for u in range(48):
                    for offset in range(6):
                        f.write(f'{u}\t{(u+offset)%24}\t{0 if offset<4 else offset-3}\n')
            np.save(data / 'image_feat.npy', rng.normal(size=(24,12)).astype('float32'))
            np.save(data / 'text_feat.npy', rng.normal(size=(24,8)).astype('float32'))
            output = root / 'run'
            command = [sys.executable, str(ROOT / 'experiments/run_rearm.py'),
                       '--data-path', str(data.parent), '--output', str(output),
                       '--cpu', '--epochs', '2', '--num-workers', '0', '--graph-block', '13']
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=100,
                                 env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            status = json.loads((output / 'status.json').read_text())
            result = json.loads((output / 'result.json').read_text())
            self.assertEqual(status['state'], 'complete')
            self.assertIn('Recall@20', result['test'])
            self.assertIn(result['best_epoch'], (0, 1))
            self.assertTrue((output / 'summary.csv').exists())
            self.assertFalse(list(output.rglob('*.pt')) + list(output.rglob('*.pth')))
            again = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(again.returncode,0)
            self.assertIn('Output already used',again.stderr)


if __name__ == '__main__':
    unittest.main()
