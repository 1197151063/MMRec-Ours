import sys
from pathlib import Path
import unittest

import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from audit_order_signal_coverage import audit


class SignalCoverageTest(unittest.TestCase):
    def test_distinguishes_order_hit_with_and_without_train_path(self):
        valid = sp.csr_matrix(([1.], ([0], [1])), shape=(3, 22))
        separate = sp.csr_matrix(([1., 1., 1.], ([0, 1, 2], [0, 1, 2])), shape=(3, 22))
        out = audit(separate, valid, bandwidth=1., radius=1)
        self.assertEqual(out['counts']['order_hit_edges'], 1)
        self.assertEqual(out['counts']['order_hit_no_threehop_edges'], 1)
        bridge = sp.csr_matrix(([1., 1., 1., 1.], ([0, 1, 2, 2], [0, 1, 0, 1])), shape=(3, 22))
        out = audit(bridge, valid, bandwidth=1., radius=1)
        self.assertEqual(out['counts']['order_hit_reachable_edges'], 1)
        self.assertEqual(out['counts']['order_hit_no_threehop_edges'], 0)


if __name__ == '__main__':
    unittest.main()
