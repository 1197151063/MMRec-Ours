"""Focused CPU checks for the FREEDOM soft-cohort intervention."""

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

import test_simmrec as fixtures

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common.soft_item_cohorts import build_membership, SoftItemCohort
from models.freedom import FREEDOM
from models.freedomsoftcohort import FreedomSoftCohort


class SoftCohortTest(unittest.TestCase):
    def test_membership_uses_train_rows_without_user_coordinates(self):
        rng = np.random.default_rng(3)
        r = sp.csr_matrix((rng.random((7, 15)) < .3).astype(np.float32))
        text = rng.normal(size=(15, 8)).astype(np.float32)
        perm = rng.permutation(7)
        original = build_membership(r, text, kind='text', count=4, seed=9)
        shuffled = build_membership(r[perm], text, kind='text', count=4, seed=9)
        np.testing.assert_allclose(original[perm], shuffled, atol=1e-6)
        degree = np.diff(r.indptr)
        np.testing.assert_allclose(original[degree > 0].sum(1), 1, atol=1e-6)

    def test_diffusion_excludes_self_and_preserves_isolated_user(self):
        p = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=np.float32)
        module = SoftItemCohort(p)
        users = torch.tensor([[1.], [3.], [8.], [11.]], requires_grad=True)
        peers = module(users)
        torch.testing.assert_close(peers[:, 0], torch.tensor([3., 1., 8., 11.]))
        peers.sum().backward()
        self.assertTrue(torch.isfinite(users.grad).all())


class FreedomSoftCohortTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        self.config.update(feat_embed_dim=6, knn_k=4, lambda_coeff=.9,
                           cf_model='lightgcn', n_mm_layers=1, n_ui_layers=2,
                           mm_image_weight=.1, dropout=0., degree_ratio=0.,
                           cohort_kind='joint', cohort_beta=0., cohort_count=4,
                           cohort_seed=2026)

    def test_zero_weight_is_exact_freedom_and_nonzero_gradients(self):
        torch.manual_seed(12)
        base = FREEDOM(self.config, self.loader)
        torch.manual_seed(12)
        control = FreedomSoftCohort(self.config, self.loader)
        base.pre_epoch_processing(); control.pre_epoch_processing()
        bu, bi = base.forward(base.norm_adj)
        cu, ci = control.forward(control.norm_adj)
        torch.testing.assert_close((bu, bi), (cu, ci), rtol=0, atol=0)
        self.assertIsNone(control.cohort)

        treatment = FreedomSoftCohort(dict(self.config, cohort_beta=.2), self.loader)
        treatment.pre_epoch_processing()
        users, items = treatment.forward(treatment.norm_adj)
        self.assertEqual((users.shape, items.shape), ((3, 6), (24, 6)))
        loss = treatment.calculate_loss(torch.tensor([[0, 1], [0, 2], [3, 4]]))[0]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(treatment.user_embedding.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
