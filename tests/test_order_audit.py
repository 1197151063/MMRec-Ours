import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import test_simmrec as fixtures
from models.lightmrecnope import LightMRecNoPE
from models.lightmrecorder import LightMRecOrder, make_positions, sinusoidal


class OrderAuditTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        self.train = pd.DataFrame({'userID': self.rows, 'itemID': self.cols})
        self.loader.dataset_bk = SimpleNamespace(df=self.train)
        self.config.update(order_source='original', order_seed=2026, pe_side='none')

    def test_sides_match_original_algebra_and_no_pe_control(self):
        torch.manual_seed(42)
        reference = LightMRecNoPE(self.config, self.loader).eval()
        base_user, base_item = reference.forward()
        for side in ['none', 'user', 'item', 'both']:
            torch.manual_seed(42)
            model = LightMRecOrder(dict(self.config, pe_side=side), self.loader).eval()
            user, item = model.forward()
            expected_user = base_user + model.user_pe if side in ('user', 'both') else base_user
            expected_item = base_item + model.item_pe if side in ('item', 'both') else base_item
            torch.testing.assert_close(user, expected_user)
            torch.testing.assert_close(item, expected_item)
            # Adding the same PE before convex fusion is identical to adding it after fusion.
            if side == 'both':
                visual = model.image_trs(model.image_embedding.weight)
                textual = model.text_trs(model.text_embedding.weight)
                torch.testing.assert_close(item, model.alpha * (textual + model.item_pe) +
                                           (1-model.alpha) * (visual + model.item_pe))

    def test_maps_are_bijections_reproducible_and_rng_independent(self):
        for source in ['original', 'random', 'train_file', 'train_shuffled']:
            torch.manual_seed(43)
            torch_state = torch.random.get_rng_state().clone()
            np.random.seed(44)
            numpy_state = np.random.get_state()
            maps = make_positions(self.train, 'userID', 'itemID', 3, 24, source, 2026)
            repeated = make_positions(self.train, 'userID', 'itemID', 3, 24, source, 2026)
            for actual, again, size in zip(maps, repeated, [3, 24]):
                np.testing.assert_array_equal(np.sort(actual), np.arange(size))
                np.testing.assert_array_equal(actual, again)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])

    def test_train_first_ignores_heldout_rows_and_orders_unseen_last(self):
        full = pd.DataFrame({'userID': [2, 1, 0, 2, 1],
                             'itemID': [20, 4, 2, 6, 9], 'split': [2, 0, 0, 1, 0]})
        train = full.loc[full['split'] == 0]
        users, items = make_positions(train, 'userID', 'itemID', 3, 24, 'train_file', 2026)
        self.assertEqual(users[1], 0)
        self.assertEqual(users[0], 1)
        np.testing.assert_array_equal(items[[4, 2, 9]], [0, 1, 2])
        self.assertGreaterEqual(items[20], 3)
        changed = pd.concat([full.iloc[::-1].loc[full['split'] != 0], train], ignore_index=True)
        again = make_positions(changed.loc[changed['split'] == 0], 'userID', 'itemID', 3, 24, 'train_file', 2026)
        np.testing.assert_array_equal(users, again[0])
        np.testing.assert_array_equal(items, again[1])

    def test_sinusoidal_has_expected_constant_norm(self):
        pe = sinusoidal(np.arange(24), 64)
        torch.testing.assert_close(pe.square().sum(-1), torch.full((24,), 32.0))
