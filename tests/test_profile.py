import sys
import copy
import json
import unittest
from pathlib import Path
import torch
import test_simmrec as fixtures
from models.simmrec import SIMMRec
from models.simmprofile import SIMMProfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from profile_plan import build_plan


class ProfileTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        self.config.update(item_id_weight=0., profile_weight=1., profile_mode='own',
                           history_power=0., leave_one_out=True, use_user_id=True,
                           detach_profile=False, profile_seed=2026)

    def test_zero_weight_exactly_matches_baseline(self):
        torch.manual_seed(99)
        a = SIMMRec(self.config, self.loader)
        torch.manual_seed(99)
        b = SIMMProfile(dict(self.config, profile_weight=0.), self.loader)
        batch = torch.tensor([[0, 1, 2], [0, 2, 4]])
        torch.manual_seed(20)
        la = a.calculate_loss(batch)
        torch.manual_seed(20)
        lb = b.calculate_loss(batch)
        torch.testing.assert_close(la, lb, atol=0, rtol=0)
        la.backward(); lb.backward()
        for p,q in zip(a.parameters(), b.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        torch.testing.assert_close(a.eval().full_sort_predict(torch.arange(3)),
                                   b.eval().full_sort_predict(torch.arange(3)), atol=0, rtol=0)

    def test_leave_target_out_and_empty_history(self):
        model = SIMMProfile(self.config, self.loader)
        users = torch.tensor([0, 1, 2])
        v,t,has = model.history_means(users, torch.tensor([0, 2, 4]))
        torch.testing.assert_close(v, model.image_embedding(torch.tensor([1, 3, 5])))
        torch.testing.assert_close(t, model.text_embedding(torch.tensor([1, 3, 5])))
        self.assertTrue(has.all())
        # Singleton history after removal must become zero, despite projector biases.
        model.history_mass[0] = 1
        model.visual_sum[0] = model.image_embedding.weight[0]
        model.text_sum[0] = model.text_embedding.weight[0]
        actual = model.user_representations(torch.tensor([0]), torch.tensor([0]))
        expected = torch.nn.functional.normalize(model.user_embedding(torch.tensor([0])), dim=-1)
        torch.testing.assert_close(actual, expected)
        model.use_user_id = False
        self.assertEqual(model.user_representations(torch.tensor([0]), torch.tensor([0])).abs().sum(), 0)

    def test_shuffled_removal_only_for_present_target(self):
        model = SIMMProfile(dict(self.config, profile_mode='shuffled'), self.loader)
        model.profile_source.copy_(torch.tensor([1, 2, 0]))
        users = torch.tensor([0, 0])
        v, _, _ = model.history_means(users, torch.tensor([2, 0]))
        torch.testing.assert_close(v[0], model.image_embedding.weight[3])
        torch.testing.assert_close(v[1], model.image_embedding.weight[[2, 3]].mean(0))

    def test_all_plan_configs_have_finite_gradients_and_no_extra_parameters(self):
        jobs = build_plan()
        self.assertEqual(len(jobs), 70)
        self.assertEqual(len(build_plan((999, 2024, 2025))), 210)
        self.assertEqual(len({j['name'] for j in jobs}), len(jobs))
        batch = torch.tensor([[0, 1, 2], [0, 2, 4]])
        reference = SIMMRec(self.config, self.loader)
        for job in jobs:
            model = SIMMProfile(dict(self.config, **job['overrides']), self.loader)
            self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in reference.parameters()))
            loss = model.calculate_loss(batch)
            self.assertTrue(torch.isfinite(loss), job['name'])
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(), job['name'])
            self.assertIsNone(model.image_embedding.weight.grad)
            self.assertIsNone(model.text_embedding.weight.grad)

    def test_consistent_relabeling_preserves_scores_and_loo(self):
        model = SIMMProfile(self.config, self.loader).eval()
        remapped = copy.deepcopy(model)
        up, ip = torch.tensor([2, 0, 1]), torch.randperm(24)
        ui, ii = torch.argsort(up), torch.argsort(ip)
        with torch.no_grad():
            remapped.user_embedding.weight.copy_(model.user_embedding.weight[up])
            for name in ('image_embedding', 'text_embedding'):
                getattr(remapped, name).weight.copy_(getattr(model, name).weight[ip])
            for name in ('visual_sum', 'text_sum', 'history_mass'):
                getattr(remapped, name).copy_(getattr(model, name)[up])
            remapped.history_item_weights.copy_(model.history_item_weights[ip])
            remapped.profile_source.copy_(ui[model.profile_source[up]])
            keys = model.train_positive_keys
            remapped.train_positive_keys = (ui[keys // 24] * 24 + ii[keys % 24]).sort().values
        original = model.full_sort_predict(torch.arange(3))
        actual = remapped.full_sort_predict(torch.arange(3))
        torch.testing.assert_close(actual, original[up][:, ip])
        targets = torch.tensor([0, 2, 4])[up]
        torch.testing.assert_close(remapped.user_representations(torch.arange(3), ii[targets]),
                                   model.user_representations(up, targets))

    def test_review_uses_validation_and_requires_completed_seed_groups(self):
        from review_profile import review
        folder = Path(self.folder.name) / 'review'
        folder.mkdir()
        jobs = build_plan((999, 2024))[:6]
        rows = [dict(name=j['name'], state='complete', valid_recall=0.) for j in jobs]
        for index, row in enumerate(rows):
            row.update({'valid_recall@20': .1 if index < 2 else .05,
                        'test_recall@20': .01 if index < 2 else .2})
        rows[-1]['state'] = 'pending'
        (folder / 'manifest.json').write_text(json.dumps(dict(jobs=jobs)))
        (folder / 'summary.json').write_text(json.dumps(rows))
        result = review(folder)
        self.assertEqual(result['incomplete_groups'], 1)
        self.assertEqual(result['eligible'][0]['name'], 'baseline')
        self.assertEqual(result['eligible'][0]['seeds'], 2)
