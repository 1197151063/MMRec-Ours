import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import torch
import test_simmrec as fixtures
from models.simmrec import SIMMRec
from models.simminit import SIMMInit, history_directions
from models.lightmrecorder import LightMRecOrder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from init_plan import build_plan


class UserInitTest(unittest.TestCase):
    def setUp(self):
        fixtures.SIMMRecTest.setUp(self)
        self.config.update(item_id_weight=0., user_init='none', init_strength=0., init_seed=2026)

    def test_binary_history_and_permutation(self):
        vectors = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        keys = torch.tensor([0, 1, 1, 5])
        actual = history_directions(keys, 3, 3, vectors)
        expected = torch.nn.functional.normalize(torch.tensor([[1., 1.], [-1., 0.], [0., 0.]]), dim=-1)
        torch.testing.assert_close(actual, expected)
        up, ip = torch.tensor([2, 0, 1]), torch.tensor([1, 2, 0])
        remapped_keys = up[keys // 3] * 3 + ip[keys % 3]
        remapped_vectors = torch.empty_like(vectors)
        remapped_vectors[ip] = vectors
        remapped = history_directions(remapped_keys, 3, 3, remapped_vectors)
        torch.testing.assert_close(remapped[up], actual)

    def test_initialization_only_and_rng_isolation(self):
        torch.manual_seed(40)
        baseline = SIMMRec(self.config, self.loader)
        rng = torch.get_rng_state().clone()
        for kind in ('none', 'random', 'interaction', 'content', 'mismatched'):
            torch.manual_seed(40)
            model = SIMMInit(dict(self.config, user_init=kind, init_strength=1.), self.loader)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(set(model.state_dict()), set(baseline.state_dict()))
            for key, value in model.state_dict().items():
                if key != 'user_embedding.weight' or kind == 'none':
                    torch.testing.assert_close(value, baseline.state_dict()[key])
            model.calculate_loss(torch.tensor([[0, 1, 2], [0, 2, 4]])).backward()
            self.assertTrue(torch.isfinite(model.user_embedding.weight.grad).all())
            self.assertIsNone(model.image_embedding.weight.grad)

    def test_pe_translation_through_adam_steps(self):
        self.loader.dataset_bk = SimpleNamespace(df=pd.DataFrame({'userID': self.rows, 'itemID': self.cols}))
        config = dict(self.config, order_source='original', order_seed=2026, pe_side='user')
        models = []
        for application in ('forward', 'init'):
            torch.manual_seed(42)
            models.append(LightMRecOrder(dict(config, pe_application=application), self.loader))
        # Check the parameter-translation identity in float64. Float32 rounding in
        # BatchNorm's nearly-zero bias gradients can be amplified by Adam.
        for model in models:
            model.double()
        with torch.no_grad():
            models[1].user_embedding.weight.copy_(models[0].user_embedding.weight + models[0].user_pe)
        optimizers = [torch.optim.Adam(m.parameters(), lr=.001, weight_decay=0) for m in models]
        batch = torch.tensor([[0, 1, 2], [0, 2, 4]])
        for step in range(3):
            losses = []
            for model, optimizer in zip(models, optimizers):
                torch.manual_seed(100 + step)
                optimizer.zero_grad()
                loss = model.calculate_loss(batch)
                loss.backward()
                losses.append(loss.detach())
            torch.testing.assert_close(losses[0], losses[1], atol=1e-5, rtol=1e-5)
            for p, q in zip(models[0].parameters(), models[1].parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)
            for optimizer in optimizers:
                optimizer.step()
            for model in models:
                model.eval()
            torch.testing.assert_close(models[0].full_sort_predict(torch.arange(3)),
                                       models[1].full_sort_predict(torch.arange(3)), atol=2e-5, rtol=2e-4)
            for model in models:
                model.train()

    def test_plan(self):
        self.assertEqual(len(build_plan()), 11)
        self.assertEqual(len(build_plan((999, 1000, 1001))), 33)
        self.assertEqual(len({j['name'] for j in build_plan()}), 11)
