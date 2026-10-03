import unittest
import numpy as np
import torch
from a4_return_credit import discounted_returns, marginal_baseline, return_advantage, ReturnCritic, publish_selected_bundle

class ReturnCreditTests(unittest.TestCase):
    def test_return_contains_future_cost_and_terminal_boundary(self):
        np.testing.assert_allclose(discounted_returns([1., -2., 100.], [0., 1., 1.], .5), [0., -2., 100.])
    def test_probability_mask_and_no_nan(self):
        q = torch.tensor([[2., 6., float('nan')]])
        p = torch.tensor([[.25, .75, 0.]])
        mask = torch.tensor([[True, True, False]])
        self.assertAlmostEqual(marginal_baseline(q, p, mask).item(), 5.)
        with self.assertRaises(ValueError):
            marginal_baseline(q, p, torch.zeros_like(mask))
    def test_advantage_is_real_return_and_not_pressure(self):
        got = return_advantage(np.array([4., -2., 0.]), np.array([1., 1., 1.]))
        np.testing.assert_allclose(got, np.array([1.3363062, -1.069045, -.26726124]), rtol=1e-5)
    def test_critic_fits_chosen_returns_and_serializes(self):
        torch.manual_seed(7)
        c = ReturnCritic(3, 4, 2, lr=.01)
        g = torch.randn(64, 3); obs = torch.randn(64, 4)
        acts = torch.arange(64) % 2
        ret = 2*g[:, 0] + obs[:, 0] + acts.float()
        for _ in range(80):
            loss = c.fit(g, obs, acts, ret, epochs=1)
        self.assertLess(loss, .08)
        restored = ReturnCritic(3, 4, 2)
        restored.load_state_dict(c.state_dict())
        torch.testing.assert_close(c.model(g, obs), restored.model(g, obs))
        self.assertEqual(c.updates, restored.updates)
    def test_baseline_requires_predictive_quality_before_activation(self):
        c = ReturnCritic(2, 2, 2)
        self.assertFalse(c.ready)
        c.observe_quality(q_mse=1., value_mse=2.)
        self.assertFalse(c.ready)
        c.observe_quality(q_mse=1., value_mse=2.)
        self.assertTrue(c.ready)
        c.observe_quality(q_mse=3., value_mse=2.)
        self.assertFalse(c.ready)

    def test_selected_bundle_keeps_its_sla_and_failed_publish_keeps_old_pointer(self):
        import tempfile, json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for role in ('a1', 'a2', 'a3', 'a4', 'critic'):
                (root / f'{role}_gang_best.pt').write_bytes(b'weights')
            for name in ('sla_multipliers.json', 'deployment_decoder.json', 'metric_manifest.json'):
                (root / name).write_text('{"value": 1}')
            publish_selected_bundle(root, 4, 2., 'pointwise', (1, 0))
            original = (root / 'selected').resolve()
            (root / 'sla_multipliers.json').write_text('{"value": 9}')
            self.assertEqual(json.loads((root/'selected'/'sla_multipliers.json').read_text())['value'], 1)
            (root / 'a4_gang_best.pt').unlink()
            with self.assertRaises(FileNotFoundError):
                publish_selected_bundle(root, 8, 3., 'sequence', (1, 1))
            self.assertEqual((root/'selected').resolve(), original)

if __name__ == '__main__': unittest.main()
