"""Training-state checkpoint regression tests (no Java bridge)."""
import os
import tempfile
import unittest

import numpy as np
import torch

from marl_gang_train import load_training_state, save_training_state


class _Agents:
    def __init__(self):
        self.actors = {"a1": torch.nn.Linear(2, 2)}
        self.critic = torch.nn.Linear(3, 1)
        self.opt = {
            "a1": torch.optim.Adam(self.actors["a1"].parameters(), lr=1e-3),
            "critic": torch.optim.Adam(self.critic.parameters(), lr=1e-3),
        }
        self.ent_coef = {"a1": 0.03}
        self.use_stgnn = False


class _Env:
    def __init__(self):
        self.sla_lambda_adm = 0.7
        self.sla_lambda_comp = 0.8
        self._sla_rate_ema = np.array([0.3, 0.4], dtype=np.float64)
        self._sla_dual_updates = 9
        self._sla_ema_initialized = True


class GangResumeTests(unittest.TestCase):
    def test_roundtrip_restores_models_optimizers_progress_and_sla_state(self):
        source = _Agents()
        env = _Env()
        loss = source.actors["a1"](torch.ones(1, 2)).sum()
        source.opt["a1"].zero_grad()
        loss.backward()
        source.opt["a1"].step()

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "training_state.pt")
            save_training_state(
                path, source, env, episode=168, best_batch=30.4,
                best_eval=29.58, best_deploy_key=(1, 1, -0.1),
                best_decoder="pointwise", stale=0)
            restored = _Agents()
            restored_env = _Env()
            restored_env.sla_lambda_adm = 0.0
            state = load_training_state(path, restored, restored_env)

        self.assertEqual(state["episode"], 168)
        self.assertEqual(state["best_decoder"], "pointwise")
        self.assertAlmostEqual(restored_env.sla_lambda_adm, 0.7)
        self.assertTrue(torch.equal(
            source.actors["a1"].weight, restored.actors["a1"].weight))
        self.assertTrue(restored.opt["a1"].state_dict()["state"])


if __name__ == "__main__":
    unittest.main()
