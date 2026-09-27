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
        self.use_history_mlp = False


class _Env:
    def __init__(self):
        self.sla_lambda_adm = 0.7
        self.sla_lambda_restart = 0.6
        self.sla_lambda_comp = 0.8
        self._sla_rate_ema = np.array([0.3, 0.2, 0.4], dtype=np.float64)
        self._sla_dual_updates = 9
        self._sla_ema_initialized = True


class GangResumeTests(unittest.TestCase):
    def test_training_resume_rejects_checkpoint_from_old_simulator(self):
        agents, env = _Agents(), _Env()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "old_state.pt")
            save_training_state(path, agents, env, episode=4, best_batch=0.,
                                best_eval=0., best_deploy_key=None,
                                best_decoder="pointwise", stale=0)
            state = torch.load(path, weights_only=False)
            state.pop("simulator_semantics", None)
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "simulator semantics"):
                load_training_state(path, agents, env)

    def test_training_resume_rejects_v3_single_relief_state(self):
        agents, env = _Agents(), _Env()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "v3_state.pt")
            save_training_state(path, agents, env, episode=100, best_batch=0.,
                                best_eval=0., best_deploy_key=None,
                                best_decoder="sequence", stale=0)
            state = torch.load(path, weights_only=False)
            state["simulator_semantics"] = "gang-replay-v3-checkpoint-ring"
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "simulator semantics"):
                load_training_state(path, agents, env)

    def test_v4_descriptor_rejected_before_actor_or_sla_mutation(self):
        agents, env = _Agents(), _Env()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "v4_state.pt")
            save_training_state(path, agents, env, episode=12, best_batch=0.,
                                best_eval=0., best_deploy_key=None,
                                best_decoder="sequence", stale=0)
            state = torch.load(path, weights_only=False)
            state.pop("relief_credit")
            torch.save(state, path)
            before = agents.actors["a1"].weight.detach().clone()
            with self.assertRaisesRegex(ValueError, "relief_credit"):
                load_training_state(path, agents, env)
            self.assertTrue(torch.equal(before, agents.actors["a1"].weight))
            self.assertEqual(env.sla_lambda_adm, 0.7)

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
        self.assertAlmostEqual(restored_env.sla_lambda_restart, 0.6)
        self.assertTrue(torch.equal(
            source.actors["a1"].weight, restored.actors["a1"].weight))
        self.assertTrue(restored.opt["a1"].state_dict()["state"])


if __name__ == "__main__":
    unittest.main()
