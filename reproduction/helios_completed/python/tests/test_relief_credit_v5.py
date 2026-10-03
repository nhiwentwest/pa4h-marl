"""Behavioral checks for v5 whole-gang relief and shared A3/A1 credit."""
import os
import tempfile
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gang_env import JOB_OBS_DIM, RACK_OBS_DIM
from marl_gang_train import (A1_OBS_DIM, A3_OBS_DIM, A3_SLOT_DIM,
                             A1_VIOL_WEIGHT, build_relief_candidate_credit,
                             gang_step, relief_action_utilities,
                             relief_credit_metadata)
from relief_credit import ReliefCredit
from eval_gang import load_agents


class FakeEnv:
    num_racks = 2
    num_hosts = 8
    sla_lambda_comp = 1.0
    _sla_priority_total = 100.0

    def __init__(self):
        self.pending = []
        self.jobs = [SimpleNamespace(job_id=i, rack=0, num_nodes=2,
                                     priority=1, placed_hosts=[0, 4]) for i in (1, 2)]
        self.running = {j.job_id: j for j in self.jobs}
        self.ep_stats = {}
        self.projections = []

    def rack_of(self, host):
        return host // 4

    def rack_features(self):
        return np.zeros((2, RACK_OBS_DIM), dtype=np.float32)

    def project_rack_violation_fraction(self, rack, remove_job=None):
        self.projections.append((rack, None if remove_job is None else remove_job.job_id))
        before = [0.6, 0.5][rack]
        if remove_job is None:
            return before
        return before - ({0: 0.3, 1: 0.2} if remove_job.job_id == 1
                         else {0: 0.2, 1: -0.3})[rack]

    def jobs_on_rack(self, rack):
        return self.jobs if rack == 0 else []

    def preemption_sla_cost(self, job):
        return float(job.job_id)

    def project_job_energy_saved_normalized(self, job):
        return 0.1 * job.job_id

    def job_features(self, job):
        return np.zeros(JOB_OBS_DIM, dtype=np.float32)

    def preempt_job(self, job):
        del self.running[job.job_id]
        self.jobs.remove(job)


class ReliefCreditV5Tests(unittest.TestCase):
    def test_eval_rejects_missing_or_mismatched_descriptor_before_actor_load(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"GANG_CHECKPOINT_DIR": td}):
            with self.assertRaisesRegex(ValueError, "missing metric_manifest"):
                load_agents(SimpleNamespace())
            with open(os.path.join(td, "metric_manifest.json"), "w") as f:
                json.dump({"simulator_semantics": "gang-replay-v5-global-relief-credit",
                           "relief_credit": {"schema": "old"}}, f)
            with self.assertRaisesRegex(ValueError, "descriptor mismatch"):
                load_agents(SimpleNamespace())

    def test_features_and_numeric_delta(self):
        credit = ReliefCredit(.225, .05, .30, .02)
        np.testing.assert_array_equal(credit.features().shape, (4,))
        self.assertEqual(credit.features().dtype, np.float32)
        self.assertAlmostEqual(credit.delta_utility(3), .405)
        np.testing.assert_allclose(credit.a1_utilities(3), [0, .405], atol=1e-7)
        self.assertEqual(relief_credit_metadata()["a1_obs_dim"], 16)
        self.assertEqual((A1_OBS_DIM, A3_SLOT_DIM, A3_OBS_DIM), (16, 14, 63))

    def test_unique_affected_racks_and_signed_global_effect(self):
        env = FakeEnv()
        job = env.jobs[0]
        job.placed_hosts = [0, 1, 4, 5]
        before = np.array([.5, .5], dtype=np.float64)
        after, credit = build_relief_candidate_credit(env, job, before, 1.0)
        self.assertEqual(env.projections, [(0, 1), (1, 1)])
        np.testing.assert_allclose(after, [.3, .3])
        self.assertAlmostEqual(credit.global_violation_relief, .2)
        env.projections.clear()
        job.placed_hosts = [0, 1]
        after, credit = build_relief_candidate_credit(env, env.jobs[1],
                                                      np.array([.5, .5]), 2.0)
        # Job 2 still spans both racks, and its adverse second-rack effect is signed.
        self.assertAlmostEqual(credit.global_violation_relief, -.1)
        self.assertAlmostEqual(after[1], .8)

    def test_single_rack_legacy_delta_and_constant_shift(self):
        c = ReliefCredit(.3 / 2, .05, .10, .02)
        old = relief_action_utilities(.5, .2, 0, num_racks=2,
                                      energy_benefit=.05, extra_preempt_cost=.10,
                                      sla_penalty=.02)
        self.assertAlmostEqual(c.delta_utility(A1_VIOL_WEIGHT), old[1] - old[0], places=6)

    def test_cost_and_dual_changes_are_visible_to_a1_and_a3(self):
        def observe(env):
            seen = {}
            def act(name, obs, mask, step, **_):
                if name in ("a3", "a1") and name not in seen:
                    seen[name] = obs.copy()
                return int(np.flatnonzero(mask)[0])
            gang_step(env, None, 0, act)
            return seen
        base = FakeEnv()
        first = observe(base)
        changed = FakeEnv()
        changed.jobs[0].priority = 20
        changed.sla_lambda_comp = 10
        second = observe(changed)
        self.assertNotEqual(first["a1"][-2], second["a1"][-2])
        self.assertNotEqual(first["a1"][-1], second["a1"][-1])
        self.assertNotEqual(first["a3"][RACK_OBS_DIM + A3_SLOT_DIM - 2],
                            second["a3"][RACK_OBS_DIM + A3_SLOT_DIM - 2])

    def test_wait_and_preempt_both_have_finite_legal_credit(self):
        self.assertGreater(ReliefCredit(.5, .2, .01, .01).a1_utilities(3)[1], 0)
        self.assertLess(ReliefCredit(.01, 0, 2, 1).a1_utilities(3)[1], 0)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ReliefCredit(float("nan"), 0, 0, 0)

    def test_a3_and_a1_receive_same_credit_and_observation_features(self):
        env = FakeEnv()
        records = {}
        def act(name, obs, mask, step, credit_context=None, **kwargs):
            records[name] = (obs.copy(), mask.copy(), credit_context["utility"].copy())
            return int(np.flatnonzero(mask)[0])
        gang_step(env, None, 0, act)
        a3_obs, a3_mask, a3_u = records["a3"]
        a1_obs, _, a1_u = records["a1"]
        self.assertEqual((len(a3_obs), len(a1_obs)), (A3_OBS_DIM, A1_OBS_DIM))
        np.testing.assert_allclose(a3_obs[RACK_OBS_DIM + JOB_OBS_DIM + 2:
                                          RACK_OBS_DIM + A3_SLOT_DIM], a1_obs[-4:])
        self.assertAlmostEqual(a3_u[0], a1_u[1], places=6)
        self.assertEqual(a1_u[0], 0)
        self.assertEqual(a3_mask.tolist(), [True, True, False, False])
        np.testing.assert_array_equal(a3_obs[RACK_OBS_DIM + 2 * A3_SLOT_DIM:], 0)
        self.assertEqual(env.ep_stats["a2_choices"], 1)
        self.assertEqual(env.ep_stats["a2_choice_correct"], 1)
        self.assertEqual(env.ep_stats["a3_choices"], 1)
        self.assertEqual(env.ep_stats["a3_choice_correct"], 1)


if __name__ == "__main__":
    unittest.main()
