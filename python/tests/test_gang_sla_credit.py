"""SLA reward and four-agent credit regression tests (no Java bridge)."""
import unittest

import numpy as np

from gang_env import (GangEnv, JOB_OBS_DIM, SLA_TARGET_ADMISSION,
                      SLA_TARGET_COMPLETION)
from marl_gang_train import relief_action_utilities, weighted_sla_penalty


class _Job:
    def __init__(self, jid="j", arrival=0, duration=10, priority=3):
        self.job_id = jid
        self.arrival_step = arrival
        self.duration_steps = duration
        self.priority = priority
        self.num_nodes = 1
        self.peak_per_node_w = 500.0
        self.mean_per_node_w = 300.0
        self.placed_hosts = None
        self.arrival_step_placed = -1


class GangSlaCreditTests(unittest.TestCase):
    def _env(self, step=0):
        env = object.__new__(GangEnv)
        env.step_idx = step
        env.max_steps = 40
        env.per_node_budget = 1000.0
        env._first_wait = {}
        env._admission_breached = set()
        env._sla_adm_cost_step = 0.0
        env.ep_stats = dict(admission_breach=0, admission_breach_w=0.0)
        return env

    def test_job_observation_exposes_wait_and_both_deadline_slacks(self):
        env = self._env(step=10)
        job = _Job(arrival=0, duration=20)
        feat = env.job_features(job)
        self.assertEqual(JOB_OBS_DIM, 8)
        self.assertEqual(feat.shape, (8,))
        self.assertGreater(feat[5], 0.0)       # normalized wait
        self.assertGreater(feat[6], 0.0)       # admission slack remains
        self.assertGreaterEqual(feat[7], -1.0) # completion slack is bounded

    def test_admission_deadline_breach_is_charged_once_per_job(self):
        env = self._env(step=13)
        job = _Job(priority=3)
        self.assertTrue(env._record_admission_breach(job))
        self.assertFalse(env._record_admission_breach(job))
        self.assertEqual(env.ep_stats["admission_breach"], 1)
        self.assertEqual(env._sla_adm_cost_step, 3.0)

    def test_defer_cost_grows_as_admission_deadline_approaches(self):
        job = _Job(priority=3)
        early = self._env(step=1).admission_defer_cost(job)
        urgent = self._env(step=12).admission_defer_cost(job)
        self.assertGreater(urgent, early)

    def test_preemption_cost_grows_when_completion_slack_shrinks(self):
        job = _Job(duration=20, priority=3)
        job.placed_hosts = [0]
        job.arrival_step_placed = 0
        roomy = self._env(step=3).preemption_sla_cost(job)
        urgent = self._env(step=25).preemption_sla_cost(job)
        self.assertGreater(urgent, roomy)

    def test_a1_balances_rack_risk_against_preemption_sla_cost(self):
        low_cost = relief_action_utilities(
            before_violation=0.8, after_violation=0.1, preempt_cost=0.1)
        high_cost = relief_action_utilities(
            before_violation=0.2, after_violation=0.19, preempt_cost=0.9)
        self.assertGreater(low_cost[1], low_cost[0])
        self.assertLess(high_cost[1], high_cost[0])

    def test_exact_violation_projection_rewards_effective_victim(self):
        env = self._env(step=2)
        env.rack_budget = 3000.0
        env._rack_base_fine = lambda rack: np.array([3100.0, 2600.0])
        env.rack_of = lambda host: 0
        env._fine_window = lambda job, elapsed: np.array([800.0, 800.0])
        job = _Job(duration=10)
        job.placed_hosts = [0]
        job.arrival_step_placed = 0
        before = env.project_rack_violation_fraction(0)
        after = env.project_rack_violation_fraction(0, remove_job=job)
        self.assertEqual(before, 0.5)
        self.assertEqual(after, 0.0)

    def test_python_replay_includes_idle_hosts_and_cross_rack_tax(self):
        env = self._env(step=0)
        env.num_hosts = 8
        env._fine_per_step = 2
        env._base_cache = {}
        env.rack_of = lambda host: host // 4
        job = _Job(duration=10)
        job.placed_hosts = [0, 4]  # one node per rack -> multiplier 1.15
        job.arrival_step_placed = 0
        env.running = {job.job_id: job}
        env.jobs_on_rack = lambda rack: [job]
        env._fine_window = lambda job, elapsed: np.array([2000.0, 2000.0])
        rack0 = env._rack_base_fine(0)
        self.assertTrue(np.allclose(rack0, 3 * 612.0 + 2000.0 * 1.15))

    def test_dual_multiplier_rises_above_target_and_falls_below_it(self):
        env = self._env()
        env.sla_lambda_adm = 5.0
        env.sla_lambda_comp = 5.0
        env._sla_rate_ema = np.array([1.0, 1.0])
        env._sla_dual_updates = 5
        env.update_sla_multipliers(admission_rate=1.0, completion_rate=1.0)
        raised = (env.sla_lambda_adm, env.sla_lambda_comp)
        env._sla_rate_ema = np.zeros(2)
        env.update_sla_multipliers(admission_rate=0.0, completion_rate=0.0)
        self.assertGreater(raised[0], env.sla_lambda_adm)
        self.assertGreater(raised[1], env.sla_lambda_comp)

    def test_default_sla_budget_is_stricter_than_observed_failure_rate(self):
        self.assertLessEqual(SLA_TARGET_ADMISSION, 0.25)
        self.assertLessEqual(SLA_TARGET_COMPLETION, 0.25)

    def test_local_credit_keeps_base_sla_penalty_when_dual_is_zero(self):
        penalty = weighted_sla_penalty(
            risk_cost=1.5, priority_total=100.0, multiplier=0.0)
        self.assertGreater(penalty, 0.0)

    def test_a4_defer_credit_keeps_base_sla_penalty_when_dual_is_zero(self):
        env = self._env(step=12)
        env.sla_lambda_adm = 0.0
        job = _Job(priority=3)
        self.assertGreater(env.admission_defer_penalty(job), 0.0)


if __name__ == "__main__":
    unittest.main()
