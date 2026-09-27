"""Decision-loop regressions for simultaneous rack-power risk."""
import unittest
from types import SimpleNamespace

import numpy as np

from gang_env import JOB_OBS_DIM, RACK_OBS_DIM
from marl_gang_train import gang_step


class _TwoRackRisk:
    num_racks = 2
    num_hosts = 8
    sla_lambda_comp = 1.0
    _sla_priority_total = 2.0

    def __init__(self):
        self.pending = []
        self.running = {
            i: SimpleNamespace(job_id=i, rack=i, num_nodes=1, priority=1,
                               placed_hosts=[i * 4])
            for i in range(2)
        }
        self.ep_stats = {}
        self.preempted = []

    def rack_features(self):
        return np.zeros((self.num_racks, RACK_OBS_DIM), dtype=np.float32)

    def project_rack_violation_fraction(self, rack, remove_job=None):
        return 0.25 if any(j.rack == rack and j is not remove_job
                           for j in self.running.values()) else 0.0

    def jobs_on_rack(self, rack):
        return [j for j in self.running.values() if j.rack == rack]

    def rack_of(self, host):
        return host // 4

    def project_job_energy_saved_normalized(self, job):
        return 0.0

    def preemption_sla_cost(self, job):
        return 0.0

    def job_features(self, job):
        return np.zeros(JOB_OBS_DIM, dtype=np.float32)

    def preempt_job(self, job):
        self.preempted.append(job.rack)
        del self.running[job.job_id]
        self.pending.append(job)


class MultiRackReliefTests(unittest.TestCase):
    def test_preempts_both_risky_racks_before_advance(self):
        env = _TwoRackRisk()
        calls = []

        def act(name, obs, mask, step, **kwargs):
            calls.append(name)
            return 1 if name == "a1" else int(np.flatnonzero(mask)[0])

        gang_step(env, None, 0, act)
        self.assertEqual(env.preempted, [0, 1])
        self.assertEqual(env.ep_stats["risk_steps"], 1)
        self.assertEqual(env.ep_stats["a1_preempt"], 2)
        self.assertEqual(calls.count("a4"), 0)

    def test_wait_on_one_rack_does_not_block_other_rack(self):
        env = _TwoRackRisk()
        current_rack = None

        def act(name, obs, mask, step, **kwargs):
            nonlocal current_rack
            if name == "a2":
                current_rack = int(np.flatnonzero(mask)[0])
                return current_rack
            if name == "a1":
                return int(current_rack == 1)
            return int(np.flatnonzero(mask)[0])

        gang_step(env, None, 0, act)
        self.assertEqual(env.preempted, [1])
        self.assertEqual(env.ep_stats["a1_decisions"], 3)


if __name__ == "__main__":
    unittest.main()
