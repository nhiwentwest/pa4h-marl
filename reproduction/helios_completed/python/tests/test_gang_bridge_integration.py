"""End-to-end gang replay checks using the compiled Java bridge."""

import os
import socket
import subprocess
import time
import unittest
from pathlib import Path

import numpy as np
from py4j.java_gateway import GatewayParameters, JavaGateway

from gang_env import GangEnv
from nrel_injection_bridge import Job


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless((ROOT / "target/classes/com/dacn/advanced/GangBridge.class").exists(),
                     "run mvn package dependency:copy-dependencies first")
class GangBridgeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        cls.process = subprocess.Popen(
            ["java", "-cp", "target/classes:target/dependency/*",
             "com.dacn.advanced.GangBridge", str(cls.port)],
            cwd=ROOT, env=dict(os.environ, NUM_HOSTS="8", GANG_MAX_STEPS="6"),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(40):
            try:
                cls.gateway = JavaGateway(
                    gateway_parameters=GatewayParameters(port=cls.port))
                if cls.gateway.entry_point.getNumHosts() == 8:
                    return
            except Exception:
                time.sleep(0.1)
        cls.process.terminate()
        raise RuntimeError("GangBridge did not start")

    @classmethod
    def tearDownClass(cls):
        cls.gateway.close()
        cls.process.terminate()
        cls.process.wait(timeout=5)

    def test_gang_placement_replay_and_completion(self):
        job = Job("smoke", 0, 2, 2, np.full(1500, 1000., dtype=np.float64),
                  0.2, 1.0, "test", "training", "test-profile", 8.0)
        env = GangEnv(port=self.port, jobs=[job], max_steps=4)
        try:
            self.assertTrue(env.place_job(job, 0))
            self.assertEqual(sum(not free for free in env.bridge.getHostFreeMap()), 2)
            first = env.advance()
            self.assertAlmostEqual(first["energy_kwh"],
                                   (2 * 1000 + 6 * 612) * 300 / 3_600_000,
                                   places=5)
            env.advance()
            self.assertEqual(env.sla_stats()["served"], 1)
            self.assertTrue(all(env.bridge.getHostFreeMap()))
        finally:
            env.gw.close()

    def test_cross_rack_ring_traffic_uses_shared_links(self):
        bridge = self.gateway.entry_point
        bridge.reset()
        hosts = self.gateway.new_array(self.gateway.jvm.int, 2)
        hosts[0], hosts[1] = 0, 4
        power = self.gateway.new_array(self.gateway.jvm.double, 1500)
        for i in range(1500):
            power[i] = 1000.0
        self.assertTrue(bridge.submitJob("cross", hosts, power, 0.2, 2, 1.0))
        self.assertFalse(bridge.submitJob("overlap", hosts, power, 0.2, 2, 1.0))
        bridge.stepGang()
        net = list(bridge.getGangNetworkMetrics())
        self.assertGreater(net[0], 0.0)
        self.assertEqual(net[1], 0.0)
        self.assertGreater(net[5], 0.0)
        self.assertGreater(bridge.getGangCrossRackFraction(), 0.0)

    def test_packed_trace_rejects_invalid_payload(self):
        bridge = self.gateway.entry_point
        bridge.reset()
        hosts = self.gateway.new_array(self.gateway.jvm.int, 1)
        hosts[0] = 0
        self.assertFalse(bridge.submitJobPacked("bad", hosts, "not-base64!", 0.2, 1, 1.0))
        self.assertTrue(bridge.getHostFreeMap()[0])

    def test_checkpoint_resumes_remaining_work_and_trace_phase(self):
        trace = np.repeat([1000., 1400., 1800.], 1500)
        job = Job("resume", 0, 1, 3, trace, 0.2, 1., "test", "training", "resume-profile", 4.)
        env = GangEnv(port=self.port, jobs=[job], max_steps=6)
        try:
            self.assertTrue(env.place_job(job, 0))
            env.advance()
            env.preempt_job(job)
            env.advance()  # queued time must not consume work or advance trace phase
            self.assertEqual(env.running_remaining(job), 2)
            self.assertTrue(env.place_job(job, 1))
            env.advance()
            self.assertAlmostEqual(env.bridge.getRackPowerW()[1], 1400 + 3 * 612)
            env.advance()
            self.assertEqual(env.sla_stats()["served"], 1)
            env.reset()
            self.assertEqual(env.running_remaining(job), 3)
            self.assertTrue(env.place_job(job, 0))
            env.advance()
            self.assertAlmostEqual(env.bridge.getRackPowerW()[0], 1000 + 3 * 612)
        finally:
            env.gw.close()

    def test_repeated_checkpoints_do_not_repeat_completed_work(self):
        job = Job("repeat", 0, 1, 3, np.repeat([1000., 1400., 1800.], 1500),
                  0.2, 1., "test", "training", "repeat-profile", 4.)
        env = GangEnv(port=self.port, jobs=[job], max_steps=6)
        try:
            for remaining in (3, 2, 1):
                self.assertTrue(env.place_job(job, 0))
                self.assertEqual(env.running_remaining(job), remaining)
                env.advance()
                if remaining > 1:
                    env.preempt_job(job)
            self.assertEqual(env.sla_stats()["served"], 1)
            self.assertEqual(env.step_idx, 3)
        finally:
            env.gw.close()

    def test_live_power_projection_matches_ring_replay_across_racks(self):
        job = Job("ring", 0, 6, 3, np.full(1500, 1200.),
                  0.2, 1., "test", "training", "ring-profile", 24.)
        env = GangEnv(port=self.port, jobs=[job], max_steps=5)
        try:
            self.assertTrue(env.place_job(job, 0))
            expected = [env.project_rack_peak(r) for r in range(env.num_racks)]
            env.advance()
            np.testing.assert_allclose(expected, list(env.bridge.getRackPeakW()), atol=1e-7, rtol=0)
        finally:
            env.gw.close()

    def test_a4_projection_predicts_actual_host_plan_power(self):
        job = Job("candidate", 0, 6, 3, np.full(1500, 1200.),
                  0.2, 1., "test", "training", "candidate-profile", 24.)
        env = GangEnv(port=self.port, jobs=[job], max_steps=5)
        try:
            _, mask, projected = env.a4_counterfactuals(job)
            self.assertTrue(mask[0])
            self.assertTrue(env.place_job(job, 0))
            env.advance()
            np.testing.assert_allclose(projected[0], list(env.bridge.getRackPeakW()), atol=1e-3, rtol=0)
        finally:
            env.gw.close()

    def test_cross_rack_tax_cannot_make_an_accepted_plan_unsafe(self):
        job = Job("unsafe", 0, 2, 3, np.full(1500, 7000.),
                  0.2, 1., "test", "training", "unsafe-profile", 8.)
        env = GangEnv(port=self.port, jobs=[job], max_steps=5)
        try:
            _, mask, _ = env.a4_counterfactuals(job)
            self.assertFalse(mask[:-1].any())
            self.assertTrue(mask[-1])
            self.assertFalse(env.place_job(job, 0))
        finally:
            env.gw.close()

    def test_picker_finds_safe_two_by_two_split_when_three_by_one_is_unsafe(self):
        job = Job("split", 0, 4, 3, np.full(1500, 3000.),
                  0.2, 1., "test", "training", "split-profile", 16.)
        env = GangEnv(port=self.port, jobs=[job], max_steps=5)
        try:
            self.assertTrue(env.place_job(job, 0))
            counts = np.bincount([env.rack_of(h) for h in job.placed_hosts], minlength=2)
            self.assertEqual(counts.tolist(), [2, 2])
            env.advance()
            self.assertEqual(env.ep_stats["rack_viol"], 0)
        finally:
            env.gw.close()

    def test_failed_resume_does_not_discard_checkpoint(self):
        job = Job("retry", 0, 1, 3, np.full(1500, 1000.),
                  0.2, 1., "test", "training", "retry-profile", 4.)
        env = GangEnv(port=self.port, jobs=[job], max_steps=5)
        try:
            self.assertTrue(env.place_job(job, 0))
            env.advance()
            env.preempt_job(job)
            self.assertFalse(env.bridge.submitJobPacked(
                job.job_id, env._jarr_int([0, 0]), env._packed_trace_for(job), 0.2, 3, 1.))
            self.assertTrue(env.place_job(job, 1))
            self.assertEqual(env.running_remaining(job), 2)
            env.advance()
            env.advance()
            self.assertEqual(env.sla_stats()["served"], 1)
        finally:
            env.gw.close()

    def test_checkpoint_preserves_accumulated_network_delay(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        process = subprocess.Popen(
            ["java", "-cp", "target/classes:target/dependency/*",
             "com.dacn.advanced.GangBridge", str(port)], cwd=ROOT,
            env=dict(os.environ, NUM_HOSTS="8", GANG_MAX_STEPS="20", NETWORK_BACKGROUND_UTIL="0.99"),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        gateway = None
        try:
            for _ in range(40):
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.1)
            gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))
            bridge = gateway.entry_point
            hosts = gateway.new_array(gateway.jvm.int, 2)
            hosts[0], hosts[1] = 0, 4
            # A one-sample power trace is enough; network delay is independent of trace length.
            trace = gateway.new_array(gateway.jvm.double, 1)
            trace[0] = 1000.
            remaining = []
            for interrupted in (False, True):
                bridge.reset()
                self.assertTrue(bridge.submitJob("network", hosts, trace, 0.2, 10, 1.))
                bridge.stepGang()
                bridge.stepGang()
                if interrupted:
                    bridge.preemptJob("network")
                    self.assertTrue(bridge.submitJob("network", hosts, trace, 0.2, 10, 1.))
                bridge.stepGang()
                remaining.append(int(bridge.getJobProgress("network")[1]))
            self.assertGreater(remaining[0], 7)  # congestion has extended the duration
            self.assertEqual(remaining[0], remaining[1])
        finally:
            if gateway is not None:
                gateway.close()
            process.terminate()
            process.wait(timeout=5)
