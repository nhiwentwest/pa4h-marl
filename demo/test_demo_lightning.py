import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class DemoLightningTest(unittest.TestCase):
    def test_demo_reports_real_run_and_policy_table(self):
        script = Path(__file__).with_name("demo_lightning.py")
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            with (run / "train.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "ep", "team_reward", "served", "rack_viol",
                    "sla_admission_pct", "sla_restart_pct",
                    "sla_completion_pct", "energy_kwh",
                    "network_cross_byte_pct", "a1_decisions",
                    "a1_preempt", "a3_executed",
                ])
                writer.writeheader()
                writer.writerow(dict(
                    ep=299, team_reward=6.57, served=34, rack_viol=212,
                    sla_admission_pct=0, sla_restart_pct=0,
                    sla_completion_pct=10.53, energy_kwh=584.4,
                    network_cross_byte_pct=44.8, a1_decisions=4,
                    a1_preempt=3, a3_executed=3,
                ))
            with (run / "eval_baselines_helios.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "policy", "reward", "served_pct", "sla_pct",
                    "rack_viol", "energy_kwh", "network_cross_byte_pct",
                ])
                writer.writeheader()
                writer.writerow(dict(policy="MARL", reward=6.57,
                                     served_pct=89, sla_pct=10.53,
                                     rack_viol=212, energy_kwh=584.4,
                                     network_cross_byte_pct=44.8))
            (run / "deployment_decoder.json").write_text(
                json.dumps({"name": "pointwise"}))
            (run / "training_state.pt").write_bytes(b"checkpoint")

            result = subprocess.run(
                ["python3", str(script)],
                env={**os.environ, "DEMO_RUN_DIR": str(run)},
                capture_output=True, text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GPU CLUSTER SCHEDULING", result.stdout)
        self.assertIn("A4 -> A2 -> A3 -> A1", result.stdout)
        self.assertIn("episode checkpoint", result.stdout)
        self.assertIn("300 / 300", result.stdout)
        self.assertIn("MARL", result.stdout)
        self.assertIn("pointwise argmax", result.stdout)


if __name__ == "__main__":
    unittest.main()
