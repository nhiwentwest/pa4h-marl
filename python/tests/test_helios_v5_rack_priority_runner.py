import json
import os
import subprocess
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "reproduction/templates" / "helios_v5_rack_priority_env.sh"
PROTOCOL = REPO / "reproduction/templates" / "helios_v5_rack_priority_protocol.json"


class HeliosV5RackPriorityRunnerTests(unittest.TestCase):
    def test_common_environment_is_helios_only_and_uses_v5_settings(self):
        script = (
            f'source "{HELPER}"; '
            'helios_v5_export_common_env 1; env'
        )
        env = dict(os.environ)
        env.pop("POD_HOURLY_JOBS", None)
        env.pop("NREL_ROOT", None)
        env["PA4H_DATA_BASE"] = "/tmp/pa4h-test-data"
        result = subprocess.run(
            ["bash", "-c", script], check=True, capture_output=True,
            text=True, env=env,
        )
        values = dict(line.split("=", 1) for line in result.stdout.splitlines()
                      if "=" in line)

        self.assertEqual(
            values["POD_HOURLY_JOBS"],
            "/tmp/pa4h-test-data/helios_earth_pod_hourly_jobs.csv",
        )
        self.assertEqual(values["NREL_ROOT"], "/tmp/pa4h-test-data/extracted")
        self.assertEqual(values["NUM_EPISODES"], "300")
        self.assertEqual(values["NUM_HOSTS"], "64")
        self.assertEqual(values["MAX_STEPS"], "180")
        self.assertEqual(values["SUBSAMPLE"], "0.50")
        self.assertEqual(values["A1_VIOL_WEIGHT"], "8")
        self.assertEqual(values["W_VIOL"], "8")
        self.assertEqual(
            values["HELIOS_OUTPUT_ROOT"],
            "outputs/helios_conditioned_300_seed1",
        )
        self.assertEqual(
            values["DACN_DATA"],
            "outputs/helios_conditioned_300_seed1/data_cache",
        )

    def test_seed_roots_are_disjoint_and_invalid_seed_is_rejected(self):
        script = (
            f'source "{HELPER}"; '
            'helios_v5_export_common_env 1; seed1="$HELIOS_OUTPUT_ROOT"; '
            'helios_v5_export_common_env 2; '
            'test "$seed1" != "$HELIOS_OUTPUT_ROOT"'
        )
        subprocess.run(["bash", "-c", script], check=True, capture_output=True)

        invalid = subprocess.run(
            ["bash", "-c", f'source "{HELPER}"; helios_v5_export_common_env 6'],
            capture_output=True,
            text=True,
        )
        self.assertEqual(invalid.returncode, 2)

    def test_protocol_keeps_train_validation_and_holdout_disjoint(self):
        protocol = json.loads(PROTOCOL.read_text())
        train = protocol["training_workload"]
        validation = protocol["validation_workload"]
        holdout = protocol["holdout_workload"]

        self.assertEqual(protocol["seed_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(train["start_hour"], 1000)
        self.assertEqual(train["window_hours"], 10)
        self.assertEqual(train["subsample"], 0.5)
        self.assertEqual(validation["start_hour"], 1010)
        self.assertEqual(validation["window_hours"], 10)
        self.assertEqual(holdout["start_hour"], 1020)
        self.assertEqual(holdout["window_hours"], 10)
        self.assertLessEqual(
            train["start_hour"] + train["window_hours"],
            validation["start_hour"],
        )
        self.assertLessEqual(
            validation["start_hour"] + validation["window_hours"],
            holdout["start_hour"],
        )


if __name__ == "__main__":
    unittest.main()
