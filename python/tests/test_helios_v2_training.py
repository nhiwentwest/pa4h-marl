import subprocess
import unittest
from unittest.mock import patch
from pathlib import Path

from nrel_injection_bridge import _arrival_step_in_window


REPO = Path(__file__).resolve().parents[1]


class HeliosV2TrainingTest(unittest.TestCase):
    def test_arrival_window_has_inclusive_start_and_exclusive_end(self):
        self.assertTrue(_arrival_step_in_window(12000, 12000, 12120))
        self.assertTrue(_arrival_step_in_window(12119, 12000, 12120))
        self.assertFalse(_arrival_step_in_window(11999, 12000, 12120))
        self.assertFalse(_arrival_step_in_window(12120, 12000, 12120))

    def test_launcher_uses_current_v2_trainer_and_observability(self):
        source = (REPO / "launch_helios_train.sh").read_text()
        self.assertIn('bash run_gang.sh', source)
        self.assertNotIn('marl_gang_train_helios.py', source)
        self.assertIn('POD_HOURLY_JOBS', source)
        self.assertIn('WORKLOAD_START_HOUR', source)
        self.assertIn('WORKLOAD_WINDOW_HOURS', source)
        self.assertIn('NUM_EPISODES="${NUM_EPISODES:-300}"', source)
        self.assertIn('USE_STGNN="${USE_STGNN:-1}"', source)
        self.assertIn('START_TENSORBOARD="${START_TENSORBOARD:-1}"', source)
        self.assertIn('OBS_ENABLED="${OBS_ENABLED:-1}"', source)

    def test_launcher_shell_syntax(self):
        result = subprocess.run(
            ["bash", "-n", str(REPO / "launch_helios_train.sh")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_helios_eval_uses_same_window_and_shifts_arrivals(self):
        import eval_gang

        class Job:
            def __init__(self, arrival_step):
                self.arrival_step = arrival_step

        class Env:
            _all_jobs = [Job(12001), Job(12119)]

        with patch.dict("os.environ", {
                "WORKLOAD_START_HOUR": "1000",
                "WORKLOAD_WINDOW_HOURS": "10",
                "POD_HOURLY_JOBS": "data/helios_earth_pod_hourly_jobs.csv",
        }, clear=False), patch.object(eval_gang, "GangEnv", return_value=Env()) as ctor:
            env = eval_gang.build_eval_env()

        kwargs = ctor.call_args.kwargs
        self.assertEqual(kwargs["min_arrival_step"], 12000)
        self.assertEqual(kwargs["arrival_end_step"], 12120)
        self.assertNotIn("max_arrival_step", kwargs)
        self.assertEqual([job.arrival_step for job in env._all_jobs], [1, 119])

    def test_helios_entrypoint_sets_dataset_before_importing_harness(self):
        source = (Path(__file__).with_name("eval_baselines_helios.py").read_text())
        self.assertLess(source.index("POD_HOURLY_JOBS"),
                        source.index("from eval_gang import main"))
        self.assertIn('USE_STGNN", "1"', source)
        self.assertIn('WORKLOAD_START_HOUR", "1000"', source)
        self.assertIn('WORKLOAD_WINDOW_HOURS", "10"', source)


if __name__ == "__main__":
    unittest.main()
