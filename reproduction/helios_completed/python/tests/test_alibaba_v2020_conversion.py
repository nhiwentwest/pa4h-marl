import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data"))
from convert_alibaba_v2020 import JOB_COLUMNS, TASK_COLUMNS, TAG_COLUMNS, convert


class AlibabaV2020ConversionTest(unittest.TestCase):
    def test_joins_gpu_demand_and_execution_span_without_queue_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            job_path, task_path, tag_path, out_path = (
                base / "jobs.csv", base / "tasks.csv", base / "tags.csv", base / "queue.csv")
            pd.DataFrame([
                ["good", "id-1", "u", "Terminated", 0, 7200],
                ["failed", "id-2", "u", "Failed", 3600, 3900],
            ], columns=JOB_COLUMNS).to_csv(job_path, header=False, index=False)
            pd.DataFrame([
                ["good", "worker", 2, "Terminated", 300, 3900, 100, 1, 50, "A100"],
                ["good", "ps", 1, "Terminated", 400, 3000, 100, 1, 100, "A100"],
                ["failed", "worker", 1, "Terminated", 3600, 3800, 100, 1, 100, "A100"],
            ], columns=TASK_COLUMNS).to_csv(task_path, header=False, index=False)
            pd.DataFrame([
                ["id-1", "u", "A100", "g", "bert"],
            ], columns=TAG_COLUMNS).to_csv(tag_path, header=False, index=False)
            out = convert(job_path, task_path, out_path, tag_path)
            self.assertEqual(len(out), 1)
            row = out.iloc[0]
            self.assertEqual(row.workload_id, "id-1")
            self.assertEqual(row.total_gpu_request, 2.0)
            self.assertEqual(row.num_pods, 3)
            self.assertEqual(row.duration_hours, 1.0)
            self.assertEqual(row.model_type, "bert")
            self.assertEqual(pd.read_csv(out_path).iloc[0].arrival_hour, 0)


if __name__ == "__main__":
    unittest.main()
