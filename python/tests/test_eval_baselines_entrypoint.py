import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval_gang import load_sla_multipliers


class BaselineEntrypointTest(unittest.TestCase):
    def test_eval_baselines_delegates_to_canonical_gang_harness(self):
        source = (Path(__file__).resolve().parents[1] / "eval_baselines.py").read_text()
        tree = ast.parse(source)
        imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
        self.assertTrue(any(n.module == "eval_gang" and
                            any(a.name == "main" for a in n.names)
                            for n in imports))
        self.assertNotIn("num_hosts=20", source)
        self.assertNotIn("helios_earth", source)

    def test_v2_eval_restores_all_three_sla_multipliers(self):
        class Env:
            sla_lambda_adm = -1.0
            sla_lambda_restart = -1.0
            sla_lambda_comp = -1.0

        with tempfile.TemporaryDirectory() as td:
            Path(td, "sla_multipliers.json").write_text(json.dumps({
                "admission": 0.11,
                "restart": 0.22,
                "completion": 0.33,
            }))
            env = Env()
            with patch.dict(os.environ, {"GANG_CHECKPOINT_DIR": td}):
                load_sla_multipliers(env)

        self.assertAlmostEqual(env.sla_lambda_adm, 0.11)
        self.assertAlmostEqual(env.sla_lambda_restart, 0.22)
        self.assertAlmostEqual(env.sla_lambda_comp, 0.33)


if __name__ == "__main__":
    unittest.main()
