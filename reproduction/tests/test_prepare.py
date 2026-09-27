import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prepare', ROOT / 'reproduction/prepare.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

class SourcePreparationTests(unittest.TestCase):
    def test_all_seed_source_hashes_match_recorded_trainers(self):
        records = prepare.load_json('alibaba_seeds.json')
        with tempfile.TemporaryDirectory() as tmp:
            for seed in range(1, 6):
                dst = Path(tmp) / str(seed)
                prepare.create_source(dst, 'alibaba', seed)
                self.assertEqual(prepare.digest(dst / 'python/marl_gang_train.py'),
                                 records[str(seed)]['trainer_sha256'])
                self.assertEqual(prepare.digest(dst / 'python/models.py'),
                                 prepare.digest(ROOT / 'python/models.py'))

    def test_helios_patch_and_existing_destination_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / 'helios'
            prepare.create_source(dst, 'helios', 1)
            self.assertIn('action_context_dim=NR', (dst / 'python/marl_gang_train.py').read_text())
            self.assertNotEqual(prepare.digest(dst / 'python/models.py'),
                                prepare.digest(ROOT / 'python/models.py'))
            with self.assertRaises(ValueError):
                prepare.create_source(ROOT / 'experiment_runs', 'alibaba', 1)
            with self.assertRaises(FileExistsError):
                prepare.create_source(dst, 'alibaba', 1)

    def test_corrupted_input_manifest_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'data.csv'
            file.write_text('wrong')
            with self.assertRaises(ValueError):
                prepare.verify_records(Path(tmp), [{'path': 'data.csv', 'sha256': '0' * 64}])

if __name__ == '__main__':
    unittest.main()
