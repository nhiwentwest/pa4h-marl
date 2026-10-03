"""Source isolation and integrity checks; no data or training is required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prepare_completed', ROOT/'reproduction/prepare_helios_completed.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class CompletedPreparationTests(unittest.TestCase):
    def test_source_only_isolated_and_legacy_trainer_unchanged(self):
        original = (ROOT/'python/marl_gang_train.py').read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)/'run'
            prepare.prepare(destination)
            self.assertEqual((destination/'python/marl_gang_train.py').read_bytes(),
                             (prepare.CAPSULE/'python/marl_gang_train.py').read_bytes())
            self.assertEqual((ROOT/'python/marl_gang_train.py').read_bytes(), original)
            self.assertFalse((destination/'outputs').exists())
            self.assertFalse((destination/'reproduction_env.sh').exists())
            self.assertTrue((destination/'tests/test_qualified_branch_credit.py').is_file())
            protocol=json.loads((destination/'scripts/helios_completed_protocol.json').read_text())
            self.assertEqual(protocol['total_episodes'], 300)
            self.assertFalse(protocol['heldout_used'])
            with self.assertRaises(FileExistsError):prepare.prepare(destination)

    def test_refuses_repository_destination(self):
        with self.assertRaises(ValueError):prepare.prepare(ROOT/'nested_run')

    def test_rejects_tampered_capsule_before_creating_run(self):
        with tempfile.TemporaryDirectory() as temp:
            capsule=Path(temp)/'capsule';capsule.mkdir()
            (capsule/'source.py').write_text('modified')
            (capsule/'manifest.json').write_text(json.dumps(dict(files={'source.py':'wrong'})))
            destination=Path(temp)/'run'
            with patch.object(prepare,'CAPSULE',capsule):
                with self.assertRaises(ValueError):prepare.prepare(destination)
            self.assertFalse(destination.exists())


if __name__=='__main__':unittest.main()
