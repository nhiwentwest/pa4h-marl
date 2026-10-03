import json
from pathlib import Path
import pytest
from longrun_checkpoint import publish_bundle, validate_migration, selection_score


def test_bundle_keeps_previous_checkpoint_if_writer_fails(tmp_path):
    publish_bundle(tmp_path, 'ep024_import', lambda p: p.write_bytes(b'full state'), {'episode':24})
    def broken(p):
        p.write_bytes(b'partial')
        raise RuntimeError('restart')
    with pytest.raises(RuntimeError):
        publish_bundle(tmp_path, 'ep028_train', broken, {'episode':28})
    assert (tmp_path/'checkpoint/training_state.pt').read_bytes() == b'full state'
    assert json.loads((tmp_path/'checkpoint/run_metadata.json').read_text())['episode'] == 24


def test_bundle_can_publish_same_episode_after_evaluation(tmp_path):
    for phase in ('train', 'eval'):
        publish_bundle(tmp_path, 'ep028_'+phase, lambda p: p.write_bytes(b'state'), {'phase':phase})
    assert (tmp_path/'checkpoint').readlink() == Path('ep028_eval')
    assert (tmp_path/'ep028_train').is_dir()


def test_import_only_accepts_exact_pilot():
    meta = dict(episode=24, recipe_sha256='recipe', source_sha256='source')
    validate_migration(meta, 'recipe', 'source')
    for change in ({'episode':20}, {'source_sha256':'wrong'}, {'recipe_sha256':'wrong'}):
        with pytest.raises(ValueError):
            validate_migration(dict(meta, **change), 'recipe', 'source')


def test_selection_checks_both_windows_before_reward():
    refs = [dict(hour=h, completed=10, sla=1, violations=0, reward=10.) for h in (940,980)]
    good = [dict(r, reward=10.1) for r in refs]
    assert selection_score(good, refs) == pytest.approx(.01)
    assert selection_score([good[0],dict(good[1], completed=9, reward=100)],refs) is None
    assert selection_score([good[0],dict(good[1], sla=2)],refs) is None
    assert selection_score([good[0],dict(good[1], violations=1)],refs) is None
    with pytest.raises(ValueError):
        selection_score([good[0],good[0]],refs)
