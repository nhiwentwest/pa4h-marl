"""Publish complete, durable training bundles; keep the previous pointer on failure."""
import json
import os
import shutil
from pathlib import Path


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_bundle(root, name, save_state, metadata):
    root = Path(root)
    staging = root / ('.'+name)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    save_state(staging/'training_state.pt')
    (staging/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for filename in ('training_state.pt','run_metadata.json'):
        with (staging/filename).open('rb') as handle:
            os.fsync(handle.fileno())
    sync_directory(staging)
    final = root/name
    # An orphan after a restart may be retried, but an active bundle is immutable.
    if final.exists():
        if (root/'checkpoint').is_symlink() and (root/'checkpoint').resolve()==final.resolve():
            raise ValueError('cannot replace the authoritative bundle')
        shutil.rmtree(final)
    os.replace(staging,final)
    sync_directory(root)
    pointer = root/'checkpoint.next'
    pointer.unlink(missing_ok=True)
    pointer.symlink_to(name,target_is_directory=True)
    os.replace(pointer,root/'checkpoint')
    sync_directory(root)


def validate_migration(metadata, recipe_sha, source_sha):
    if (metadata['episode'] != 24 or metadata['recipe_sha256'] != recipe_sha
            or metadata['source_sha256'] != source_sha):
        raise ValueError('pilot import differs from the approved seed-1 episode-24 parent')


def selection_score(rows, baselines):
    expected = {r['hour'] for r in baselines}
    if len(rows)!=len(expected) or {r['hour'] for r in rows} != expected:
        raise ValueError('development windows missing or duplicated')
    refs = {r['hour']:r for r in baselines}
    if any(r['completed']<refs[r['hour']]['completed'] or r['sla']>refs[r['hour']]['sla']
           or r['violations']>refs[r['hour']]['violations'] for r in rows):
        return None
    return sum((r['reward']-refs[r['hour']]['reward'])/max(abs(refs[r['hour']]['reward']),1e-12)
               for r in rows)/len(rows)
