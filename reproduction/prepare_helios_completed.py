#!/usr/bin/env python3
"""Prepare the versioned completed-only Helios experiment without starting it."""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / 'reproduction' / 'helios_completed'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(destination, data_root=None, python='python3'):
    destination = Path(destination).resolve()
    if destination == ROOT or ROOT in destination.parents or destination in ROOT.parents:
        raise ValueError('Destination must be separate from the repository')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    manifest = json.loads((CAPSULE / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(CAPSULE / name) != expected:
            raise ValueError(f'Helios source checksum mismatch: {name}')
    for name, expected in manifest['base_files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Simulator source checksum mismatch: {name}')
    if data_root is not None:
        data_root = Path(data_root).resolve()
        if digest(data_root / 'helios_cluster_log.csv') != manifest['data']['source_sha256']:
            raise ValueError('Helios raw-data checksum differs')
        records = json.loads((ROOT / 'reproduction/data_manifest.json').read_text())
        # The archived manifest remains the authority for all 41 power profiles.
        nrel = [row for row in records if row['path'].startswith('extracted/')]
        if len(nrel) != 42:
            raise ValueError('Expected NREL metadata and 41 recorded profiles')
        for row in nrel:
            if digest(data_root / row['path']) != row['sha256']:
                raise ValueError(f'NREL checksum mismatch: {row["path"]}')
    destination.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '._*', '*.pyc', '*.csv', '*.parquet', '*.zip', '*.tar.gz')
    for name in ('python', 'src', 'scripts', 'data'):
        shutil.copytree(ROOT / name, destination / name, ignore=ignore)
    shutil.copy2(ROOT / 'pom.xml', destination / 'pom.xml')
    for name in ('python', 'scripts', 'tests'):
        shutil.copytree(CAPSULE / name, destination / name, dirs_exist_ok=True)
    shutil.copy2(CAPSULE / 'requirements.txt', destination / 'requirements.txt')
    (destination / 'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if data_root is not None:
        subprocess.run([python, str(destination / 'scripts/prepare_helios_workload.py'),
                        '--input', str(data_root / 'helios_cluster_log.csv'),
                        '--output', str(destination / 'data/helios_completed.csv'),
                        '--outcome-policy', 'completed'], check=True,
                       env={**os.environ, 'PYTHONPATH': str(destination / 'python')})
        if digest(destination / 'data/helios_completed.csv') != manifest['data']['output_sha256']:
            raise ValueError('Converted Helios checksum differs; inspect the recorded pandas version')
        values = dict(PYBIN=python, PA4H_DATA_BASE=str(data_root), NREL_ROOT=str(data_root/'extracted'))
        (destination / 'reproduction_env.sh').write_text(''.join(
            f'export {key}={shlex.quote(value)}\n' for key, value in values.items()))
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-root')
    group.add_argument('--source-only', action='store_true')
    parser.add_argument('--python', default='python3')
    args = parser.parse_args()
    print(prepare(args.destination, args.data_root, args.python))
