#!/usr/bin/env python3
"""Prepare isolated, checksum-verified source; never launch training implicitly."""
import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'reproduction'

def digest(file):
    h = hashlib.sha256()
    with file.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_json(name):
    return json.loads((META / name).read_text())

def verify_records(root, records):
    for row in records:
        file = root / row['path']
        if not file.is_file() or digest(file) != row['sha256']:
            raise ValueError(f'Checksum mismatch or missing file: {file}')

def create_source(destination, dataset, seed):
    if dataset not in ('alibaba', 'helios') or seed not in range(1, 6):
        raise ValueError('dataset must be alibaba/helios and seed must be 1..5')
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError('Destination must be outside the source repository')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    # Check recorded core source before copying or applying a dataset-specific patch.
    for line in (META / 'alibaba_source.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if name.startswith(('python/', 'src/')) or name in (
                'scripts/run_gang.sh', 'scripts/run_eval.sh',
                'scripts/check_bridge_contract.py'):
            verify_records(ROOT, [{'path': name, 'sha256': expected}])
    destination.mkdir(parents=True)
    for name in ('python', 'src', 'scripts', 'data', 'reproduction'):
        shutil.copytree(ROOT / name, destination / name,
                        ignore=shutil.ignore_patterns('__pycache__', '._*', '*.pyc', '*.csv', '*.parquet', '*.zip', '*.tar.gz'))
    shutil.copy2(ROOT / 'pom.xml', destination / 'pom.xml')
    patch = ('alibaba_pre_entropy.patch' if seed <= 2 else None) if dataset == 'alibaba' else 'helios_conditioned.patch'
    if patch:
        subprocess.run(['git', 'apply', '--check', str(META / patch)], cwd=destination, check=True)
        subprocess.run(['git', 'apply', str(META / patch)], cwd=destination, check=True)
    if dataset == 'alibaba':
        expected = load_json('alibaba_seeds.json')[str(seed)]['trainer_sha256']
        if digest(destination / 'python/marl_gang_train.py') != expected:
            raise ValueError('Restored trainer does not match the recorded seed source')
    else:
        protocol = load_json('templates/helios_v5_rack_priority_protocol.json')
        for name in ('models', 'marl_gang_train'):
            if digest(destination / f'python/{name}.py') != protocol['training_source'][f'{name}_sha256']:
                raise ValueError(f'Helios source hash mismatch: {name}')
        # Tests for the revised actor belong only to this architecture.
        for name in ('test_action_conditioned_racks.py', 'test_counterfactual_zero_loss.py'):
            shutil.copy2(META / 'tests' / name, destination / 'python/tests' / name)
    return destination

def manifest(file, root, paths):
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(''.join(f'{digest(root / p)}  {p}\n' for p in sorted(paths)))

def prepare_runtime(dst, dataset, seed, data_root, python):
    workload = 'alibaba_v2020/pod_hourly_jobs.csv' if dataset == 'alibaba' else 'helios_earth_pod_hourly_jobs.csv'
    records = [r for r in load_json('data_manifest.json')
               if r['path'].startswith('extracted/') or r['path'] == workload]
    verify_records(data_root, records)
    templates = META / 'templates'
    reference = 'outputs/source_reference'
    if dataset == 'helios':
        names = ('helios_v5_rack_priority_env.sh', 'helios_v5_rack_priority_protocol.json',
                 'run_helios_v5_rack_priority_300_seed.sh', 'eval_helios_v5_rack_priority_300_seed.sh')
        for name in names:
            text = (templates / name).read_text().replace(
                'outputs/helios_conditioned_source_reference', reference)
            text = text.replace('26200 + seed', '28200 + seed').replace('26300 + seed', '28300 + seed')
            (dst / 'scripts' / name).write_text(text)
        protocol_file = dst / 'scripts/helios_v5_rack_priority_protocol.json'
        protocol = json.loads(protocol_file.read_text())
        protocol['data_provenance']['converted_csv'] = str(data_root / workload)
        protocol['data_provenance']['raw_csv'] = str(data_root / 'helios_cluster_log.csv')
        protocol['data_provenance']['power_root'] = str(data_root / 'extracted')
        protocol_file.write_text(json.dumps(protocol, indent=2) + '\n')
        runner = 'scripts/run_helios_v5_rack_priority_300_seed.sh'
        evaluator = None  # The Helios runner already evaluates all four cases.
        extra_env = {}
    else:
        runner = 'scripts/run_v5_rack_priority_300_seed.sh'
        evaluator = 'scripts/eval_v5_rack_priority_300_seed.sh'
        for name in (Path(runner).name, Path(evaluator).name):
            text = (templates / name).read_text().replace('2|3|4|5)', '1|2|3|4|5)')
            text = text.replace('SEED (2-5)', 'SEED (1-5)').replace('seed must be 2-5', 'seed must be 1-5')
            text = text.replace('26090 + seed', '28090 + seed').replace('26100 + seed', '28100 + seed')
            text = text.replace('        scripts/run_v5_rack_priority_300_batch.sh \\\n', '')
            if name == Path(runner).name:
                text = text.replace('mkdir -p "$live" "$root/snapshots"',
                    'mkdir -p outputs\nexec 9>"outputs/.alibaba_seed${seed}.lock"\nflock -n 9 || exit 1\nmkdir -p "$live" "$root/snapshots"', 1)
            (dst / 'scripts' / name).write_text(text)
        record = load_json('alibaba_seeds.json')[str(seed)]
        (dst / 'scripts/v5_rack_priority_300_seed1_protocol.json').write_text(json.dumps(record['protocol'], indent=2) + '\n')
        extra_env = dict(record['config']['env'])
    numerics = load_json('alibaba_seeds.json')['1']['config']['trainer_defaults']
    numerics = {key: value for key, value in numerics.items() if key != 'CLIP'}
    env = {**extra_env, **numerics, 'A4_ENT_COEF': numerics['ENT_COEF'], 'PYBIN': str(python), 'PA4H_DATA_BASE': str(data_root),
           'NREL_ROOT': str(data_root / 'extracted'), 'POD_HOURLY_JOBS': str(data_root / workload),
           'SOURCE_REFERENCE': reference, 'DACN_DATA': f'outputs/{dataset}_seed{seed}_data_cache',
           'PYTHONPATH': 'python:vendor', 'CUDA_VISIBLE_DEVICES': '',
           'START_TENSORBOARD': '0', 'OBS_ENABLED': '0', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    launch = '#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\n'
    launch += ''.join(f'export {key}={shlex.quote(str(value))}\n' for key, value in env.items())
    launch += f'bash {runner} {seed}\n'
    if evaluator:
        launch += f'bash {evaluator} {seed}\n'
    (dst / 'run_experiment.sh').write_text(launch)
    ref = dst / reference
    source = [str(p.relative_to(dst)) for folder in ('python', 'src', 'scripts')
              for p in (dst / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    manifest(ref / 'source_sha256.txt', dst, source + ['pom.xml', 'run_experiment.sh'])
    manifest(ref / 'nrel_sha256.txt', Path('/'), [str(data_root / r['path']).lstrip('/')
             for r in records if r['path'].startswith('extracted/')])
    manifest(ref / 'workload_sha256.txt', Path('/'), [str(data_root / workload).lstrip('/')])
    # Runtime manifests must contain absolute data paths.
    for name in ('nrel_sha256.txt', 'workload_sha256.txt'):
        file = ref / name
        file.write_text(file.read_text().replace('  ', '  /'))
    for file in (dst / 'scripts').glob('*.sh'):
        subprocess.run(['bash', '-n', str(file)], check=True)
    subprocess.run(['bash', '-n', str(dst / 'run_experiment.sh')], check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('alibaba', 'helios'), required=True)
    parser.add_argument('--seed', type=int, choices=range(1, 6), required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--source-only', action='store_true', help='restore source only; no executable launcher')
    args = parser.parse_args()
    if not args.source_only and not args.data_root:
        parser.error('--data-root is required unless --source-only is used')
    if args.data_root:
        workload = 'alibaba_v2020/pod_hourly_jobs.csv' if args.dataset == 'alibaba' else 'helios_earth_pod_hourly_jobs.csv'
        verify_records(args.data_root.resolve(), [r for r in load_json('data_manifest.json')
                      if r['path'].startswith('extracted/') or r['path'] == workload])
    dst = create_source(args.destination.resolve(), args.dataset, args.seed)
    if not args.source_only:
        prepare_runtime(dst, args.dataset, args.seed, args.data_root.resolve(), args.python.resolve())
    print(f'Prepared {dst}; training has NOT started.')

if __name__ == '__main__':
    main()
