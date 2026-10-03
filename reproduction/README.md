# Recorded experiments

Run from the repository root. Preparation copies source into a **new** directory,
checks recorded source and data hashes, and restores the dataset/seed-specific
trainer. It never starts training or overwrites an existing directory. Linux,
Git, Bash, `flock`, `sha256sum`, Java/Maven and the Python dependencies in the
root README are required. Historical Python/Torch versions are recorded in
`alibaba_seeds.json`; cross-version runs are not guaranteed bit-identical.

## Alibaba

Prepare each seed separately, including source selection for seeds 1–2:

```bash
python3 reproduction/prepare.py --dataset alibaba --seed 1 \
  --destination /tmp/pa4h-alibaba-seed1 \
  --data-root /path/to/data \
  --python /path/to/venv/bin/python3
cd /tmp/pa4h-alibaba-seed1
mvn package dependency:copy-dependencies -DskipTests
bash run_experiment.sh
```

Repeat preparation with seeds 2–5 and distinct destinations. Seed 1–2 source
hash is `d103…`; seeds 3–5 use `ba7c…`. The generic `launch_seed1.sh` and
`run_gang.sh` entry points do not select the historical trainer for each seed.
There is no single launcher that recreates the five historical trajectories.
In particular, seed 3's historical checkpoint-224 transition is not recreated
by fresh training. See `../RUN_STATUS.md`.

The prepared launcher trains 300 episodes, verifies the checkpoint, snapshots,
and evaluates the five recorded Alibaba validation cases using sequence decoding.
Run it again in the **same** prepared directory to resume its own training state.
Checkpoints are specific to their source version. Runtime manifests detect changes
to source, workload and NREL profiles; each seed has a lock and an isolated cache.
Fresh replication results stay in the prepared directory, separate from the
archived five-seed CSV in this repo. Generic historical scripts remain for reference.

## Current Helios completed-only and queue-relief experiment

Use the root README command with `prepare_helios_completed.py`. It leaves the
base Alibaba files intact and verifies the separate Helios source manifest.
Only seed 1 is recorded for this variant. The prepared directory refuses an
overwrite; an existing run resumes through its launcher, not through preparation.

Required data-root layout:

```text
helios_cluster_log.csv
extracted/01_aggregated_datasets/training/metadata.csv
extracted/01_aggregated_datasets/training/results/000000.parquet ...
```

Use the archived Earth CSV and all 41 profiles with the checksums in
`data_manifest.json`. The generated CSV has 313,953 completed jobs, retains
the pre-filter time origin and must hash to `f11325ed…`. No raw data or trained
weights are distributed in this source package. Dependency pins record the
Lightning environment (Python 3.12); use a compatible CPU Torch build and
record deviations. Numerical parity across dependency versions is not promised.

After preparing and building Java, run the targeted checks:

```bash
A4_BRANCH_RANK=1 QUEUE_RELIEF_ENABLED=1 PYTHONPATH=python python3 -m pytest -q \
  tests python/tests/test_queue_relief.py python/tests/test_a4_return_credit.py \
  python/test_grouped_admission.py python/test_completion_victim.py
```

The launcher uses 64 exclusive GPU hosts, 16 racks, 300-second steps and a
180-step horizon. It evaluates RPA once, then trains newly initialized weights
for 300 episodes. Each four-episode PPO update publishes an atomic checkpoint.
Development evaluation runs every 24 episodes and at episode 300. Reuse of branch
labels requires the same prefix, policy, recipe and SLA multipliers; missing SLA
evidence is not filled in. Aliases preserve ordered host plans and tied-best mass.

Train hours are 740, 790, 890, 570, 760 and 860; development hours are 1310,
1290 and 1660, with the exact sampling fractions in
`helios_completed/scripts/helios_completed_protocol.json`. Hours 1020/1030 remain
held out. Branch collection never reads development or heldout states.
The best checkpoint must satisfy each development window's proposed completion,
SLA and power gate; among qualifying checkpoints, higher mean raw return per
simulator step wins. RPA is a comparison, not a selection target. These gates
are experiment proposals, not additional requirements claimed from the paper.

For source and tests without data, prepare with `--source-only` instead of
`--data-root`. `STOP_AFTER=4 bash scripts/run_helios_completed.sh` makes a bounded
four-episode run while retaining the 300-episode recipe; rerun without that
variable to continue. Training state semantics reject unrelated checkpoints.

## Earlier Helios job-conditioned variant

```bash
python3 reproduction/prepare.py --dataset helios --seed 1 \
  --destination /tmp/pa4h-helios-seed1 \
  --data-root /path/to/data \
  --python /path/to/venv/bin/python3
cd /tmp/pa4h-helios-seed1
mvn package dependency:copy-dependencies -DskipTests
PYTHONPATH=python python3 -m unittest discover -s python/tests
bash run_experiment.sh
```

The launcher trains newly initialized weights for 300 episodes, then evaluates
5% primary and 50% stress validation using sequence and pointwise decoders.
The 1020–1030 holdout is excluded. Seed 2–5 replication is pending seed-1
evaluation, as recorded in [the experiment protocols](../RUN_STATUS.md).

To restore source without data or an executable training launcher, use
`--source-only` instead of `--data-root`. To test source selection without training:

```bash
python3 -m unittest discover -s reproduction/tests -p test_prepare.py -v
```

`alibaba_source.sha256` is historical provenance, not an executable checksum list
for the curated repository. Preparation verifies its retained simulation/model
files and produces fresh runtime manifests for the prepared harness. Helios
templates specify the job-conditioned protocol; preparation resolves data paths
and assigns bridge ports for the generated launcher. Java binaries, dependency packages,
raw traces, checkpoints, caches and full logs are deliberately not committed.
