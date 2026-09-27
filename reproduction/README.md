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
  --data-root /teamspace/studios/this_studio/pa4h_data \
  --python /teamspace/studios/this_studio/c/.venv/bin/python3
cd /tmp/pa4h-alibaba-seed1
mvn package dependency:copy-dependencies -DskipTests
bash run_experiment.sh
```

Repeat preparation with seeds 2–5 and distinct destinations. Seed 1–2 source
hash is `d103…`; seeds 3–5 use `ba7c…`. Do not bypass preparation by using
`launch_seed1.sh`, `run_gang.sh` or an old batch launcher on one shared source.
There is no single launcher that recreates the five historical trajectories.
In particular, seed 3's historical checkpoint-224 transition is not recreated
by fresh training. See `../RUN_STATUS.md`.

The prepared launcher trains 300 episodes, verifies the checkpoint, snapshots,
and evaluates the five recorded Alibaba validation cases using sequence decoding.
Run it again in the **same** prepared directory to resume its own training state.
Do not move checkpoints between source versions. Runtime manifests detect changes
to source, workload and NREL profiles; each seed has a lock and an isolated cache.
Fresh replication results stay in the prepared directory, separate from the
archived five-seed CSV in this repo. Generic historical scripts remain for reference.

## Helios corrected pilot

```bash
python3 reproduction/prepare.py --dataset helios --seed 1 \
  --destination /tmp/pa4h-helios-seed1 \
  --data-root /teamspace/studios/this_studio/pa4h_data \
  --python /teamspace/studios/this_studio/c/.venv/bin/python3
cd /tmp/pa4h-helios-seed1
mvn package dependency:copy-dependencies -DskipTests
PYTHONPATH=python python3 -m unittest discover -s python/tests
bash run_experiment.sh
```

This example starts a **new** pilot. The existing Lightning run is already active;
inspect that run rather than starting a duplicate. Prepared launchers use different
ports from the active run. New weights require fresh training. The runner evaluates
5% primary and 50% stress validation with sequence and pointwise decoders after
300 episodes. Seeds 2–5 must wait for the seed-1 comparison. No automatic queue
is included, and the 1020–1030 holdout is not evaluated.

To restore source without data or an executable training launcher, use
`--source-only` instead of `--data-root`. To test source selection without training:

```bash
python3 -m unittest discover -s reproduction/tests -p test_prepare.py -v
```

`alibaba_source.sha256` is historical provenance, not an executable checksum list
for the curated repository. Preparation verifies its retained simulation/model
files and produces fresh runtime manifests for the prepared harness. Helios
templates preserve the active run's settings; generated paths and unused port
numbers are adapted for the new directory. Java binaries, dependency packages,
raw traces, checkpoints, caches and full logs are deliberately not committed.
