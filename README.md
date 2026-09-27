# PA4H-MARL

GPU gang scheduling with a four-agent causal chain **A4 → A2 → A3 → A1**,
CTDE-PPO, rack-level ST-GNNs, counterfactual supervision and SLA dual ascent.
Python connects to the Java gang replay bridge through Py4J. Workloads come from
Alibaba GPU Cluster Trace v2020 and HeliosData Earth, with deterministic mappings
to NREL/NLR 0.2-second power profiles. Helios power profiles are a proxy mapping,
not power measured from the Helios cluster; see [data provenance](DATA_PROVENANCE.md).

This repository includes the recorded **five Alibaba seeds**, their settings,
compact validation results and source hashes. The current Helios actor variant is
an isolated patch: the base model/trainer remains the Alibaba version so the
Helios change cannot silently alter Alibaba replication.

- [Recorded experiment protocols](RUN_STATUS.md)
- [Prepare and run recorded experiments](reproduction/README.md)
- [Five-seed Alibaba results](results/alibaba_5seeds.csv)

Alibaba seeds 1–2 require the old trainer `d103…`; seeds 3–5 use the masked-entropy
variant `ba7c…`. Preparation restores and verifies the appropriate source in a
separate directory. Seed 3's historical resume at episode 224 must be distinguished
from a fresh run using the variant from the beginning.

The Helios job-conditioned seed-1 experiment is in progress as of 2026-09-27.
The committed Helios CSV files describe the **prior** 300-episode pilot, which did not outperform RPA
on stress reward. They do not establish performance of the job-conditioned architecture.
Seeds 2–5 remain pending that pilot's evaluation.

## Dependencies and checks

Use Java 17+, Maven and Python with `numpy`, `pandas`, `pyarrow`, `torch`,
`py4j`, `psutil`, `tensorboard` and `prometheus-client`. Historical runs used
Python 3.12.3 and Torch 2.13.0+cu130 on CPU; their complete recorded configuration
is in `reproduction/alibaba_seeds.json`. Install a Torch build suitable for your
runtime and record any version difference when reporting a fresh replication.

```bash
mvn package dependency:copy-dependencies -DskipTests
PYTHONPATH=python python3 -m unittest discover -s python/tests
python3 -m unittest discover -s reproduction/tests -p test_prepare.py -v
```

Raw data must be obtained separately; [DATA_PROVENANCE.md](DATA_PROVENANCE.md)
documents conversion and matching. `reproduction/data_manifest.json` contains
checksums for the converted workloads and training power profiles. Checkpoints,
raw traces, archives, build outputs and full training logs are excluded from Git.

The shared switch-tier network model assumes fixed bandwidth attributes; the
latency/scaling measurements in `results/reviewer_scaling_actor_cpu.json` are
specific to their recorded machine and workload. The two datasets and decoders
have workload-dependent behavior; report power, completion, SLA, preemptions,
energy and reward together.
