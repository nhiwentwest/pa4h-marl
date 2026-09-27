# Recorded experiment protocols

## Alibaba: five seeds

Each recorded seed contains 300 training episodes. Validation results are in
[results/alibaba_5seeds.csv](results/alibaba_5seeds.csv); configurations and
protocols are in [reproduction/alibaba_seeds.json](reproduction/alibaba_seeds.json).
The historical source manifest is
[reproduction/alibaba_source.sha256](reproduction/alibaba_source.sha256).

| Seeds | Trainer SHA-256 | Preparation |
| --- | --- | --- |
| 1–2 | `d10309428215ec4e23066c7d19af016ae7f38edf0f2fd9143190c2c602b2a761` | Apply `reproduction/alibaba_pre_entropy.patch` |
| 3–5 | `ba7c4dd8f262cc722048a442f1e2b08c8542cca792b3ee722c90b8018e0309b4` | Use the base trainer in `python/` |

The versions differ in the implementation of masked categorical entropy.
[reproduction/prepare.py](reproduction/prepare.py) restores and verifies the
recorded trainer for each seed in a separate source directory. A shared trainer
version does not reproduce the source selection of all five recorded seeds.

Seed 3 used the earlier trainer through checkpoint 224 and resumed with the
masked-entropy variant. A fresh seed-3 run using that variant from episode 0
has a different training history. The earlier seed 1–2 implementation can
produce non-finite masked-entropy gradients on some runtimes; matching source
hashes alone does not guarantee an identical or numerically stable replay.

## Helios: job-conditioned variant

The Helios variant uses a nonlinear rack scoring head with job context and
per-action counterfactual utility. The source changes are packaged in
[reproduction/helios_conditioned.patch](reproduction/helios_conditioned.patch).
Preparation applies them to a separate Helios source copy, preserving the
Alibaba model and trainer. The architecture requires newly initialized weights.

| Parameter | Setting |
| --- | --- |
| Training episodes | 300 |
| Training window | Hours 1000–1010, 50% sampling |
| Scheduling horizon | 180 steps |
| Primary validation | Hours 1010–1020, 5% sampling |
| Stress validation | Hours 1010–1020, 50% sampling |
| Decoders | Sequence and pointwise |
| Reserved holdout | Hours 1020–1030 |

As of 2026-09-27, the job-conditioned seed-1 experiment is in progress.
No completed validation results or five-seed estimates for this variant are
included in the repository.

The completed preceding 50% training pilot is recorded in
`results/helios_prior_*.csv`. At 5% validation, all normal policies had zero
rack violations. At 50% stress validation, RPA reward was 39.107, MARL sequence
38.911 and MARL pointwise 38.938; all three had zero rack violations.
These measurements belong to the preceding architecture. Replication of the
job-conditioned variant across seeds 2–5 is pending its seed-1 evaluation.
