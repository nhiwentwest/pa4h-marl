# Recorded measurements

Updated 2026-10-03. Paired values are **MARL / RPA** on the same window.
SLA percentages count unique violating jobs. Wait cost is the simulator metric,
not minutes; power violations count rack replay samples. Compare within each
dataset because workloads and horizons differ.

## Alibaba — best recorded seed 5

This is the sequence decoder from the completed 300-episode seed-5 run.
Seed 5 has the highest mean raw validation reward across all five recorded
cases (16.646947); one seed is used for every row. This best-seed view
is descriptive and does not replace the five-seed statistics.
Source: [all five seeds](alibaba_5seeds.csv),
[seed-specific protocols](../reproduction/alibaba_seeds.json).

| Window / sampling | Reward ↑ | Completed jobs ↑ | SLA violations (%) ↓ | Wait cost ↓ | Preemptions ↓ | Power violations ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 980 / 1% | 6.215 / 6.168 | 48 / 48 | 14.29 / 14.29 | 59 / 65 | 4 / 5 | 0 / 0 |
| 990 / 1% | -2.375 / -2.388 | 44 / 44 | 4.35 / 4.35 | 46 / 48 | 0 / 1 | 0 / 0 |
| 1010 / 1% | 7.821 / 7.712 | 36 / 36 | 12.20 / 12.20 | 43 / 45 | 2 / 2 | 0 / 0 |
| 980 / 5% | 46.154 / 41.761 | 215 / 213 | 11.57 / 15.29 | 380 / 621 | 27 / 24 | 0 / 0 |
| 990 / 5% | 25.420 / 25.318 | 214 / 214 | 8.94 / 8.94 | 237 / 241 | 4 / 4 | 0 / 0 |

## Helios — seed 1, best checkpoint ep72

This is the completed-only queue-relief variant with grouped-admission and
completion-preference decoding. Among completed development evaluations through
ep120, ep72 has the highest mean raw return per step
(-0.05219813) while satisfying each window's proposed completion,
SLA and power gate. RPA does not select the checkpoint. One checkpoint is used
for all three rows. The 300-episode training run is still in progress; seed 2–5
and heldout hours 1020/1030 have not been evaluated for this variant.
Source: [development evaluations and RPA comparisons](helios_completed_seed1_development.json),
[reproduction instructions](../reproduction/README.md).

| Window / sampling | Reward ↑ | Completed jobs ↑ | SLA violations (%) ↓ | Wait cost ↓ | Preemptions ↓ | Power violations ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Long 1310 / 50% | 33.253 / 33.085 | 174 / 174 | 7.45 / 7.45 | 191 / 195 | 2 / 3 | 0 / 0 |
| Gang 1290 / 75% | 50.096 / 50.313 | 71 / 71 | 12.50 / 12.50 | 191 / 189 | 13 / 12 | 0 / 0 |
| Queue 1660 / 100% | -111.535 / -398.631 | 4161 / 4161 | 5.79 / 66.51 | 37223 / 91687 | 6 / 1 | 0 / 0 |

The queue-relief mechanism is an experimental extension to the paper's power-only
risk detector. These simulator SLA rules are not native Helios service guarantees.
The grouped decoder is different from the paper's two original decoders.

Full five-seed statistics, earlier Helios pilots and latency measurements remain
available in this directory with their original scopes; these two best-run tables
are not five-seed averages or heldout results.
