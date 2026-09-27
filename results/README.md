# Recorded measurements

`alibaba_5seeds.csv` combines the 25 archived validation CSVs (five cases × five
training seeds) with explicit seed/case columns. `MARL` is sequence decoding;
`MARL_POINT` is pointwise argmax. Baseline rows are retained. Raw deterministic
baseline repetitions do not create independent samples. Training seeds are the
unit of uncertainty; the historical seed-3 trainer transition is documented in
`../RUN_STATUS.md`.

`reviewer_multiseed_statistics.json` records descriptive 95% Student-t intervals
(df=4), without multiple-comparison adjustment. Small n and zero-inflated power
violation counts limit statistical interpretation. These are Alibaba measurements;
Helios does not yet have five completed corrected seeds.

`helios_prior_*.csv` belongs to the completed 50% training pilot **before** the
job-conditioned actor repair. Both CSVs use the sequence evaluation harness,
which includes a separate `MARL_POINT` audit row. The 5% and 50% cases share
hours 1010–1020. `helios_placement_attribution.json` is train-only frozen-weight
oracle diagnosis for that prior pilot. None is a corrected-architecture result.

`latency/` contains raw decision-step measurements, summary and machine details
for the Alibaba seed-1 sequence decoder, 64 hosts/16 racks, validation 980/5%.
`reviewer_scaling_actor_cpu.json` measures only A2/A4 forward passes on synthetic
observations and random weights; it excludes state construction, masks,
counterfactual projection and the bridge. Do not present it as full scheduling
loop latency or counterfactual scalability.
