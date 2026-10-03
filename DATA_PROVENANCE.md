# Data provenance

The experiments use Alibaba GPU Cluster Trace v2020 and HeliosData Earth as
workload sources, paired with 41 training power profiles from NLR submission
312. The power traces have a 0.2-second sampling interval. Source archives and
converted workloads are identified by SHA-256 below; per-profile checksums are
listed in [the data manifest](reproduction/data_manifest.json).

The data root is supplied through `--data-root` during experiment preparation.
Paths in this table describe the expected directory layout independently of the
machine used for training.

| Dataset | Path relative to the data root | SHA-256 |
|---|---|---|
| [HeliosData Earth](https://github.com/S-Lab-System-Group/HeliosData) `cluster_log.csv` | `helios_cluster_log.csv` | `bb62d7842ee50a093c1f872afa293a8e27500f38dfc5cd1dab3b10c53746b3db` |
| Converted Helios workload (427,148 GPU jobs) | `helios_earth_pod_hourly_jobs.csv` | `7bed08070ef3332d33081fd95327d9254cb1199683a17de2bb06b5c1ddbdc531` |
| [NLR GenAI power profiles, submission 312](https://data.nlr.gov/submissions/312), source ZIP | Archive filename chosen at download | `dcad6de800fb565d850b163902e2eddae48aabd1ed1c7336f9a1cdaf3012f137` |
| NLR training metadata (41 profiles) | `extracted/01_aggregated_datasets/training/metadata.csv` | `385f20516edeef936115d3f4d9a874ac7ee2e36a69863be496e9f017df67b5a8` |
| [Alibaba GPU trace v2020](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020) `pai_job_table.tar.gz` | `alibaba_v2020/pai_job_table.tar.gz` | `5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0` |
| Alibaba v2020 `pai_task_table.tar.gz` | `alibaba_v2020/pai_task_table.tar.gz` | `cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5` |
| Alibaba v2020 `pai_group_tag_table.tar.gz` | `alibaba_v2020/pai_group_tag_table.tar.gz` | `722fef30b7fb7aa50dabd79155614b5423a9d65cf45a9b26c590d57725423a14` |
| Converted Alibaba workload (714,903 GPU jobs) | `alibaba_v2020/pod_hourly_jobs.csv` | `97edc8da95c798df99f5c4559be60f5532a9b718174892ac36db7cc231c249c7` |

## Helios conversion

[data/convert_helios_to_alibaba_schema.py](data/convert_helios_to_alibaba_schema.py)
produces 427,148 GPU jobs. `duration_hours` is calculated from the source
`duration` field in seconds. Jobs without positive GPU demand or duration are
excluded. Submission times determine arrival hours.

Helios jobs have `model_type=unknown`. Compatible power profiles are selected
with a deterministic hash of the job ID. These profiles provide a reproducible
power proxy; they are not power measurements from the Helios cluster.

## Current completed-only Helios view

The current experiment uses `python/helios_workload.py` from the isolated
`reproduction/helios_completed/` package. It retains the common time origin
`2020-03-20T16:06:15` before filtering outcomes, then selects COMPLETED jobs.
This produces **313,953 jobs**, SHA-256
`f11325eda7595fb9180280eca3653ada40754fab17f5f51da0b1061989c65528`.
The earlier 427,148-job conversion above is a different outcome view.
Cancelled, failed and timed-out jobs do not contribute fabricated completion labels.
The all-observed view is an ablation; observed runtime does not identify its
unobserved remaining work. Preparation checks the raw-source hash before converting.

The simulator rounds measured durations to 300-second steps and allocates hosts
exclusively, with four GPUs per host. A one-GPU source job therefore occupies
one whole simulated host. SLA uses the configured 12-step limits; it is a
simulator evaluation rule, not a service-level promise native to HeliosData.

## Alibaba conversion

[data/convert_alibaba_v2020.py](data/convert_alibaba_v2020.py), with `--tags`,
produces 714,903 GPU jobs from the job, task and group-tag tables:

- Submission time: `pai_job_table.start_time`.
- GPU demand: the sum of `inst_num * plan_gpu / 100` across GPU tasks.
- Execution duration: latest GPU-task finish minus earliest GPU-task start.
- Model label: `pai_group_tag_table.workload` when available, otherwise `unknown`.

Only terminated jobs and tasks with positive GPU demand and execution duration
are retained. The queue label `training` denotes the selected GPU workload;
it does not establish that every source job represents model training.

## Power assignment and evaluation windows

Profiles are matched by model family and node count where those attributes
are available. Unknown model families use deterministic job-ID hashing within
the compatible registry. Each scheduling interval lasts 300 seconds and replays
the corresponding fine-grained power samples.

The five recorded Alibaba seeds use training hours 1000–1010 at 3% sampling,
with a 120-step horizon. Their five validation cases are recorded in
[alibaba_seeds.json](reproduction/alibaba_seeds.json).

The Helios job-conditioned variant uses training hours 1000–1010 at 50%
sampling (358 matched jobs for seed 1) and a 180-step horizon. Primary validation
uses hours 1010–1020 at 5%; stress validation uses the same hours at 50%.
Both sequence and pointwise decoders are evaluated. The earlier 5% training
pilot contained 38 matched jobs and represents a different training protocol.
Hours 1020–1030 are reserved for holdout evaluation.
