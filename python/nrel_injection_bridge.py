"""
Real gang-job workload injection for the GPU-exclusive rack-power env (hướng A).

Builds a deterministic arrival queue by joining:
  - pod_hourly_jobs.csv   (workload_id, arrival_hour, total_gpu_request, job_type, model_type)
  - job_execution_summary (workload_id -> duration_hours)              [real durations]
  - NREL registry         (node-count + model family -> per-node power trace)

No random offsets, no dummy traces: every field is derived from data. Intra-hour
arrival offset and the profile pick are hash-deterministic on workload_id, so the
same episode is reproducible across processes (matches CTDE multi-env determinism).

Each Job carries ONE PERIOD of the NREL per-node fine trace (0.2s, W). The Java
bridge loops that period over the job's full duration (see setHostJobTrace), so
memory stays bounded even for 24h jobs.
"""
import os
import hashlib
import numpy as np
import pandas as pd

import alibaba_nlr_matcher as M

INTERVAL_SEC = int(os.environ.get("INTERVAL_SEC", 300))
DT_SEC = 0.2
GPUS_PER_NODE = 4
NUM_HOSTS = int(os.environ.get("NUM_HOSTS", 20))
# Only jobs that physically fit and can co-reside to create rack pressure.
MAX_JOB_NODES = int(os.environ.get("MAX_JOB_NODES", 8))
STEPS_PER_HOUR = max(1, 3600 // INTERVAL_SEC)

DATA_ROOT = os.environ.get("DACN_DATA", "data")
JOBS_CSV = os.environ.get(
    "POD_HOURLY_JOBS", os.path.join(DATA_ROOT, "alibaba", "pod_hourly_jobs.csv"))
SUMMARY_PARQUET = os.environ.get(
    "ALIBABA_SUMMARY",
    os.path.join(DATA_ROOT, "alibaba",
                 "asi_opensource_job_execution_summary", "part-000.parquet"))
NREL_ROOT = os.environ.get("NREL_ROOT", os.path.join(DATA_ROOT, "nlr", "extracted"))
DURATION_CACHE = os.path.join(DATA_ROOT, "alibaba", "workload_duration_h.parquet")

# SLA priority weight: higher = more important, preempted last. Online serving >
# training > offline/batch. Used by A1/A3 victim selection and the wait-cost.
PRIORITY = {
    "online_inference": 3.0,
    "training": 1.0,
    "dev": 0.7,
    "offline_inference": 0.5,
    "unknown": 0.5,
}


def _hash_int(s):
    return int(hashlib.md5(str(s).encode()).hexdigest(), 16)


class Job:
    __slots__ = ("job_id", "arrival_step", "num_nodes", "duration_steps",
                 "per_node_trace_w", "dt_sec", "priority", "model_type",
                 "job_type", "profile_path", "gpu_request",
                 # runtime placement state (set by GangEnv)
                 "placed_hosts", "arrival_step_placed",
                 # baseline telemetry tracking
                 "estimated_power_w", "start_step", "assigned_rack", "assigned_host_ids",
                 "last_preempted_rack", "not_before_step", "preemption_count", "status",
                 "planned_start_step", "planned_rack", "remaining_duration_steps")

    def __init__(self, job_id, arrival_step, num_nodes, duration_steps,
                 per_node_trace_w, dt_sec, priority, model_type, job_type,
                 profile_path, gpu_request):
        self.job_id = job_id
        self.arrival_step = int(arrival_step)
        self.num_nodes = int(num_nodes)
        self.duration_steps = int(duration_steps)
        self.per_node_trace_w = per_node_trace_w      # np.float64[W] @ dt_sec, one period
        self.dt_sec = float(dt_sec)
        self.priority = float(priority)
        self.model_type = model_type
        self.job_type = job_type
        self.profile_path = profile_path
        self.gpu_request = float(gpu_request)
        self.placed_hosts = None
        self.arrival_step_placed = -1
        
        # Telemetry tracking for Baselines
        self.estimated_power_w = self.mean_per_node_w * self.num_nodes
        self.start_step = None
        self.assigned_rack = None
        self.assigned_host_ids = ()
        self.last_preempted_rack = None
        self.not_before_step = 0
        self.preemption_count = 0
        self.status = "pending"
        self.planned_start_step = None
        self.planned_rack = None
        self.remaining_duration_steps = self.duration_steps

    @property
    def peak_per_node_w(self):
        return float(np.max(self.per_node_trace_w)) if len(self.per_node_trace_w) else 0.0

    @property
    def mean_per_node_w(self):
        return float(np.mean(self.per_node_trace_w)) if len(self.per_node_trace_w) else 0.0

    def __repr__(self):
        return (f"Job({self.job_id[:8]}, arr={self.arrival_step}, "
                f"nodes={self.num_nodes}, dur={self.duration_steps}st, "
                f"peak/node={self.peak_per_node_w:.0f}W, prio={self.priority})")


def _load_duration_map():
    """workload_id -> duration_hours (real), from the Alibaba summary. Cached."""
    if os.path.exists(DURATION_CACHE):
        d = pd.read_parquet(DURATION_CACHE)
        return dict(zip(d["workload_id"], d["duration_hours"]))
    if not os.path.exists(SUMMARY_PARQUET):
        return {}
    s = pd.read_parquet(SUMMARY_PARQUET, columns=["workload_id", "duration_hours"])
    s = (s.dropna(subset=["duration_hours"])
           .groupby("workload_id", as_index=False)["duration_hours"].max())
    try:
        s.to_parquet(DURATION_CACHE, index=False)
    except Exception:
        pass
    return dict(zip(s["workload_id"], s["duration_hours"]))


def _arrival_step_in_window(arrival_step, start_step=None, end_step=None):
    """Return whether an arrival is in the half-open interval [start, end)."""
    return ((start_step is None or arrival_step >= start_step) and
            (end_step is None or arrival_step < end_step))


def build_job_queue(jobs_csv=JOBS_CSV, nrel_root=NREL_ROOT,
                    job_types=("training",), subsample=None, seed=0,
                    max_arrival_step=None, min_arrival_step=None,
                    arrival_end_step=None, verbose=True):
    """Return a deterministic, arrival-sorted list[Job].

    job_types: which Alibaba job_type values to admit (training has real NREL
      gang power profiles -> the gang jobs of interest). Inference jobs map to
      single-node inference profiles if included.
    subsample: keep fraction (0,1] of admitted jobs (hash-deterministic) to
      shape cluster load; None = keep all.
    """
    if verbose:
        print(f"[inject] csv={jobs_csv}")
    df = pd.read_csv(jobs_csv)
    reg = M.build_nrel_registry(nrel_root)
    dur_map = _load_duration_map()
    if verbose:
        print(f"[inject] {len(df)} workloads, {len(reg)} NREL profiles, "
              f"{len(dur_map)} durations")

    trace_cache = {}   # profile_path -> (t, per_node_W) to avoid re-reading parquet
    jobs = []
    n_skip_fit = n_skip_type = n_skip_match = 0

    for _, row in df.iterrows():
        jtype = str(row["job_type"])
        if jtype not in job_types:
            n_skip_type += 1
            continue
        gpus = row["total_gpu_request"]
        if pd.isna(gpus) or gpus <= 0:
            n_skip_fit += 1
            continue
        num_nodes = int(np.ceil(float(gpus) / GPUS_PER_NODE))
        if num_nodes < 1 or num_nodes > MAX_JOB_NODES:
            n_skip_fit += 1
            continue

        wid = str(row["workload_id"])
        arrival_step = (int(row["arrival_hour"]) * STEPS_PER_HOUR
                        + (_hash_int(wid) % STEPS_PER_HOUR))
        if not _arrival_step_in_window(
                arrival_step, min_arrival_step, arrival_end_step):
            continue
        # Preserve the legacy inclusive cutoff for existing callers.
        if max_arrival_step is not None and arrival_step > max_arrival_step:
            continue
        if subsample is not None and (_hash_int(wid) % 10000) >= int(subsample * 10000):
            continue

        job_rec = {"workload_id": wid, "gpu_sum": float(gpus),
                   "job_type": jtype, "model_type": row["model_type"]}
        cand, primary, meta = M.match_job(job_rec, reg)
        if primary is None:
            n_skip_match += 1
            continue
        ppath = primary["path"]        # registry rows expose the parquet under 'path'
        snap = int(meta["nodes"])

        if ppath not in trace_cache:
            t, pn = M.per_node_power(ppath, snap)
            trace_cache[ppath] = np.ascontiguousarray(pn, dtype=np.float64)
        per_node = trace_cache[ppath]

        dur_h = dur_map.get(wid, None)
        if dur_h is None or not np.isfinite(dur_h) or dur_h <= 0:
            # fallback: one NREL period length (min meaningful run), never 0
            dur_h = max(len(per_node) * DT_SEC / 3600.0, INTERVAL_SEC / 3600.0)
        dur_steps = max(1, int(round(dur_h * 3600.0 / INTERVAL_SEC)))

        jobs.append(Job(
            job_id=wid, arrival_step=arrival_step, num_nodes=min(num_nodes, NUM_HOSTS),
            duration_steps=dur_steps, per_node_trace_w=per_node, dt_sec=DT_SEC,
            priority=PRIORITY.get(jtype, 0.5), model_type=str(row["model_type"]),
            job_type=jtype, profile_path=ppath, gpu_request=float(gpus)))

    jobs.sort(key=lambda j: (j.arrival_step, j.job_id))
    if verbose:
        print(f"[inject] built {len(jobs)} jobs "
              f"(skip: fit={n_skip_fit} type={n_skip_type} match={n_skip_match}); "
              f"distinct profiles={len(trace_cache)}")
        if jobs:
            print(f"[inject] arrival_step range {jobs[0].arrival_step}..{jobs[-1].arrival_step}")
    return jobs


if __name__ == "__main__":
    q = build_job_queue(subsample=None, max_arrival_step=48)
    print(f"\nFirst 12 jobs:")
    for j in q[:12]:
        print(" ", j)
    if q:
        nodes = np.array([j.num_nodes for j in q])
        peaks = np.array([j.peak_per_node_w for j in q])
        durs = np.array([j.duration_steps for j in q])
        print(f"\nnodes: min={nodes.min()} max={nodes.max()} "
              f"mean={nodes.mean():.1f}")
        print(f"peak/node W: min={peaks.min():.0f} max={peaks.max():.0f} "
              f"mean={peaks.mean():.0f}  (rack budget/4 = {12248/1.2/4:.0f})")
        print(f"duration steps: min={durs.min()} max={durs.max()} "
              f"median={np.median(durs):.0f}")
