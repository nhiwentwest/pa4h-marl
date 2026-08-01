"""
alibaba_nlr_matcher.py — deterministic Alibaba-job → NREL-power-profile match.

NOTE: reconstructed (2026-08-01) from the call sites in
nrel_injection_bridge.py after the original stayed on the Lightning
workspace. Contract used by the bridge:

    reg = build_nrel_registry(nrel_root)           # len()-able registry
    cand, primary, meta = match_job(job_rec, reg)  # primary None if no match
        # primary["path"] -> parquet path, meta["nodes"] -> profile node count
    t, per_node_w = per_node_power(path, nodes)    # trace / node count

Matching rule (as described in the paper): a job is matched by its model
family and requested node count; ties between repeats are resolved from a
hash of workload_id so the pairing is reproducible. Aggregated NREL training
profiles are per-job power (power[W] @ 0.2 s); dividing by the recorded node
count yields the per-node trace replayed by the Java bridge.
"""
import hashlib
import os

import numpy as np
import pandas as pd

# Alibaba `model_type` values are free-form; map keywords onto the two NREL
# training families. Unknown families fall back to the full registry so every
# job still receives a real measured profile deterministically.
_FAMILY_KEYWORDS = {
    "llama": "llama",
    "gpt": "llama",
    "bert": "llama",
    "transformer": "llama",
    "nlp": "llama",
    "diffusion": "stable diffusion",
    "stable": "stable diffusion",
    "image": "stable diffusion",
    "cv": "stable diffusion",
    "vision": "stable diffusion",
}


def _hash_int(s):
    return int(hashlib.md5(str(s).encode()).hexdigest(), 16)


def _family_of(model_type):
    m = str(model_type or "").lower()
    for key, family in _FAMILY_KEYWORDS.items():
        if key in m:
            return family
    return None


def build_nrel_registry(nrel_root):
    """Read the aggregated-training metadata into a list of profile rows:
    {"model", "family", "nodes", "repeat", "path"}."""
    training = os.path.join(nrel_root, "01_aggregated_datasets", "training")
    meta_csv = os.path.join(training, "metadata.csv")
    meta = pd.read_csv(meta_csv)
    rows = []
    for idx, rec in meta.iterrows():
        path = os.path.join(training, "results", f"{idx:06d}.parquet")
        if not os.path.exists(path):
            continue
        model = str(rec["model"])
        rows.append({
            "model": model,
            "family": model.lower(),
            "nodes": int(rec["nodes"]),
            "repeat": int(rec.get("repeat", 0)),
            "path": path,
        })
    return rows


def match_job(job_rec, registry):
    """Deterministically match one Alibaba job record to an NREL profile.

    job_rec: {"workload_id", "gpu_sum", "job_type", "model_type"}
    Returns (candidates, primary_row, meta) with primary_row None when the
    registry is empty.
    """
    if not registry:
        return [], None, {}
    gpus = float(job_rec.get("gpu_sum", 0.0) or 0.0)
    want_nodes = max(1, int(np.ceil(gpus / 4.0)))

    family = _family_of(job_rec.get("model_type"))
    pool = [r for r in registry if family and family in r["family"]] or list(registry)

    # Closest recorded node count (NREL training profiles exist for
    # 2/4/8/16 nodes); prefer the smallest gap, then the smaller profile.
    best_gap = min(abs(r["nodes"] - want_nodes) for r in pool)
    candidates = [r for r in pool if abs(r["nodes"] - want_nodes) == best_gap]
    candidates.sort(key=lambda r: (r["nodes"], r["repeat"], r["path"]))

    pick = _hash_int(job_rec.get("workload_id", "")) % len(candidates)
    primary = candidates[pick]
    return candidates, primary, {"nodes": primary["nodes"]}


def per_node_power(path, nodes):
    """Load one aggregated profile and divide by its node count.

    Returns (timestamps_s, per_node_power_w) as float64 arrays.
    """
    frame = pd.read_parquet(path)
    series = frame["power[W]"].dropna().astype(float)
    t = series.index.to_numpy(dtype=np.float64)
    per_node = series.to_numpy(dtype=np.float64) / max(1, int(nodes))
    return t, per_node
