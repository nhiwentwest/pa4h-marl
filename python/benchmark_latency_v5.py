"""Post-training latency probe for exact v5 A3 whole-gang credit evaluation."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import benchmark_latency as baseline
import marl_gang_train as training


def distribution(values):
    return {"p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(max(values))}


def run(output_dir, decoder, warmup, tag):
    original_credit = training.build_relief_candidate_credit
    original_step = baseline.gang_step
    elapsed = {"ms": 0.0, "calls": 0}
    step_credit = []

    def timed_credit(*args, **kwargs):
        start = time.perf_counter_ns()
        result = original_credit(*args, **kwargs)
        elapsed["ms"] += (time.perf_counter_ns() - start) / 1e6
        elapsed["calls"] += 1
        return result

    def timed_step(*args, **kwargs):
        before_ms, before_calls = elapsed["ms"], elapsed["calls"]
        result = original_step(*args, **kwargs)
        step_credit.append((elapsed["ms"] - before_ms,
                            elapsed["calls"] - before_calls))
        return result

    training.build_relief_candidate_credit = timed_credit
    baseline.gang_step = timed_step
    try:
        baseline.run(output_dir, decoder, warmup, tag)
    finally:
        training.build_relief_candidate_credit = original_credit
        baseline.gang_step = original_step

    out = Path(output_dir)
    with (out / "steps.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    measured = step_credit[-len(rows):]
    if len(measured) != len(rows):
        raise ValueError("credit instrumentation missed scheduler steps")
    for row, (ms, calls) in zip(rows, measured):
        row["relief_credit_ms"] = ms
        row["relief_candidates"] = calls
    with (out / "steps.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    values = [float(r["relief_credit_ms"]) for r in rows]
    calls = sum(int(r["relief_candidates"]) for r in rows)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["a3_global_relief_credit"] = {
        "calls": calls,
        "steps_with_candidates": sum(int(r["relief_candidates"]) > 0 for r in rows),
        "total_ms": sum(values),
        "mean_ms_per_candidate": sum(values) / calls if calls else None,
        "per_step_ms": distribution(values),
        "max_candidates_in_step": max(int(r["relief_candidates"]) for r in rows),
        "instrumentation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["a3_global_relief_credit"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decoder", choices=("sequence", "pointwise"), default="sequence")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tag", default="gang")
    args = parser.parse_args()
    run(args.output_dir, args.decoder, args.warmup, args.tag)
