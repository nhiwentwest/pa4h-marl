"""Measure full scheduler decision latency on a fixed workload replay.

Run while GangBridge is listening, with the same workload/checkpoint environment
variables as eval_gang.py. The simulator advance is timed separately.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import eval_gang as evaluation
from eval_gang import build_eval_env, load_agents, load_sla_multipliers
from marl_gang_train import gang_step, make_deterministic_act


def summarize(rows):
    out = {}
    for name, subset in (("all", rows),
                         ("active", [r for r in rows if r["actions"] > 0]),
                         ("empty", [r for r in rows if r["actions"] == 0])):
        out[name] = {"n": len(subset)}
        for field in ("decision_ms", "counterfactual_ms", "actor_decoder_ms",
                      "other_decision_ms", "advance_ms"):
            vals = [r[field] for r in subset]
            out[name][field] = ({"p50": float(np.percentile(vals, 50)),
                                 "p95": float(np.percentile(vals, 95)),
                                 "p99": float(np.percentile(vals, 99)),
                                 "max": float(max(vals))} if vals else None)
    return out


def run(output_dir, decoder, warmup, tag):
    evaluation.TAG = tag
    env = build_eval_env()
    load_sla_multipliers(env)
    agents, missing = load_agents(env)
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")

    # Exercise model, Py4J, and NLR cache before the measured episode.
    warm_act = make_deterministic_act(agents, a4_mode=decoder)
    for t in range(min(warmup, env.max_steps)):
        gang_step(env, agents, t, warm_act)
        env.advance()
    env.reset()
    act = make_deterministic_act(agents, a4_mode=decoder)
    raw_counterfactual = env.a4_counterfactual_details
    elapsed = {"counterfactual_ms": 0.0, "actor_decoder_ms": 0.0,
               "actions": 0, "a2_actions": 0, "a4_actions": 0}

    def counterfactual(job):
        begin = time.perf_counter_ns()
        result = raw_counterfactual(job)
        elapsed["counterfactual_ms"] += (time.perf_counter_ns() - begin) / 1e6
        return result

    def timed_act(*args, **kwargs):
        begin = time.perf_counter_ns()
        result = act(*args, **kwargs)
        elapsed["actor_decoder_ms"] += (time.perf_counter_ns() - begin) / 1e6
        elapsed["actions"] += 1
        if args[0] == "a2":
            elapsed["a2_actions"] += 1
        elif args[0] == "a4":
            elapsed["a4_actions"] += 1
        return result

    env.a4_counterfactual_details = counterfactual
    rows = []
    for t in range(env.max_steps):
        elapsed.update(counterfactual_ms=0.0, actor_decoder_ms=0.0,
                       actions=0, a2_actions=0, a4_actions=0)
        pending, running = len(env.pending), len(env.running)
        begin = time.perf_counter_ns()
        gang_step(env, agents, t, timed_act)
        decision_ms = (time.perf_counter_ns() - begin) / 1e6
        begin = time.perf_counter_ns()
        env.advance()
        advance_ms = (time.perf_counter_ns() - begin) / 1e6
        rows.append(dict(step=t, pending=pending, running=running,
                         actions=elapsed["actions"],
                         a2_actions=elapsed["a2_actions"],
                         a4_actions=elapsed["a4_actions"],
                         decision_ms=decision_ms,
                         counterfactual_ms=elapsed["counterfactual_ms"],
                         actor_decoder_ms=elapsed["actor_decoder_ms"],
                         other_decision_ms=max(0.0, decision_ms
                                               - elapsed["counterfactual_ms"]
                                               - elapsed["actor_decoder_ms"]),
                         advance_ms=advance_ms))
        if env.done:
            break
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "steps.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"decoder": decoder, "checkpoint_tag": tag,
              "workload_csv": os.environ.get("POD_HOURLY_JOBS"),
              "checkpoint_dir": os.environ.get("GANG_CHECKPOINT_DIR"),
              "hosts": env.num_hosts, "racks": env.num_racks,
              "jobs": len(env._all_jobs), "warmup_steps": warmup,
              "a2_decisions": sum(r["a2_actions"] for r in rows),
              "a4_decisions": sum(r["a4_actions"] for r in rows),
              "torch_threads": torch.get_num_threads(),
              "latency_ms": summarize(rows)}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decoder", choices=("sequence", "pointwise"), default="sequence")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tag", default="gang_best")
    args = parser.parse_args()
    run(args.output_dir, args.decoder, args.warmup, args.tag)
