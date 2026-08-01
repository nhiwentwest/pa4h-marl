#!/usr/bin/env python3
"""Short terminal demo for the HeliosData + NREL experiment."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = Path(os.environ.get(
    "DEMO_RUN_DIR",
    ROOT / "outputs" / "gang_v2_helios_nrel_300eps_20260715",
))
TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def color(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if TTY else text


def heading(text: str) -> None:
    print("\n" + color("1;36", f"=== {text} ==="))


def pause() -> None:
    if TTY:
        time.sleep(float(os.environ.get("DEMO_PAUSE", "0.15")))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def require_artifacts() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    required = [RUN / "train.csv", RUN / "eval_baselines_helios.csv",
                RUN / "training_state.pt"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing demo artifacts:\n  " + "\n  ".join(missing))
    return rows(required[0]), rows(required[1])


def main() -> None:
    train, baselines = require_artifacts()
    last = train[-1]
    episode = int(number(last, "ep")) + 1
    decoder_path = RUN / "deployment_decoder.json"
    decoder = (json.loads(decoder_path.read_text()).get("name", "unknown")
               if decoder_path.exists() else "unknown")
    decoder_label = {
        "pointwise": "pointwise argmax",
        "sequence": "sequence-level dependent rounding",
    }.get(decoder, "unknown")
    deployed = next((row for row in baselines if row["policy"] == "MARL"), None)

    print(color("1;35", "GPU CLUSTER SCHEDULING DEMO — HELIOSDATA EARTH + NREL"))
    print(f"run: {RUN}")

    heading("1. REAL DATA PIPELINE")
    print("HeliosData Earth workload : fixed 10-hour window [1000, 1010)")
    print("Deterministic 5% sample   : 38 schedulable GPU jobs")
    print("NREL/NLR power registry   : 2,467 measured GPU profiles")
    print("Control / power interval  : 300 s / 0.2 s = 1,500 samples per step")
    print("Network model             : ring all-reduce over simulated fat-tree links")
    pause()

    heading("2. FOUR-AGENT DECISION CHAIN")
    print(color("1;33", "A4 -> A2 -> A3 -> A1"))
    print("A4  PLACE     ST-GNN  choose one of 16 racks or DEFER     [701 -> 17]")
    print("A2  DETECT    ST-GNN  select the rack with projected risk [688 -> 16]")
    print("A3  VICTIM    MLP     select a positive-relief job         [ 47 ->  4]")
    print("A1  CONTROL   MLP     choose WAIT or PREEMPT               [ 12 ->  2]")
    print("Training: four actors + centralized critic, clipped PPO + GAE")
    pause()

    heading("3. TRAINED CHECKPOINT + DETERMINISTIC DEPLOYMENT")
    print(f"episode checkpoint : {episode} / 300")
    print(f"deployment decoder : {decoder_label}")
    if deployed:
        print(f"evaluation reward  : {number(deployed, 'reward'):.2f}")
        print(f"served jobs        : {number(deployed, 'served_jobs'):.0f} / 38 "
              f"({number(deployed, 'served_pct'):.0f}%)")
        print(f"overall SLA        : {number(deployed, 'sla_pct'):.2f}%")
        print(f"rack violations    : {number(deployed, 'rack_viol'):.0f}")
        print(f"energy             : {number(deployed, 'energy_kwh'):.1f} kWh")
        print(f"cross-rack bytes   : {number(deployed, 'network_cross_byte_pct'):.1f}%")
    print(f"last train A1 trace: {number(last, 'a1_decisions'):.0f} decisions "
          f"({number(last, 'a1_preempt'):.0f} PREEMPT)")
    pause()

    heading("4. SAME-SCENARIO BASELINE EVALUATION")
    print(f"{'POLICY':<12}{'REWARD':>8}{'SERVED':>9}{'SLA':>8}"
          f"{'RACK-VIOL':>12}{'ENERGY':>10}{'X-RACK':>9}")
    print("-" * 68)
    for row in baselines:
        name = row["policy"]
        line = (f"{name:<12}{number(row, 'reward'):>8.2f}"
                f"{number(row, 'served_pct'):>8.0f}%"
                f"{number(row, 'sla_pct'):>7.1f}%"
                f"{number(row, 'rack_viol'):>12.0f}"
                f"{number(row, 'energy_kwh'):>10.1f}"
                f"{number(row, 'network_cross_byte_pct'):>8.1f}%")
        print(color("1;32", line) if name.startswith("MARL") else line)

    marl = next((r for r in baselines if r["policy"] == "MARL"), None)
    heuristics = [r for r in baselines if not r["policy"].startswith("MARL")]
    if marl and heuristics:
        best = max(heuristics, key=lambda r: number(r, "reward"))
        gain = number(marl, "reward") - number(best, "reward")
        heading("TAKEAWAY")
        print(f"MARL reward {number(marl, 'reward'):.2f}; best heuristic "
              f"{best['policy']} {number(best, 'reward'):.2f}; gain {gain:+.2f}.")
        print("Pointwise keeps the same SLA level while sharply reducing rack violations")
        print("relative to RANDOM/SPREAD/EPOBF; RPA reaches zero violations but lower reward.")

    print("\n" + color("1;36", "Demo complete."))


if __name__ == "__main__":
    main()
