"""Verify a completed v5 training boundary before creating a snapshot."""
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch

from gang_env import SIMULATOR_SEMANTICS
from marl_gang_train import relief_credit_metadata


def verify(directory, target):
    root = Path(directory)
    state = torch.load(root / "training_state.pt", map_location="cpu", weights_only=False)
    if state.get("episode") != target or state.get("simulator_semantics") != SIMULATOR_SEMANTICS:
        raise ValueError("wrong training episode or simulator semantics")
    if state.get("relief_credit") != relief_credit_metadata():
        raise ValueError("training state relief_credit descriptor mismatch")
    manifest = json.loads((root / "metric_manifest.json").read_text())
    if (manifest.get("simulator_semantics") != SIMULATOR_SEMANTICS or
            manifest.get("relief_credit") != relief_credit_metadata() or
            manifest.get("numerics_checked") is not True):
        raise ValueError("metric manifest mismatch")
    with (root / "train.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if [int(row["ep"]) for row in rows] != list(range(target)):
        raise ValueError("training CSV has missing or duplicate episodes")
    for field in ("team_reward", "energy_kwh", "wall_time_s", "peak_rss_mb"):
        if any(not math.isfinite(float(row[field])) for row in rows):
            raise ValueError(f"non-finite training {field}")
    for role in ("a2", "a3"):
        if sum(int(row[f"{role}_samples"]) for row in rows) <= 0:
            raise ValueError(f"no genuine {role} training samples")
    for row in rows:
        for role in ("a2", "a3"):
            choices = int(row[f"{role}_choices"])
            correct = int(row[f"{role}_choice_correct"])
            if not 0 <= correct <= choices:
                raise ValueError(f"invalid {role} choice counters")
            accuracy = float(row[f"{role}_choice_accuracy"])
            if choices and abs(accuracy - correct / choices) > 1e-6:
                raise ValueError(f"invalid {role} choice accuracy")
            if not choices and not math.isnan(accuracy):
                raise ValueError(f"undefined {role} accuracy must be NaN")
    for name, saved in state["actors"].items():
        final = torch.load(root / f"{name}_gang.pt", map_location="cpu", weights_only=True)
        if final.keys() != saved.keys() or any(not torch.equal(final[k], saved[k]) for k in saved):
            raise ValueError(f"{name} final actor differs from training state")
        if any(not torch.isfinite(v).all() for v in final.values() if torch.is_tensor(v)):
            raise ValueError(f"non-finite {name} model")
    critic = torch.load(root / "critic_gang.pt", map_location="cpu", weights_only=True)
    if critic.keys() != state["critic"].keys() or any(
            not torch.equal(critic[k], state["critic"][k]) for k in critic):
        raise ValueError("final critic differs from training state")
    sidecar = json.loads((root / "sla_multipliers.json").read_text())
    for field, key in (("admission", "lambda_admission"),
                       ("restart", "lambda_restart"),
                       ("completion", "lambda_completion")):
        if abs(float(sidecar[field]) - float(state["sla"][key])) > 1e-9:
            raise ValueError("SLA sidecar differs from state")
    print(f"verified v5 episode {target}, {len(rows)} unique CSV rows, finite matching final weights")


if __name__ == "__main__":
    verify(sys.argv[1], int(sys.argv[2]))
