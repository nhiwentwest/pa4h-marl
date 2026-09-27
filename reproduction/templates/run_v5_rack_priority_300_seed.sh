#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
cd "$(dirname "$0")/.."

seed="${1:?usage: run_v5_rack_priority_300_seed.sh SEED (2-5)}"
case "$seed" in 2|3|4|5) ;; *) echo "seed must be 2-5" >&2; exit 2;; esac

root="outputs/v5_rack_priority_300_seed${seed}"
live="$root/live"
snap="$root/snapshots/ep300"
reference="${SOURCE_REFERENCE:-outputs/v5_global_relief_seed0}"
protocol_template="scripts/v5_rack_priority_300_seed1_protocol.json"

export PYBIN="${PYBIN:-/teamspace/studios/this_studio/c/.venv/bin/python3}"
export PYTHONPATH="python:vendor${PYTHONPATH:+:$PYTHONPATH}"
export NREL_ROOT="${NREL_ROOT:-/teamspace/studios/this_studio/pa4h_data/extracted}"
export POD_HOURLY_JOBS="${POD_HOURLY_JOBS:-/teamspace/studios/this_studio/pa4h_data/alibaba_v2020/pod_hourly_jobs.csv}"
export SEED="$seed" NUM_EPISODES=300 NUM_HOSTS=64 MAX_STEPS=120 GANG_MAX_STEPS=120
export BATCH_EPISODES=4 USE_STGNN=1 USE_HISTORY_MLP=0 RACK_HISTORY_LEN=6 USE_CF_SUPERVISION=1
export SUBSAMPLE=0.03 WORKLOAD_START_HOUR=1000 WORKLOAD_WINDOW_HOURS=10
export LR=0.0003 PPO_EPOCHS=4 GAMMA=0.99 GAE_LAMBDA=0.95 EARLY_STOP_PATIENCE=0
export A1_VIOL_WEIGHT=8 RELIEF_TEAM_ADV_COEF=0.05 A4_TEAM_ADV_COEF=0.15 CF_AUX_COEF=1
export W_SERVE=1 W_ENERGY=0.3 W_VIOL=8 W_WAIT=1 W_CKPT=0.15 W_SLA_BASE=0.5 W_DEADLINE=2
export SLA_TARGET_ADMISSION=0.2 SLA_TARGET_RESTART=0.2 SLA_TARGET_COMPLETION=0.2 SLA_DUAL_LR=0.1
export SLA_MAX_WAIT_STEPS=12 SLA_MAX_RESTART_WAIT_STEPS=12 SLA_COMPLETION_GRACE_STEPS=12 SLA_DROP_SEVERITY=2
export NET_COMM_TAX=0.3 RACK_OVERSUB=1.2 GANG_LINK_MBPS=1000 NETWORK_BACKGROUND_UTIL=0
export FEAS_MARGIN=0.98 INTERVAL_SEC=300 START_TENSORBOARD=0 OBS_ENABLED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 BRIDGE_PORT="$((26090 + seed))"
export GANG_CHECKPOINT_DIR="$live" GANG_CSV="$live/train.csv" TRAIN_STATE_PATH="$live/training_state.pt"
unset VALIDATION_START_HOUR VALIDATION_WINDOW_HOURS RESUME_PATH

test -f "$POD_HOURLY_JOBS" && test -d "$NREL_ROOT"
sha256sum -c "$reference/source_sha256.txt" > /dev/null
sha256sum -c "$reference/workload_sha256.txt" > /dev/null
sha256sum -c "$reference/nrel_sha256.txt" > /dev/null
mkdir -p "$live" "$root/snapshots"

for name in source_sha256.txt workload_sha256.txt nrel_sha256.txt; do
    if test -f "$root/$name"; then
        cmp "$reference/$name" "$root/$name"
    else
        cp "$reference/$name" "$root/$name"
    fi
done

"$PYBIN" - "$seed" "$protocol_template" "$root/protocol.json" <<'PY'
import json
import sys
from pathlib import Path

seed, template, target = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if target.exists():
    data = json.loads(target.read_text())
    if data.get("seed") != seed:
        raise SystemExit(f"protocol seed mismatch in {target}")
else:
    data = json.loads(template.read_text())
    data["seed"] = seed
    data["status"] = "multi-seed-rack-priority-run"
    target.write_text(json.dumps(data, indent=2) + "\n")
PY

harness="$root/harness_sha256.txt"
if ! test -f "$harness"; then
    sha256sum scripts/run_v5_rack_priority_300_seed.sh \
        scripts/eval_v5_rack_priority_300_seed.sh \
        scripts/run_v5_rack_priority_300_batch.sh \
        "$protocol_template" > "$harness"
fi
sha256sum -c "$harness" > /dev/null

if ! test -f "$root/effective_config.json"; then
    "$PYBIN" - <<'PY' > "$root/effective_config.json"
import json
import os
import platform
import torch
import marl_gang_train as trainer

keys = "SEED NUM_EPISODES NUM_HOSTS MAX_STEPS GANG_MAX_STEPS BATCH_EPISODES USE_STGNN USE_HISTORY_MLP RACK_HISTORY_LEN USE_CF_SUPERVISION SUBSAMPLE WORKLOAD_START_HOUR WORKLOAD_WINDOW_HOURS LR PPO_EPOCHS GAMMA GAE_LAMBDA EARLY_STOP_PATIENCE A1_VIOL_WEIGHT RELIEF_TEAM_ADV_COEF A4_TEAM_ADV_COEF CF_AUX_COEF W_SERVE W_ENERGY W_VIOL W_WAIT W_CKPT W_SLA_BASE W_DEADLINE SLA_TARGET_ADMISSION SLA_TARGET_RESTART SLA_TARGET_COMPLETION SLA_DUAL_LR SLA_MAX_WAIT_STEPS SLA_MAX_RESTART_WAIT_STEPS SLA_COMPLETION_GRACE_STEPS SLA_DROP_SEVERITY NET_COMM_TAX RACK_OVERSUB GANG_LINK_MBPS NETWORK_BACKGROUND_UTIL FEAS_MARGIN INTERVAL_SEC".split()
constants = "ENT_COEF ENT_FLOOR_PENALTY ENT_ADAPT_LR ENT_COEF_MAX RELIEF_SLA_WEIGHT REWARD_SCALE CF_GAP_EPS CLIP".split()
print(json.dumps({"env": {k: os.environ[k] for k in keys},
                  "python": platform.python_version(), "torch": torch.__version__,
                  "trainer_defaults": {k: getattr(trainer, k) for k in constants},
                  "reference": "v5 300-episode rack-priority candidate"}, indent=2))
PY
fi
"$PYBIN" - "$root/effective_config.json" "$seed" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
expected_seed = int(sys.argv[2])
env = config.get("env", {})
if (int(env.get("SEED", -1)) != expected_seed or
        int(env.get("NUM_EPISODES", -1)) != 300 or
        float(env.get("A1_VIOL_WEIGHT", -1)) != 8 or
        float(env.get("W_VIOL", -1)) != 8):
    raise SystemExit("effective config does not match this seed-300 rack-priority run")
PY

if test -d "$snap"; then
    "$PYBIN" scripts/verify_v5_stage.py "$snap" 300
    echo "seed $seed already has verified ep300 snapshot"
    exit 0
fi

training_done=0
if test -f "$live/training_state.pt"; then
    saved_episode=$("$PYBIN" - "$live/training_state.pt" <<'PY'
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(state["episode"]))
PY
)
    if test "$saved_episode" -eq 300; then
        "$PYBIN" scripts/verify_v5_stage.py "$live" 300
        training_done=1
    elif test "$saved_episode" -gt 300; then
        echo "training state exceeds episode target: $saved_episode" >&2
        exit 1
    else
        export RESUME=1
    fi
else
    if test -f "$live/train.csv" || compgen -G "$live/*_gang.pt" > /dev/null; then
        echo "live artifacts exist without training_state.pt; inspect before resuming" >&2
        exit 1
    fi
    export RESUME=0
fi

if test "$training_done" -eq 0; then
    bash scripts/run_gang.sh >> "$live/train.log" 2>&1
    "$PYBIN" scripts/verify_v5_stage.py "$live" 300
fi

tmp="$root/snapshots/.ep300.tmp.$$"
mkdir "$tmp"
cp "$live"/*_gang.pt "$live/training_state.pt" "$live/sla_multipliers.json" \
   "$live/metric_manifest.json" "$live/deployment_decoder.json" \
   "$live/train.csv" "$live/training_resources.json" "$tmp/"
cp "$root/source_sha256.txt" "$root/workload_sha256.txt" "$root/nrel_sha256.txt" \
   "$root/harness_sha256.txt" "$root/protocol.json" "$root/effective_config.json" "$tmp/"
"$PYBIN" scripts/verify_v5_stage.py "$tmp" 300
mv "$tmp" "$snap"
echo "seed $seed rack-priority ep300 snapshot complete: $snap"
