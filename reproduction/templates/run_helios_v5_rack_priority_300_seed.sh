#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
cd "$(dirname "$0")/.."

seed="${1:?usage: run_helios_v5_rack_priority_300_seed.sh SEED (1-5)}"
case "$seed" in 1|2|3|4|5) ;; *) echo "seed must be 1-5" >&2; exit 2;; esac

source scripts/helios_v5_rack_priority_env.sh
helios_v5_export_common_env "$seed"
export BRIDGE_PORT="$((26200 + seed))"
root="$HELIOS_OUTPUT_ROOT"
live="$root/live"
snap="$root/snapshots/ep300"
reference="outputs/helios_conditioned_source_reference"
protocol_template="scripts/helios_v5_rack_priority_protocol.json"

unset VALIDATION_START_HOUR VALIDATION_WINDOW_HOURS RESUME_PATH
export GANG_CHECKPOINT_DIR="$live" GANG_CSV="$live/train.csv" TRAIN_STATE_PATH="$live/training_state.pt"

test -f "$POD_HOURLY_JOBS" || { echo "Helios workload missing: $POD_HOURLY_JOBS" >&2; exit 1; }
test -f "$NREL_ROOT/01_aggregated_datasets/training/metadata.csv" || {
    echo "NLR training metadata missing under $NREL_ROOT" >&2
    exit 1
}
test -f "$reference/source_sha256.txt" || { echo "source reference missing: $reference" >&2; exit 1; }
test -f "$reference/nrel_sha256.txt" || { echo "NLR reference manifest missing: $reference" >&2; exit 1; }

sha256sum -c "$reference/source_sha256.txt" > /dev/null
sha256sum -c "$reference/nrel_sha256.txt" > /dev/null
mkdir -p outputs
exec 9>"outputs/.helios_v5_rack_priority_300_seed${seed}.lock"
if ! flock -n 9; then
    echo "Helios v5 seed $seed is already running" >&2
    exit 1
fi

mkdir -p "$live" "$root/snapshots"
for name in source_sha256.txt nrel_sha256.txt; do
    if test -f "$root/$name"; then
        cmp "$reference/$name" "$root/$name"
    else
        cp "$reference/$name" "$root/$name"
    fi
done

actual_workload="$(sha256sum "$POD_HOURLY_JOBS")"
if test -f "$root/workload_sha256.txt"; then
    test "$(<"$root/workload_sha256.txt")" = "$actual_workload" || {
        echo "Helios workload checksum changed for $root" >&2
        exit 1
    }
else
    printf '%s\n' "$actual_workload" > "$root/workload_sha256.txt"
fi
sha256sum -c "$root/workload_sha256.txt" > /dev/null

"$PYBIN" - "$seed" "$protocol_template" "$root/protocol.json" <<'PY'
import json
import sys
from pathlib import Path

seed, template, target = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
protocol = json.loads(template.read_text())
protocol["seed"] = seed
protocol["status"] = "pilot" if seed == 1 else "five-seed-follow-on"
if target.exists():
    existing = json.loads(target.read_text())
    if existing != protocol:
        raise SystemExit(f"protocol mismatch in {target}")
else:
    target.write_text(json.dumps(protocol, indent=2) + "\n")
PY

harness="$root/harness_sha256.txt"
if ! test -f "$harness"; then
    sha256sum scripts/helios_v5_rack_priority_env.sh \
        scripts/run_helios_v5_rack_priority_300_seed.sh \
        scripts/eval_helios_v5_rack_priority_300_seed.sh \
        scripts/helios_v5_rack_priority_protocol.json \
        scripts/run_gang.sh scripts/run_eval.sh scripts/verify_v5_stage.py > "$harness"
fi
sha256sum -c "$harness" > /dev/null

if ! test -f "$root/effective_config.json"; then
    "$PYBIN" - "$root/effective_config.json" <<'PY'
import json
import os
import platform
import sys
import torch
import marl_gang_train as trainer

keys = "SEED NUM_EPISODES NUM_HOSTS MAX_STEPS GANG_MAX_STEPS BATCH_EPISODES USE_STGNN USE_HISTORY_MLP RACK_HISTORY_LEN USE_CF_SUPERVISION SUBSAMPLE WORKLOAD_START_HOUR WORKLOAD_WINDOW_HOURS LR PPO_EPOCHS GAMMA GAE_LAMBDA EARLY_STOP_PATIENCE A1_VIOL_WEIGHT RELIEF_TEAM_ADV_COEF A4_TEAM_ADV_COEF CF_AUX_COEF W_SERVE W_ENERGY W_VIOL W_WAIT W_CKPT W_SLA_BASE W_DEADLINE SLA_TARGET_ADMISSION SLA_TARGET_RESTART SLA_TARGET_COMPLETION SLA_DUAL_LR SLA_MAX_WAIT_STEPS SLA_MAX_RESTART_WAIT_STEPS SLA_COMPLETION_GRACE_STEPS SLA_DROP_SEVERITY NET_COMM_TAX RACK_OVERSUB GANG_LINK_MBPS NETWORK_BACKGROUND_UTIL FEAS_MARGIN INTERVAL_SEC POD_HOURLY_JOBS NREL_ROOT BRIDGE_PORT".split()
constants = "ENT_COEF ENT_FLOOR_PENALTY ENT_ADAPT_LR ENT_COEF_MAX RELIEF_SLA_WEIGHT REWARD_SCALE CF_GAP_EPS CLIP".split()
record = {
    "env": {key: os.environ[key] for key in keys},
    "python": platform.python_version(),
    "torch": torch.__version__,
    "trainer_defaults": {key: getattr(trainer, key) for key in constants},
    "source_reference": "outputs/helios_conditioned_source_reference",
    "protocol": "scripts/helios_v5_rack_priority_protocol.json",
}
with open(sys.argv[1], "w") as f:
    json.dump(record, f, indent=2)
    f.write("\n")
PY
fi
"$PYBIN" - "$root/effective_config.json" "$seed" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text())
env = record.get("env", {})
if (int(env.get("SEED", -1)) != int(sys.argv[2]) or
        int(env.get("NUM_EPISODES", -1)) != 300 or
        int(env.get("MAX_STEPS", -1)) != 180 or
        float(env.get("SUBSAMPLE", -1)) != 0.50 or
        float(env.get("A1_VIOL_WEIGHT", -1)) != 8 or
        float(env.get("W_VIOL", -1)) != 8):
    raise SystemExit("effective config does not match this Helios v5 run")
PY

if test -d "$snap"; then
    "$PYBIN" scripts/verify_v5_stage.py "$snap" 300
else
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
    test ! -e "$tmp" || { echo "stale temporary snapshot: $tmp" >&2; exit 1; }
    mkdir "$tmp"
    cp "$live"/*_gang.pt "$live/training_state.pt" "$live/sla_multipliers.json" \
       "$live/metric_manifest.json" "$live/deployment_decoder.json" \
       "$live/train.csv" "$live/training_resources.json" "$tmp/"
    cp "$root/source_sha256.txt" "$root/workload_sha256.txt" "$root/nrel_sha256.txt" \
       "$root/harness_sha256.txt" "$root/protocol.json" "$root/effective_config.json" "$tmp/"
    "$PYBIN" scripts/verify_v5_stage.py "$tmp" 300
    mv "$tmp" "$snap"
fi

bash scripts/eval_helios_v5_rack_priority_300_seed.sh "$seed"
echo "Helios v5 rack-priority seed $seed train+validation complete: $root"
