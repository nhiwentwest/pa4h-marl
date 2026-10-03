#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f reproduction_env.sh ]]; then source reproduction_env.sh; fi
export PA4H_DATA_BASE="${PA4H_DATA_BASE:-$PWD/data}" PYBIN="${PYBIN:-python3}"
export POD_HOURLY_JOBS="$PWD/data/helios_completed.csv"
source scripts/helios_v5_rack_priority_env.sh
helios_v5_export_common_env 1
export NUM_EPISODES=300 A1_DELAY_CREDIT=1 A1_TEMPORAL_OBSERVATION=1 A4_BRANCH_RANK=1 QUEUE_RELIEF_ENABLED=1
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 BRIDGE_PORT=29265 BRANCH_PORT=29266 STEPS=180
export HELIOS_DATASET_REVISION=completed-diverse-v1
export HELIOS_DATASET_SHA256
HELIOS_DATASET_SHA256=$("$PYBIN" -c 'import json; print(json.load(open("scripts/helios_completed_protocol.json"))["data"]["sha256"])')
export DACN_DATA="$PWD/outputs/data_cache"
root=outputs/helios_queue_relief_fresh300_seed1
stop_after="${STOP_AFTER:-300}"
export GANG_CHECKPOINT_DIR="$root/live"
mkdir -p "$root"
exec 8>"$root/workflow.lock"
flock -n 8
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
for port in "$BRIDGE_PORT" "$BRANCH_PORT"; do
    if ss -ltn | grep -q ":$port "; then echo "port $port already in use" >&2; exit 1; fi
    java -Xmx2048m -cp 'target/classes:target/dependency/*' com.dacn.advanced.GangBridge "$port" >"$root/bridge_$port.log" 2>&1 8>&- &
    pids+=("$!");ready=0
    for _ in $(seq 1 30); do
        kill -0 "${pids[-1]}"
        if ss -ltn | grep -q ":$port "; then ready=1; break; fi
        sleep 1
    done
    [[ "$ready" == 1 ]]
done
"$PYBIN" -u scripts/train_helios_completed.py --credit immediate --seed 1 --episodes 300 \
    --protocol scripts/helios_completed_protocol.json --output "$root" --stop-after "$stop_after"
