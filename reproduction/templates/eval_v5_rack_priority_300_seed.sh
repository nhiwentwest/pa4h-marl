#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
cd "$(dirname "$0")/.."

seed="${1:?usage: eval_v5_rack_priority_300_seed.sh SEED (2-5)}"
case "$seed" in 2|3|4|5) ;; *) echo "seed must be 2-5" >&2; exit 2;; esac

root="outputs/v5_rack_priority_300_seed${seed}"
export PYBIN="${PYBIN:-/teamspace/studios/this_studio/c/.venv/bin/python3}"
export PYTHONPATH="python:vendor${PYTHONPATH:+:$PYTHONPATH}"
export NREL_ROOT="${NREL_ROOT:-/teamspace/studios/this_studio/pa4h_data/extracted}"
export POD_HOURLY_JOBS="${POD_HOURLY_JOBS:-/teamspace/studios/this_studio/pa4h_data/alibaba_v2020/pod_hourly_jobs.csv}"
export GANG_CHECKPOINT_DIR="$root/snapshots/ep300"
export SEED="$seed" NUM_HOSTS=64 USE_STGNN=1 USE_HISTORY_MLP=0 RACK_HISTORY_LEN=6
export STEPS=120 WORKLOAD_WINDOW_HOURS=10 A1_VIOL_WEIGHT=8
export W_SERVE=1 W_ENERGY=0.3 W_VIOL=8 W_WAIT=1 W_CKPT=0.15 W_SLA_BASE=0.5 W_DEADLINE=2
export SLA_TARGET_ADMISSION=0.2 SLA_TARGET_RESTART=0.2 SLA_TARGET_COMPLETION=0.2 SLA_DUAL_LR=0.1
export SLA_MAX_WAIT_STEPS=12 SLA_MAX_RESTART_WAIT_STEPS=12 SLA_COMPLETION_GRACE_STEPS=12 SLA_DROP_SEVERITY=2
export NET_COMM_TAX=0.3 RACK_OVERSUB=1.2 GANG_LINK_MBPS=1000 NETWORK_BACKGROUND_UTIL=0
export FEAS_MARGIN=0.98 INTERVAL_SEC=300 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

"$PYBIN" scripts/verify_v5_stage.py "$GANG_CHECKPOINT_DIR" 300
sha256sum -c "$root/source_sha256.txt" > /dev/null
sha256sum -c "$root/workload_sha256.txt" > /dev/null
sha256sum -c "$root/nrel_sha256.txt" > /dev/null
sha256sum -c "$root/harness_sha256.txt" > /dev/null

mkdir -p "$root/eval/ep300"
idx=0
for case in 980:0.01:1pct 990:0.01:1pct 1010:0.01:1pct 980:0.05:5pct 990:0.05:5pct; do
    IFS=: read -r start sub label <<< "$case"
    export WORKLOAD_START_HOUR="$start" SUBSAMPLE="$sub" BRIDGE_PORT="$((26100 + seed * 10 + idx))"
    stem="$root/eval/ep300/validation_${start}_${label}"
    export GANG_EVAL_CSV="${stem}.csv" GANG_BRIDGE_LOG="${stem}.bridge.log"
    if test -f "${stem}.complete"; then
        test -s "${stem}.csv"
        echo "seed $seed already completed $start/$label"
    else
        echo "seed $seed evaluating $start/$label"
        bash scripts/run_eval.sh --tag gang --decoder sequence >> "${stem}.log" 2>&1
        test -s "${stem}.csv"
        touch "${stem}.complete"
        echo "seed $seed completed $start/$label"
    fi
    idx="$((idx + 1))"
done
