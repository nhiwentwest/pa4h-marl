#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export SEED="${SEED:-1}"
export NUM_EPISODES="${NUM_EPISODES:-300}"
export SUBSAMPLE="${SUBSAMPLE:-0.05}"
export MAX_STEPS="${MAX_STEPS:-120}"
export WORKLOAD_START_HOUR="${WORKLOAD_START_HOUR:-1000}"
export WORKLOAD_WINDOW_HOURS="${WORKLOAD_WINDOW_HOURS:-10}"
export POD_HOURLY_JOBS="${POD_HOURLY_JOBS:-data/helios_earth_pod_hourly_jobs.csv}"
export USE_STGNN="${USE_STGNN:-1}"
export START_TENSORBOARD="${START_TENSORBOARD:-1}"
export OBS_ENABLED="${OBS_ENABLED:-1}"
export GANG_CHECKPOINT_DIR="${GANG_CHECKPOINT_DIR:-outputs/helios_seed${SEED}}"
export GANG_CSV="${GANG_CSV:-${GANG_CHECKPOINT_DIR}/train.csv}"

bash scripts/run_gang.sh
