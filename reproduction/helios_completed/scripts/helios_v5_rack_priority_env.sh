#!/usr/bin/env bash
set -euo pipefail

helios_v5_export_common_env() {
    local seed="${1:?usage: helios_v5_export_common_env SEED (1-5)}"
    case "$seed" in 1|2|3|4|5) ;; *) echo "seed must be 1-5" >&2; return 2;; esac

    export PA4H_DATA_BASE="${PA4H_DATA_BASE:-/teamspace/studios/this_studio/pa4h_data}"
    export PYBIN="${PYBIN:-/teamspace/studios/this_studio/c/.venv/bin/python3}"
    export PYTHONPATH="python:vendor${PYTHONPATH:+:$PYTHONPATH}"
    export NREL_ROOT="${NREL_ROOT:-${PA4H_DATA_BASE}/extracted}"
    export POD_HOURLY_JOBS="${POD_HOURLY_JOBS:-${PA4H_DATA_BASE}/helios_earth_pod_hourly_jobs.csv}"

    export SEED="$seed" NUM_EPISODES=300 NUM_HOSTS=64 MAX_STEPS=180 GANG_MAX_STEPS=180
    export BATCH_EPISODES=4 USE_STGNN=1 USE_HISTORY_MLP=0 RACK_HISTORY_LEN=6 USE_CF_SUPERVISION=1
    export SUBSAMPLE=0.50 WORKLOAD_START_HOUR=1000 WORKLOAD_WINDOW_HOURS=10
    export LR=0.0003 PPO_EPOCHS=4 GAMMA=0.99 GAE_LAMBDA=0.95 EARLY_STOP_PATIENCE=0
    export A1_VIOL_WEIGHT=8 RELIEF_TEAM_ADV_COEF=0.05 A4_TEAM_ADV_COEF=0.15 CF_AUX_COEF=1
    export W_SERVE=1 W_ENERGY=0.3 W_VIOL=8 W_WAIT=1 W_CKPT=0.15 W_SLA_BASE=0.5 W_DEADLINE=2
    export SLA_TARGET_ADMISSION=0.2 SLA_TARGET_RESTART=0.2 SLA_TARGET_COMPLETION=0.2 SLA_DUAL_LR=0.1
    export SLA_MAX_WAIT_STEPS=12 SLA_MAX_RESTART_WAIT_STEPS=12 SLA_COMPLETION_GRACE_STEPS=12 SLA_DROP_SEVERITY=2
    export NET_COMM_TAX=0.3 RACK_OVERSUB=1.2 GANG_LINK_MBPS=1000 NETWORK_BACKGROUND_UTIL=0
    export FEAS_MARGIN=0.98 INTERVAL_SEC=300 START_TENSORBOARD=0 OBS_ENABLED=0
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    export HELIOS_OUTPUT_ROOT="${HELIOS_OUTPUT_BASE:-outputs}/helios_conditioned_300_seed${seed}"
    export DACN_DATA="${HELIOS_DATA_CACHE_BASE:-${HELIOS_OUTPUT_ROOT}/data_cache}"
}
