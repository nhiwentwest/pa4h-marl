#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
cd "$(dirname "$0")/.."

seed="${1:?usage: eval_helios_v5_rack_priority_300_seed.sh SEED (1-5)}"
case "$seed" in 1|2|3|4|5) ;; *) echo "seed must be 1-5" >&2; exit 2;; esac

source scripts/helios_v5_rack_priority_env.sh
helios_v5_export_common_env "$seed"
export BRIDGE_PORT="$((26300 + seed * 10))"
export WORKLOAD_START_HOUR=1010 WORKLOAD_WINDOW_HOURS=10 STEPS=180
unset VALIDATION_START_HOUR VALIDATION_WINDOW_HOURS RESUME_PATH

root="$HELIOS_OUTPUT_ROOT"
export GANG_CHECKPOINT_DIR="$root/snapshots/ep300"
test -d "$GANG_CHECKPOINT_DIR" || { echo "ep300 snapshot missing: $GANG_CHECKPOINT_DIR" >&2; exit 1; }
"$PYBIN" scripts/verify_v5_stage.py "$GANG_CHECKPOINT_DIR" 300
sha256sum -c "$root/source_sha256.txt" > /dev/null
sha256sum -c "$root/workload_sha256.txt" > /dev/null
sha256sum -c "$root/nrel_sha256.txt" > /dev/null
sha256sum -c "$root/harness_sha256.txt" > /dev/null

eval_dir="$root/eval/ep300"
mkdir -p "$eval_dir"
idx=0
for load in 0.05 0.50; do
export SUBSAMPLE="$load"
case "$load" in 0.05) case_name=validation_1010_5pct;; 0.50) case_name=stress_1010_50pct;; esac
for decoder in sequence pointwise; do
    stem="$eval_dir/${case_name}_${decoder}"
    export GANG_EVAL_CSV="${stem}.csv" GANG_BRIDGE_LOG="${stem}.bridge.log"
    export BRIDGE_PORT="$((26300 + seed * 10 + idx))"
    if test -f "${stem}.complete"; then
        test -s "${stem}.csv"
        echo "Helios seed $seed validation $decoder already complete"
    else
        bash scripts/run_eval.sh --tag gang --decoder "$decoder" > "${stem}.log" 2>&1
        test -s "${stem}.csv"
        touch "${stem}.complete"
        echo "Helios seed $seed validation $decoder complete"
    fi
    idx=$((idx + 1))
done

done
