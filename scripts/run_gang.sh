#!/usr/bin/env bash
# Full gang (hướng A) training run: starts one CloudSim bridge, trains the 4-agent
# CTDE-PPO policy over GangEnv, tears the bridge down. Everything is data-derived
# (Alibaba arrivals + NREL power). Run from the repo root:
#
#   bash run_gang.sh
#
# Override anything via env, e.g.:  NUM_EPISODES=200 SUBSAMPLE=0.05 bash run_gang.sh
set -euo pipefail
cd "$(dirname "$0")"

PORT="${BRIDGE_PORT:-25360}"
PY="${PYBIN:-python3}"
CP="target/classes:target/dependency/*"
JAVA_XMX="${JAVA_XMX:-2048m}"

# ---- tunables (defaults chosen from the diag_reward calibration) ----
export NUM_HOSTS="${NUM_HOSTS:-20}"
export NUM_EPISODES="${NUM_EPISODES:-100}"
export MAX_STEPS="${MAX_STEPS:-120}"
export GANG_MAX_STEPS="${GANG_MAX_STEPS:-$MAX_STEPS}"
export SUBSAMPLE="${SUBSAMPLE:-0.05}"
export LR="${LR:-3e-4}"
export ENT_COEF="${ENT_COEF:-0.01}"
export GANG_CSV="${GANG_CSV:-gang_train_summary.csv}"
export USE_STGNN="${USE_STGNN:-0}"
export RACK_HISTORY_LEN="${RACK_HISTORY_LEN:-6}"
export OBS_ENABLED="${OBS_ENABLED:-1}"
export PROMETHEUS_PORT="${PROMETHEUS_PORT:-8000}"
export RESUME="${RESUME:-0}"
export TRAIN_STATE_PATH="${TRAIN_STATE_PATH:-${GANG_CHECKPOINT_DIR:-.}/training_state.pt}"
export TB_LOG_DIR="${TB_LOG_DIR:-${GANG_CHECKPOINT_DIR:-.}/tensorboard}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"

echo "[run_gang] port=$PORT hosts=$NUM_HOSTS episodes=$NUM_EPISODES steps=$MAX_STEPS "
echo "[run_gang] architecture=$([ "$USE_STGNN" = 1 ] && echo STGNN || echo MLP) history=$RACK_HISTORY_LEN sub=$SUBSAMPLE"
echo "[run_gang] TensorBoard log=$TB_LOG_DIR port=$TENSORBOARD_PORT; Prometheus metrics port=$PROMETHEUS_PORT"
echo "[run_gang] resume=$RESUME state=$TRAIN_STATE_PATH"

if [ "${START_TENSORBOARD:-1}" = 1 ]; then
  pkill -f "tensorboard.*--port $TENSORBOARD_PORT" 2>/dev/null || true
  mkdir -p "$TB_LOG_DIR"
  "$PY" -m tensorboard.main --logdir "$TB_LOG_DIR" --host 0.0.0.0 \
    --port "$TENSORBOARD_PORT" > "/tmp/tensorboard_$TENSORBOARD_PORT.log" 2>&1 &
  echo "[run_gang] TensorBoard started (pid $!)"
fi

# ---- 1. start the bridge, wait for the listener ----
pkill -f "Py4jBridge $PORT" 2>/dev/null || true
sleep 1
NUM_HOSTS="$NUM_HOSTS" java -Xmx"$JAVA_XMX" -cp "$CP" com.dacn.advanced.Py4jBridge "$PORT" \
  > "/tmp/gangbridge_$PORT.log" 2>&1 &
BP=$!
port_open() {  # works on both Linux (ss) and macOS (lsof)
  ss -ltn 2>/dev/null | grep -q ":$1" && return 0
  lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}
for i in $(seq 1 60); do
  port_open "$PORT" && break
  sleep 1
done
if ! port_open "$PORT"; then
  echo "[run_gang] BRIDGE FAILED TO START"; tail -20 "/tmp/gangbridge_$PORT.log"; exit 1
fi
echo "[run_gang] bridge up (pid $BP)"

# ---- 2. train (bridge log is muted; progress is on stdout + the CSV) ----
trap 'kill $BP 2>/dev/null || true' EXIT
PYTHONPATH=python BRIDGE_PORT="$PORT" "$PY" -u python/marl_gang_train.py

echo "[run_gang] done. summary -> $GANG_CSV ; checkpoints -> {a1,a2,a3,a4,critic}_gang.pt"
