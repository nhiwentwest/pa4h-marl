#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${BRIDGE_PORT:-25361}"
PY="${PYBIN:-python3}"
export BRIDGE_PORT="$PORT"
export NUM_HOSTS="${NUM_HOSTS:-64}"
export STEPS="${STEPS:-120}"
# Java and Python must end at the same horizon (Helios uses 180 steps).
export GANG_MAX_STEPS="$STEPS"
port_open() {
  ss -ltn 2>/dev/null | grep -q ":$1" && return 0
  lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}
if port_open "$PORT"; then
  echo "[run_eval] port $PORT is already in use" >&2
  exit 1
fi
java -Xmx"${JAVA_XMX:-2048m}" -cp 'target/classes:target/dependency/*' com.dacn.advanced.GangBridge "$PORT" \
  > "${GANG_BRIDGE_LOG:-${GANG_CHECKPOINT_DIR:-.}/gang_eval_bridge.log}" 2>&1 &
BRIDGE_PID=$!
trap 'kill "$BRIDGE_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "[run_eval] bridge exited before readiness" >&2
    exit 1
  fi
  if port_open "$PORT"; then break; fi
  sleep 1
done
port_open "$PORT" || { echo "[run_eval] bridge failed to listen" >&2; exit 1; }
PYTHONPATH="python:vendor${PYTHONPATH:+:$PYTHONPATH}" "$PY" scripts/check_bridge_contract.py
PYTHONPATH="python:vendor${PYTHONPATH:+:$PYTHONPATH}" "$PY" python/eval_gang.py "$PORT" "$@"
