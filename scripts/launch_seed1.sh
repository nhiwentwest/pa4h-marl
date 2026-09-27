#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SEED="${SEED:-1}"
export NUM_EPISODES="${NUM_EPISODES:-300}"
bash scripts/run_gang.sh
