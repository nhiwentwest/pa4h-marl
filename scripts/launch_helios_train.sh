#!/bin/bash
cd "$(dirname "$0")/.."

# Make sure we don't conflict with existing py4j bridge
pkill -f Py4jBridge 2>/dev/null || true
sleep 1

# Start the java bridge on port 25444
export BRIDGE_PORT=25444
nohup java -cp 'target/classes:target/dependency/*' com.dacn.advanced.Py4jBridge $BRIDGE_PORT > helios_bridge.log 2>&1 < /dev/null &
echo "Waiting 3s for Java Py4jBridge..."
sleep 3

export SEED=1
export NUM_EPISODES=300
export SUBSAMPLE=0.01
export MAX_STEPS=1200
export HELIOS_WINDOW_HOURS=200
export POD_HOURLY_JOBS="data/helios_earth_pod_hourly_jobs.csv"

# Make sure the output directory is distinct
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export GANG_CHECKPOINT_DIR="outputs/gang_helios_300eps_${TIMESTAMP}"
export GANG_CSV="${GANG_CHECKPOINT_DIR}/train.csv"

mkdir -p $GANG_CHECKPOINT_DIR

echo "Starting Helios training for $NUM_EPISODES eps with seed $SEED..."
echo "Checkpoints and CSV will be saved to $GANG_CHECKPOINT_DIR"

nohup python3 python/marl_gang_train_helios.py > "${GANG_CHECKPOINT_DIR}/train.log" 2>&1 < /dev/null &
echo "Training started in background. Monitor with: tail -f ${GANG_CHECKPOINT_DIR}/train.log"
