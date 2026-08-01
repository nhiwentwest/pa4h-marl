#!/bin/bash
cd "$(dirname "$0")/.."
pkill -f Py4jBridge 2>/dev/null || true
sleep 1
nohup java -cp 'target/classes:target/dependency/*' com.dacn.advanced.Py4jBridge > test_bridge_seed1.log 2>&1 < /dev/null &
echo "Waiting 3s for Java..."
sleep 3
export SEED=1
export NUM_EPISODES=300
nohup python3 python/marl_gang_train.py > train_seed1.log 2>&1 < /dev/null &
echo "Training started with SEED=1."
