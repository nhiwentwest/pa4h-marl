#!/bin/bash
cd "$(dirname "$0")/.."
pkill -f Py4jBridge
java -cp "target/classes:target/dependency/*" com.dacn.advanced.Py4jBridge > java_bridge.log 2>&1 &
sleep 5
python3 python/eval_baselines.py
