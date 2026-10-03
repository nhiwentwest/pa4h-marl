"""Lock workload-only selection before evaluating a learned policy."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from workload_windows import window_metrics, select_windows, validate_splits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.jobs)
    frame = pd.read_csv(path, dtype={'workload_id': str})
    if set(frame.source_state) != {'COMPLETED'}:
        raise ValueError('primary training data must contain COMPLETED only')
    # Predeclared chronological pools. Prior dev and heldout 1020/1030 are
    # excluded; selection never consumes reward or policy evaluation results.
    pools = dict(train=(0, 900), development=(1040, 1800))
    candidates = {}
    for split, (start, end) in pools.items():
        candidates[split] = [metrics for hour in range(start, end, 10)
                             for sub in (.05, .10, .25, .50, .75, 1.0)
                             if (metrics := window_metrics(frame, hour, sub)) is not None]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_name('window_candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
    train = select_windows(candidates['train'], count_per_stratum=2)
    dev = select_windows(candidates['development'])
    validate_splits(train, dev, [1020, 1030])
    manifest = dict(schema='helios-workload-stratified-windows-v1',
                    selection_uses_policy_results=False, pools=pools,
                    data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    train_windows=train, development_windows=dev,
                    heldout_hours=[1020, 1030], window_hours=10, horizon=180,
                    selection_rules=dict(min_jobs=80, max_jobs=5000, offered_load=[.15,.85],
                        max_impossible_fraction=.10, long_min=.10, gang_min=.10,
                        queue_min_overload=.05, load_targets=dict(long=.35, gang=.45, queue=.60)),
                    candidates=candidates)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'candidates'}, indent=2))
