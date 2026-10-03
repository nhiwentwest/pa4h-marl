"""Explicit Helios outcome policy with a stable time origin for comparisons."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = 'helios-observed-runtime-outcomes-v1'


def convert_frame(raw, outcome_policy='completed'):
    if outcome_policy not in ('completed', 'all_observed'):
        raise ValueError('unknown outcome policy')
    required = {'job_id', 'gpu_num', 'node_num', 'state', 'submit_time', 'duration'}
    if not required.issubset(raw.columns):
        raise ValueError(f'missing columns: {sorted(required-set(raw.columns))}')
    frame = raw.copy()
    for name in ('gpu_num', 'node_num', 'duration'):
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
    submit = pd.to_datetime(frame.submit_time, errors='coerce')
    eligible = (np.isfinite(frame.gpu_num) & (frame.gpu_num > 0)
                & np.isfinite(frame.duration) & (frame.duration > 0)
                & np.isfinite(frame.node_num) & (frame.node_num >= 1) & submit.notna())
    if not eligible.any():
        raise ValueError('no usable execution intervals')
    # Determine origin BEFORE the outcome filter. All-state and completed-only
    # views must keep the same arrival hours, IDs and hash subsampling.
    origin = submit.loc[eligible].min()
    states = frame.state.astype(str).str.upper().str.strip()
    keep = eligible & ((states == 'COMPLETED') if outcome_policy == 'completed' else True)
    if not keep.any():
        raise ValueError('outcome policy leaves no jobs')
    ids = frame.loc[keep, 'job_id'].astype(str)
    if ids.duplicated().any():
        raise ValueError('duplicate job_id')
    result = pd.DataFrame({
        'workload_id': ids,
        'arrival_hour': ((submit.loc[keep]-origin).dt.total_seconds()//3600).astype(int),
        'total_gpu_request': frame.loc[keep, 'gpu_num'].astype(float),
        'num_pods': frame.loc[keep, 'node_num'].astype(int),
        'job_type': 'training', 'model_type': 'unknown',
        'duration_hours': frame.loc[keep, 'duration'].astype(float)/3600,
        'source_state': states.loc[keep],
    }).sort_values(['arrival_hour', 'workload_id']).reset_index(drop=True)
    metadata = dict(schema=SCHEMA, outcome_policy=outcome_policy,
                    time_origin=origin.isoformat(), raw_rows=len(frame),
                    eligible_rows=int(eligible.sum()), output_rows=len(result),
                    eligible_states=states.loc[eligible].value_counts().to_dict(),
                    runtime_semantics='observed end-start; no remaining-work imputation',
                    all_observed_is_ablation=outcome_policy == 'all_observed')
    return result, metadata


def prepare(input_path, output_path, outcome_policy):
    source = Path(input_path)
    frame, metadata = convert_frame(pd.read_csv(source, dtype={'job_id': str}), outcome_policy)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    metadata.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    destination.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata
