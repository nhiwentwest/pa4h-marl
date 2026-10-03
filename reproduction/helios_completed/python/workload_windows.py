"""Outcome-blind workload statistics and disjoint window selection."""
import hashlib
import numpy as np


def window_metrics(frame, hour, subsample, horizon=180, width=10):
    x = frame.loc[(frame.arrival_hour >= hour) & (frame.arrival_hour < hour+width)]
    nodes = np.ceil(x.total_gpu_request.to_numpy(float)/4).astype(int)
    hashes = np.array([int(hashlib.md5(str(v).encode()).hexdigest(), 16)
                       for v in x.workload_id], dtype=object)
    keep = (nodes >= 1) & (nodes <= 8) & (hashes % 10000 < int(subsample*10000))
    x = x.loc[keep]
    nodes = nodes[keep]
    if not len(x):
        return None
    arrival = ((x.arrival_hour.to_numpy(int)-hour)*12 +
               np.asarray(hashes[keep] % 12, dtype=int))
    duration = np.maximum(1, np.rint(x.duration_hours.to_numpy(float)*12)).astype(int)
    demand_delta = np.zeros(horizon+1)
    np.add.at(demand_delta, arrival, nodes)
    np.add.at(demand_delta, np.minimum(arrival+duration, horizon), -nodes)
    demand = np.cumsum(demand_delta)[:horizon]
    impossible = int(np.sum(arrival+duration > horizon))
    return dict(hour=int(hour), subsample=float(subsample), jobs=len(x),
                one_step_fraction=float(np.mean(duration == 1)),
                long_fraction=float(np.mean(duration >= 12)),
                gang_fraction=float(np.mean(nodes >= 2)),
                offered_capacity_ratio=float(demand.mean()/64),
                offered_nodes_peak=float(demand.max()),
                overload_fraction=float(np.mean(demand > 64)),
                earliest_finish_impossible=impossible,
                feasible_completion_upper_bound=len(x)-impossible)


def select_windows(candidates, count_per_stratum=1):
    """Pick workload coverage, with no policy returns or outcomes as inputs."""
    selected = []
    available = [x for x in candidates if 80 <= x['jobs'] <= 5000
                 and .15 <= x['offered_capacity_ratio'] <= .85
                 and x['earliest_finish_impossible']/x['jobs'] <= .10]
    for repetition in range(count_per_stratum):
        for stratum in ('long', 'gang', 'queue'):
            eligible = [x for x in available if all(abs(x['hour']-y['hour']) >= 10 for y in selected)]
            if stratum == 'long':
                eligible = [x for x in eligible if x['long_fraction'] >= .10]
                key = lambda x: (abs(x['offered_capacity_ratio']-.35),
                                 abs(x['long_fraction']-.20), x['hour'], x['subsample'])
            elif stratum == 'gang':
                eligible = [x for x in eligible if x['gang_fraction'] >= .10]
                key = lambda x: (abs(x['offered_capacity_ratio']-.45),
                                 abs(x['gang_fraction']-.20), x['hour'], x['subsample'])
            else:
                eligible = [x for x in eligible if x['overload_fraction'] >= .05]
                key = lambda x: (abs(x['offered_capacity_ratio']-.60),
                                 abs(x['overload_fraction']-.15), x['hour'], x['subsample'])
            if not eligible:
                raise ValueError(f'no independent {stratum} window in declared pool')
            chosen = dict(min(eligible, key=key), stratum=stratum)
            selected.append(chosen)
    return selected


def validate_splits(train, development, heldout_hours, width=10):
    if not train or not development:
        raise ValueError('empty split')
    intervals = [(x['hour'], x['hour']+width) for x in train+development]
    intervals += [(h, h+width) for h in heldout_hours]
    for index, (start, end) in enumerate(intervals):
        if any(max(start, a) < min(end, b) for a, b in intervals[index+1:]):
            raise ValueError('workload windows overlap')
