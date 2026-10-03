import pandas as pd
import pytest
from workload_windows import window_metrics, select_windows, validate_splits


def test_workload_demand_is_clipped_at_horizon_and_respects_fit():
    frame = pd.DataFrame(dict(workload_id=['a','b','oversized','outside'],
        arrival_hour=[0,0,0,10], total_gpu_request=[4,8,100,4],
        duration_hours=[1/12,20,1,1]))
    result = window_metrics(frame, 0, 1.0)
    assert result['jobs'] == 2
    assert result['earliest_finish_impossible'] == 1
    assert result['feasible_completion_upper_bound'] == 1
    assert result['gang_fraction'] == .5
    assert 0 < result['offered_capacity_ratio'] < 3/64


def test_splits_reject_overlaps_including_reserved_holdout():
    with pytest.raises(ValueError, match='overlap'):
        validate_splits([dict(hour=1015)], [dict(hour=1200)], [1020,1030])
    validate_splits([dict(hour=800)], [dict(hour=1200)], [1020,1030])


def test_selection_uses_workload_coverage_and_independent_windows():
    base=dict(subsample=.5,jobs=100,earliest_finish_impossible=0,
              offered_capacity_ratio=.35,long_fraction=.2,gang_fraction=.2,
              overload_fraction=.1,one_step_fraction=.5)
    rows=[dict(base,hour=h) for h in (0,10,20)]
    selected=select_windows(rows)
    assert {x['stratum'] for x in selected} == {'long','gang','queue'}
    assert len({x['hour'] for x in selected}) == 3
    with pytest.raises(ValueError, match='independent'):
        select_windows(rows[:1])
