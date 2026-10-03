import pandas as pd
import pytest
from helios_workload import convert_frame


def raw():
    return pd.DataFrame(dict(job_id=['failed-first', 'ok', 'cancel', 'missing', 'zero'],
        gpu_num=[4, 8, 1, 4, 4], node_num=[1, 2, 1, 1, 1],
        state=['FAILED', 'COMPLETED', 'CANCELLED', 'COMPLETED', 'COMPLETED'],
        submit_time=['2020-01-01 00:00:00', '2020-01-01 03:00:00',
                     '2020-01-01 04:00:00', 'invalid', '2020-01-01 05:00:00'],
        duration=[100, 3600, 200, 100, 0]))


def test_outcome_filter_preserves_origin_and_observed_runtime():
    completed, meta = convert_frame(raw(), 'completed')
    all_state, ablation = convert_frame(raw(), 'all_observed')
    assert completed.workload_id.tolist() == ['ok']
    assert completed.arrival_hour.tolist() == [3]
    assert completed.duration_hours.tolist() == [1.0]
    assert completed.source_state.tolist() == ['COMPLETED']
    assert all_state.workload_id.tolist() == ['failed-first', 'ok', 'cancel']
    assert meta['time_origin'] == ablation['time_origin']
    pd.testing.assert_frame_equal(completed, all_state.loc[all_state.source_state == 'COMPLETED'].reset_index(drop=True))


def test_ambiguous_ids_and_unknown_policy_fail_closed():
    duplicate = pd.concat([raw(), raw().iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match='duplicate'):
        convert_frame(duplicate)
    with pytest.raises(ValueError, match='policy'):
        convert_frame(raw(), 'success_assumed')
