import pytest
from checkpoint_quality import validation_quality, baseline_comparison


def test_best_quality_is_independent_of_rpa_even_when_reward_is_below_it():
    windows=[dict(hour=0,jobs=100,feasible_completion_upper_bound=98)]
    candidate=[dict(hour=0,reward=-10,completed=98,sla=2,violations=0,wait=10,preempt=1)]
    limits=dict(min_feasible_completion_fraction=.95,max_sla_fraction=.2,max_power_violations=0)
    quality=validation_quality(candidate,windows,180,limits)
    assert quality['qualified'] and quality['score']==-10/180
    weak=[dict(candidate[0],reward=-20)]
    strong=[dict(candidate[0],reward=20)]
    assert baseline_comparison(candidate,weak)[0]['reward_delta']==10
    assert baseline_comparison(candidate,strong)[0]['reward_delta']==-30
    assert validation_quality(candidate,windows,180,limits)==quality


def test_missing_metric_and_power_regression_are_not_qualified():
    windows=[dict(hour=0,jobs=100,feasible_completion_upper_bound=100)]
    limits=dict(min_feasible_completion_fraction=.95,max_sla_fraction=.2,max_power_violations=0)
    row=dict(hour=0,reward=100,completed=100,sla=0,violations=1,wait=0,preempt=0)
    assert not validation_quality([row],windows,180,limits)['qualified']
    with pytest.raises(ValueError,match='windows'):
        validation_quality([],windows,180,limits)
