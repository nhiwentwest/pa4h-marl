from types import SimpleNamespace
import numpy as np
import pytest
from a1_delay_credit import delay_adjusted_utility, projected_next_credit
from relief_credit import ReliefCredit


def test_recorded_sparse_risk_gets_credit_for_avoiding_later_preemption():
    current=ReliefCredit(.0017916666667,.0033013606444,.0231770833333,.0020611253567)
    delta,premium=delay_adjusted_utility(current,8.,.1562882959843,.99)
    assert current.delta_utility(8)<0
    assert delta>0 and premium<=current.immediate_cost+current.sla_proxy_cost


def test_wait_remains_best_if_future_risk_is_not_profitable():
    current=ReliefCredit(0.,0.,.1,0.)
    for future in (0.,-.1):
        delta,premium=delay_adjusted_utility(current,8.,future,.99)
        assert delta==-.1 and premium==0
    delta,_=delay_adjusted_utility(current,8.,.05,.99)
    assert delta<0


def test_existing_positive_credit_does_not_change():
    current=ReliefCredit(.1,0.,.1,0.)
    assert delay_adjusted_utility(current,8.,1.,.99)==pytest.approx((.7,0.))


@pytest.mark.parametrize('future,discount',[(float('nan'),.99),(float('inf'),.99),(1.,1.1),(1.,-.1)])
def test_invalid_future_credit_rejected(future,discount):
    with pytest.raises(ValueError):delay_adjusted_utility(ReliefCredit(0.,0.,.1,0.),8.,future,discount)


class Env:
    num_hosts=4
    num_racks=1
    step_idx=0
    rack_budget=10.
    def __init__(self,remaining=3):
        self.victim=SimpleNamespace(job_id='v',placed_hosts=[0],arrival_step_placed=0,remaining=remaining)
        self.neighbor=SimpleNamespace(job_id='n',placed_hosts=[1],arrival_step_placed=0,remaining=1)
    def running_remaining(self,job):return job.remaining
    def rack_of(self,host):return 0
    def jobs_on_rack(self,rack):return [self.victim,self.neighbor]
    def _fine_window(self,job,elapsed):
        return np.array([8.,8.]) if job is self.victim else np.array([100.,100.])


def test_completed_neighbor_is_excluded_from_future_projection():
    env=Env();current=ReliefCredit(.1,0.,.1,.02)
    future=projected_next_credit(env,env.victim,current,idle_w=1.,peak_w=10.,energy_weight=.3,network_tax=0.,rack_size=4)
    # Victim + idle hosts = 11 W. The completed neighbor must not be projected.
    assert future.global_violation_relief==1.
    assert future.energy_benefit==pytest.approx(.3*7/40)
    assert (future.immediate_cost,future.sla_proxy_cost)==(.1,.02)


def test_finishing_victim_has_no_avoidable_future_preemption():
    env=Env(remaining=1)
    assert projected_next_credit(env,env.victim,ReliefCredit(0.,0.,.1,0.),
        idle_w=1.,peak_w=10.,energy_weight=.3,network_tax=0.,rack_size=4) is None


def test_last_episode_step_cannot_claim_a_future_benefit():
    env=Env();env.step_idx=179;env.max_steps=180
    assert projected_next_credit(env,env.victim,ReliefCredit(0.,0.,.1,0.),
        idle_w=1.,peak_w=10.,energy_weight=.3,network_tax=0.,rack_size=4) is None
