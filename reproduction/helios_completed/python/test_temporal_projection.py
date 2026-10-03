from types import SimpleNamespace
import numpy as np
import pytest
from a1_delay_credit import projected_next_credit
from relief_credit import ReliefCredit


class Env:
    num_hosts=8;num_racks=2;step_idx=0;max_steps=180;rack_budget=10.
    def __init__(self):
        self.job=SimpleNamespace(job_id='v',placed_hosts=[0,4],arrival_step_placed=0)
    def running_remaining(self,j):return 3
    def rack_of(self,h):return h//4
    def jobs_on_rack(self,r):return [self.job]
    def _fine_window(self,j,elapsed):return np.array([8.,8.])


def test_local_projection_matches_selected_rack_and_global_sum():
    env=Env();current=ReliefCredit(0.,0.,.1,0.)
    kwargs=dict(idle_w=1.,peak_w=10.,energy_weight=.3,network_tax=0.,rack_size=4)
    global_credit=projected_next_credit(env,env.job,current,**kwargs)
    local_credit=projected_next_credit(env,env.job,current,rack_filter=0,**kwargs)
    assert global_credit.global_violation_relief==1.
    assert local_credit.global_violation_relief*env.num_racks==1.
    assert local_credit.energy_benefit*2==pytest.approx(global_credit.energy_benefit)
