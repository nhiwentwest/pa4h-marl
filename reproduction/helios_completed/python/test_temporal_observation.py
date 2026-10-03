from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import marl_gang_train as trainer
from relief_credit import ReliefCredit

class Env:
    num_racks=2
    num_hosts=8
    max_steps=180
    sla_lambda_comp=1.
    _sla_priority_total=100.
    def __init__(self,critical=False):
        self.pending=[];self.ep_stats={};self.preempted=[]
        jobs=[SimpleNamespace(job_id='critical' if critical else 'job',num_nodes=1,
                              priority=1,placed_hosts=[0],remaining=5)]
        if critical:jobs.append(SimpleNamespace(job_id='safe',num_nodes=1,priority=1,placed_hosts=[1],remaining=1))
        self.running={j.job_id:j for j in jobs}
    def rack_features(self):return np.zeros((2,7),dtype=np.float32)
    def rack_of(self,host):return host//4
    def jobs_on_rack(self,rack):return list(self.running.values()) if rack==0 else []
    def project_rack_violation_fraction(self,rack,remove_job=None):
        return 1/1500 if rack==0 and self.running and remove_job is None and not self.preempted else 0.
    def preemption_sla_cost(self,job):return 0.
    def project_job_energy_saved_normalized(self,job):return 0.
    def job_features(self,job):return np.zeros(8,dtype=np.float32)
    def running_remaining(self,job):return job.remaining
    def preempt_job(self,job):
        self.preempted.append(job.job_id);self.running.pop(job.job_id)




def observation(temporal,collect):
    env=Env();rows=[]
    def act(role,obs,mask,step,**kwargs):
        if role=='a1':rows.append((obs.copy(),mask.copy()));return 0
        return int(np.flatnonzero(mask)[0])
    with patch.object(trainer,'A1_DELAY_CREDIT',True),patch.object(trainer,'A1_TEMPORAL_OBSERVATION',temporal), \
         patch.object(trainer,'projected_next_credit',return_value=ReliefCredit(.2,0.,.2,0.)), \
         patch.object(trainer,'delay_adjusted_utility',return_value=(.1,.1)) as teacher:
        trainer.gang_step(env,None,0,act,collect_credit=collect)
    return env,rows[0],teacher.call_count


def test_forecast_replaces_one_redundant_scalar_without_losing_current_risk():
    _,(old,oldmask),_=observation(False,False)
    _,(new,newmask),_=observation(True,False)
    assert old.shape==new.shape
    assert np.flatnonzero(old!=new).tolist()==[10]
    assert old[10]==old[8]-old[9]
    assert np.isclose(new[10],.4)
    assert np.array_equal(oldmask,newmask) and newmask.tolist()==[True,True]


def test_training_and_deployment_features_are_identical():
    _,(training,_),_=observation(True,True)
    _,(deployment,_),_=observation(True,False)
    assert np.array_equal(training,deployment)


def test_deployment_keeps_wait_legal_and_does_not_call_the_teacher():
    env,(_,mask),calls=observation(True,False)
    assert mask.tolist()==[True,True] and calls==0 and env.preempted==[]
