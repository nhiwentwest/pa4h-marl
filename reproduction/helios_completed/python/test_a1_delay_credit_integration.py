from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
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



def replay(collect, future=None):
    env=Env();captured=[]
    def act(role,obs,mask,step,**kwargs):
        if role=='a1':
            captured.append((obs.copy(),mask.copy(),kwargs['credit_context']['utility'].copy()))
            return int(np.argmax(kwargs['credit_context']['utility']))
        return int(np.flatnonzero(mask)[0])
    with patch.object(trainer,'A1_DELAY_CREDIT',True), patch.object(trainer,'A1_VIOL_WEIGHT',8.), patch.object(trainer,'projected_next_credit',
            return_value=future) as projection:
        trainer.gang_step(env,None,0,act,collect_credit=collect)
    return env,captured[0],projection.call_count


def test_training_receives_future_credit_but_both_actions_remain_legal():
    env,(_,mask,utility),count=replay(True,ReliefCredit(.2,0.,.2,0.))
    assert mask.tolist()==[True,True]
    assert utility[1]>0 and count==1
    assert env.preempted==['job']
    assert env.ep_stats['a1_delay_credit_label_flips']==1


def test_deployment_does_not_compute_or_use_future_teacher():
    env,(_,mask,utility),count=replay(False,ReliefCredit(.2,0.,.2,0.))
    assert mask.tolist()==[True,True]
    assert count==0 and utility[1]<0 and env.preempted==[]


def test_actor_observation_remains_identical():
    _,(training,_,_),_=replay(True,ReliefCredit(.2,0.,.2,0.))
    _,(deployment,_,_),_=replay(False)
    assert np.array_equal(training,deployment)


def test_nonprofitable_future_preserves_wait_target():
    env,(_,mask,utility),_=replay(True,ReliefCredit(0.,0.,.2,0.))
    assert mask.tolist()==[True,True] and utility[1]<0 and env.preempted==[]


def test_new_recipe_rejects_unconverted_v2_checkpoint(tmp_path):
    checkpoint=tmp_path/'old.pt';torch.save({'semantic_version':'helios-a4-reward-cf-v2'},checkpoint)
    with patch.object(trainer,'A1_DELAY_CREDIT',True):
        with pytest.raises(ValueError,match='checkpoint'):
            trainer.load_training_state(str(checkpoint),None,None)
