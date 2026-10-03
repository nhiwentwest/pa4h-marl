from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
import marl_gang_train as trainer
from power_constraint import PowerConstraintError


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


def run(env,enabled,step=0,ignore_mask=False):
    calls=[]
    def act(role,obs,mask,step,**kwargs):
        calls.append((role,mask.copy(),kwargs))
        if ignore_mask and role=='a1':return 0
        return int(np.flatnonzero(mask)[0])
    with patch.object(trainer,'HARD_POWER_CONSTRAINT',enabled):
        trainer.gang_step(env,None,step,act)
    return calls


def test_training_and_eval_shared_path_forbids_unsafe_wait():
    env=Env();calls=run(env,True)
    assert env.preempted==['job']
    a1=[c for c in calls if c[0]=='a1'][0]
    assert a1[1].tolist()==[False,True]
    assert a1[2]['record'] is False  # forced events are not represented as learned choices
    assert env.ep_stats['power_constraint_forced_preempt']==1


def test_legacy_path_and_utility_are_preserved_when_disabled():
    env=Env();calls=run(env,False)
    assert env.preempted==[]
    a1=[c for c in calls if c[0]=='a1'][0]
    assert a1[1].tolist()==[True,True]
    assert a1[2]['credit_context']['utility'][1]<0


def test_completion_preference_is_shared_with_training():
    env=Env(critical=True);calls=run(env,True,step=175)
    assert env.preempted==['safe']
    a3=[c for c in calls if c[0]=='a3'][0]
    assert a3[1].tolist()==[False,True,False,False]
    assert a3[2]['record'] is False


def test_actor_cannot_ignore_constraint_mask():
    with pytest.raises(PowerConstraintError):run(Env(),True,ignore_mask=True)


def test_no_causal_victim_fails_before_advancing():
    env=Env();env.jobs_on_rack=lambda rack:[]
    with pytest.raises(PowerConstraintError):run(env,True)


def test_new_recipe_rejects_soft_budget_checkpoint(tmp_path):
    checkpoint=tmp_path/'old.pt'
    torch.save({'semantic_version':'helios-a4-reward-cf-v2'},checkpoint)
    with patch.object(trainer,'HARD_POWER_CONSTRAINT',True):
        with pytest.raises(ValueError,match='checkpoint'):
            trainer.load_training_state(str(checkpoint),None,None)
