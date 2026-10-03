import numpy as np
import pytest
import torch
from a4_branch_rank import branch_rank_loss, migrate_branch_state, TapeRecorder


def test_actual_branch_fixture_gradient():
    import json
    from pathlib import Path
    fixture = Path(__file__).resolve().parent/'fixtures/branch_train930.json'
    data=json.loads(fixture.read_text())
    row=next(r for r in data['results'] if r['job_id']=='680403')
    logits=torch.zeros((1,17),requires_grad=True)
    mask=torch.ones_like(logits,dtype=torch.bool)
    labels=torch.full_like(logits,float('nan'))
    for b in row['branches']:labels[0,b['action']]=b['reward8']
    loss,n=branch_rank_loss(logits,mask,labels)
    loss.backward()
    assert n>0 and logits.grad[0,13]<0 and logits.grad[0,0]>0
    assert logits.grad[0,16]==0


def test_unqueried_remain_legal_and_ties_zero():
    x=torch.tensor([[0.,1.,2.,float('-inf')]],requires_grad=True)
    mask=torch.tensor([[1,1,1,0]],dtype=torch.bool)
    y=torch.tensor([[3.,3.,float('nan'),float('nan')]])
    loss,n=branch_rank_loss(x,mask,y);loss.backward()
    assert n==0 and loss==0 and torch.equal(x.grad,torch.zeros_like(x))
    y[0,1]=4.;x.grad=None
    loss,n=branch_rank_loss(x,mask,y);loss.backward()
    assert n==1 and x.grad[0,2]==0 and x.grad[0,3]==0
    assert mask[0,2]


def test_invalid_label_rejected():
    with pytest.raises(ValueError):
        branch_rank_loss(torch.zeros(1,3),torch.tensor([[1,0,1]],dtype=torch.bool),torch.tensor([[1.,2.,float('nan')]]))


def test_gap_not_normalized():
    x=torch.zeros(1,3,requires_grad=True);m=torch.ones_like(x,dtype=torch.bool)
    a,_=branch_rank_loss(x,m,torch.tensor([[0.,.001,float('nan')]]))
    b,_=branch_rank_loss(x,m,torch.tensor([[0.,.1,float('nan')]]))
    assert float(b/a)==pytest.approx(100.,rel=1e-4)


def test_migration_is_explicit_preserves_state():
    state=dict(schema_version=2,episode=24,semantic_version='helios-a1-temporal-credit-v1',simulator_semantics='sim',relief_credit={'a':1})
    for k in ('actors','optimizers','rng','sla','critic','a4_return_critic'):state[k]={}
    desc={'a':1,'a4_branch_rank':{'schema':'v1'}}
    migrated=migrate_branch_state(state,desc,'sim',24)
    assert migrated['actors'] is state['actors']
    assert state['semantic_version']=='helios-a1-temporal-credit-v1'
    assert migrated['semantic_version']=='helios-a4-branch-rank-v1'
    with pytest.raises(ValueError):migrate_branch_state(migrated,desc,'sim',24)


def test_recorder_does_not_change_numpy_rng():
    np.random.seed(5);before=np.random.get_state()
    tape=TapeRecorder(3)
    tape.record('a1',np.zeros(3),np.ones(2,dtype=bool),1,0,None)
    after=np.random.get_state()
    assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    assert len(tape.tape)==1 and tape.case is None


def test_actual_prefix_control_and_branch_order(monkeypatch):
    import a4_branch_rank as module
    from types import SimpleNamespace
    class Env:
        max_steps=4
        def reset(self):self.step=0;self.chosen=0;self.ep_stats={}
        def advance(self):
            r=1.+self.chosen;self.step+=1;self.done=self.step==4
            return {'reward':r}
    e=Env()
    def gang(env,ag,step,act,**kw):
        env.chosen=act('a4',np.array([step],dtype=float),np.ones(3,dtype=bool),step)
    t=SimpleNamespace(gang_step=gang,GAMMA=.9,REWARD_SCALE=1.)
    def deployment(*args):
        def fn(*a,**kw):return 0
        fn.current={};return fn
    monkeypatch.setattr(module,'deployment_act',deployment)
    tape=[(module.signature('a4',np.array([s]),np.ones(3,dtype=bool),s),0) for s in range(4)]
    case={'index':1,'step':1}
    zero,_=module.replay_branch(t,None,e,tape,case,0,replay_suffix=True)
    assert zero==pytest.approx(1+.9+.81)
    a,_=module.replay_branch(t,None,e,tape,case,1)
    b,_=module.replay_branch(t,None,e,tape,case,2)
    b2,_=module.replay_branch(t,None,e,tape,case,2)
    a2,_=module.replay_branch(t,None,e,tape,case,1)
    assert (a,b)==(a2,b2) and b>a
    with pytest.raises(RuntimeError):
        module.replay_branch(t,None,e,[(('bad',0,''),0)],case,0)


def test_reservoir_only_distinct_host_plans():
    from types import SimpleNamespace
    env=SimpleNamespace(_branch_job=SimpleNamespace(job_id=5),_pick_hosts_pa=lambda a,j:[a//2])
    recorder=TapeRecorder(7)
    recorder.record('a4',np.zeros(2),np.ones(5,dtype=bool),20,0,env)
    assert len(recorder.case['candidates'])==2
    assert recorder.case['candidates'][0]==0
