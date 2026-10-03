from types import SimpleNamespace as NS
import numpy as np
import pytest
import queue_relief as q


def fixture():
    victim=NS(job_id='gang',num_nodes=8,arrival_step=0,duration_steps=120,remaining_duration_steps=80,priority=1.,placed_hosts=[0,1,2,3,4,5,6,7])
    pending=NS(job_id='short',num_nodes=1,arrival_step=30,duration_steps=1,remaining_duration_steps=1)
    e=NS(step_idx=40,max_steps=180,num_hosts=64,num_racks=16,pending=[NS(**{**vars(pending),'job_id':str(i)}) for i in range(64)],running={'gang':victim},
         ep_stats={},_first_wait={},sla_lambda_comp=1.,_sla_priority_total=100.)
    q.reset(e)
    e.running_remaining=lambda j:j.remaining_duration_steps
    e.rack_of=lambda h:h//4
    e.rack_features=lambda:np.zeros((16,7),dtype=np.float32)
    e.jobs_on_rack=lambda r:[victim] if r in (0,1) else []
    e.job_features=lambda j:np.zeros(8,dtype=np.float32)
    e.project_job_energy_saved_normalized=lambda j:0.01
    e.preempt_job=lambda j:e.running.pop(j.job_id)
    t=NS(RACK_SIZE=4,A3_SLOT_DIM=14,JOB_OBS_DIM=8,W_SERVE=1.,W_CKPT=.15,W_WAIT=1.,W_ENERGY=.3)
    return e,t,victim


def test_uses_all_actors_and_keeps_a1_wait_legal():
    e,t,j=fixture();calls=[]
    def act(name,obs,mask,step,**kw):
        calls.append(name)
        assert np.isfinite(obs).all() and kw['credit_context']['team_only']
        assert np.all(kw['credit_context']['utility']==0)
        if name=='a1':
            assert mask.tolist()==[True,True]
            return 1
        return 0
    assert q.step(t,e,None,40,act)
    assert calls==['a2','a3','a1'] and e.queue_holds=={'gang':98}
    assert 'gang' not in e.running
    e.step_idx=97;assert q.held(e,j)
    e.step_idx=98;assert not q.held(e,j)
    q.reset(e);assert not e.queue_holds and not e.queue_loaned


def test_a1_wait_does_not_mutate_schedule():
    e,t,j=fixture()
    assert not q.step(t,e,None,40,lambda *a,**k:0)
    assert e.running['gang'] is j and not e.queue_holds and not e.queue_loaned


def test_no_event_without_urgent_pending():
    e,t,j=fixture();
    for job in e.pending:job.arrival_step=39
    assert not q.step(t,e,None,40,lambda *a,**k:pytest.fail('unexpected decision'))


@pytest.mark.parametrize('attribute,value',[('remaining_duration_steps',135),('remaining_duration_steps',11),('num_nodes',1)])
def test_no_unsafe_or_tiny_victim(attribute,value):
    e,t,j=fixture();setattr(j,attribute,value);assert q.eligible(e)==[]


def test_no_repeat_loan():
    e,t,j=fixture();e.queue_loaned.add(j.job_id);assert q.eligible(e)==[]


def test_invalid_actor_action_is_rejected():
    e,t,j=fixture()
    with pytest.raises(ValueError):q.step(t,e,None,40,lambda *a,**k:15)


def test_migration_rejects_mismatch_and_preserves_parent():
    s=dict(episode=160,schema_version=2,semantic_version='helios-a4-completed-diverse-v3',simulator_semantics='same',relief_credit={'data':'x'})
    for k in ('actors','critic','optimizers','rng','sla','a4_return_critic'):s[k]={}
    desc=dict(data='x',queue_relief=q.descriptor())
    m=q.migrate_state(s,desc,'same','a'*64)
    assert m['semantic_version']=='helios-queue-relief-v4'
    assert s['semantic_version']=='helios-a4-completed-diverse-v3'
    assert m['queue_migration']['fresh_initialization'] is False
    with pytest.raises(ValueError):q.migrate_state(m,desc,'same','a'*64)
    with pytest.raises(ValueError):q.migrate_state(s,dict(data='y',queue_relief=q.descriptor()),'same','a'*64)


def test_no_loan_for_small_backlog():
    e,t,j=fixture();e.pending=e.pending[:5];assert q.eligible(e)==[]


def test_small_backlog_keeps_order():
    e,t,j=fixture();e.pending=e.pending[:2];e.pending[0].arrival_step=39;e.pending[1].arrival_step=0
    before=list(e.pending);q.order_pending(e);assert e.pending==before


def test_ageing_and_feasible_deadline_order():
    e,t,j=fixture()
    aged=NS(job_id='aged',arrival_step=0,duration_steps=1,remaining_duration_steps=1,num_nodes=1)
    late=NS(job_id='late',arrival_step=20,duration_steps=1,remaining_duration_steps=1,num_nodes=1)
    e.queue_holds={'borrower':98}
    e.pending=[late]+e.pending+[aged];q.order_pending(e)
    assert e.pending[0] is aged and e.pending[-1] is late


def test_migration_disables_stale_return_critic_gate():
    s=dict(episode=160,schema_version=2,semantic_version='helios-a4-completed-diverse-v3',simulator_semantics='same',relief_credit={})
    for k in ('actors','critic','optimizers','rng','sla','a4_return_critic'):s[k]={}
    s['a4_return_critic']={'ready':True,'quality_streak':5}
    m=q.migrate_state(s,{'queue_relief':q.descriptor()},'same','b'*64)
    assert m['a4_return_critic']['ready'] is False and m['a4_return_critic']['quality_streak']==0
    assert s['a4_return_critic']['ready'] is True


def test_no_queue_reordering_before_any_loan():
    e,t,j=fixture();e.pending[0].arrival_step=35;e.pending[-1].arrival_step=0
    before=list(e.pending);q.order_pending(e);assert e.pending==before
