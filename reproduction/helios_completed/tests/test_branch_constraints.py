import numpy as np
import pytest
import torch
from branch_constraints import qualify_actions, qualify_case, capture_outcome, SAFE, UNSAFE, UNKNOWN
from a4_policy_regret import policy_regret_loss


def metrics(completed=2200, sla=19, power=0):
    return dict(completed=completed, sla=sla, power=power)


def test_higher_reward_cannot_override_completion_regression():
    q, reasons = qualify_actions(np.ones(4, bool), {0:[1],1:[2],2:[2]}, 0,
                                metrics(), {1:metrics(completed=2199)})
    assert q.tolist() == [SAFE, UNSAFE, UNSAFE, UNKNOWN]
    assert reasons[1] == ['completed'] and reasons[2] == ['completed']
    logits = torch.tensor([[.4, .2, .1, -2.]], requires_grad=True)
    returns = torch.tensor([[-105.789, -102.427, -102.427, float('nan')]])
    loss, info = policy_regret_loss(logits, torch.ones_like(logits, dtype=torch.bool),
                                  returns, qualification=torch.tensor(q[None]))
    loss.backward()
    assert loss == 0 and info['active'] == 0 and info['correct'] == 1
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_policy_already_on_unsafe_high_reward_action_is_corrected():
    logits = torch.tensor([[.1, .4, -2.]], requires_grad=True)
    values = torch.tensor([[-105., -102., float('nan')]])
    loss, info = policy_regret_loss(logits, torch.ones_like(logits, dtype=torch.bool), values,
                                  qualification=torch.tensor([[SAFE, UNSAFE, UNKNOWN]]))
    loss.backward()
    assert loss > 0 and info['unsafe'] == 1 and info['active'] == 1
    assert logits.grad[0,0] < 0 and logits.grad[0,1] > 0 and logits.grad[0,2] == 0


def test_missing_sla_is_unknown_and_never_invented_from_completion():
    q, reasons = qualify_actions(np.ones(3, bool), {0:[1],1:[2]}, 0,
                                metrics(), {1:dict(completed=2200,power=0)})
    assert q[1] == UNKNOWN and reasons[1] == ['missing:sla']
    logits = torch.tensor([[.4,.1,-2.]], requires_grad=True)
    loss, info = policy_regret_loss(logits, torch.ones_like(logits,dtype=torch.bool),
        torch.tensor([[0.,3.,float('nan')]]), qualification=torch.tensor(q[None]))
    loss.backward()
    assert loss == 0 and info['active'] == 0


def test_observed_harm_is_rejected_even_when_another_metric_is_missing():
    q, reasons = qualify_actions(np.ones(3,bool), {0:[1],1:[2]}, 0,
        dict(completed=2200,power=0), {1:dict(completed=2199,power=0)})
    assert q[1] == UNSAFE and 'completed' in reasons[1]


@pytest.mark.parametrize('candidate,reason', [(metrics(sla=20),'sla'), (metrics(power=1),'power')])
def test_all_recorded_guardrails_are_checked(candidate,reason):
    q, reasons = qualify_actions(np.ones(3,bool), {0:[1],1:[2]}, 0, metrics(), {1:candidate})
    assert q[1] == UNSAFE and reasons[1] == [reason]


def test_feasible_return_improvement_still_trains():
    logits = torch.tensor([[.4,.1,-2.]], requires_grad=True)
    loss, info = policy_regret_loss(logits, torch.ones_like(logits,dtype=torch.bool),
        torch.tensor([[0.,3.,float('nan')]]), qualification=torch.tensor([[SAFE,SAFE,UNKNOWN]]))
    loss.backward()
    assert loss > 0 and info['active'] == 1 and logits.grad[0,1] < 0


def test_exact_aliases_share_metrics_but_different_ring_order_does_not():
    q, _ = qualify_actions(np.ones(4,bool), {0:[1,2],1:[1,2],2:[2,1]}, 0, metrics(), {})
    assert q.tolist() == [SAFE, SAFE, UNKNOWN, UNKNOWN]


def test_unknown_current_choice_is_not_called_correct():
    logits = torch.tensor([[.1,.4,-2.]],requires_grad=True)
    loss,info = policy_regret_loss(logits,torch.ones_like(logits,dtype=torch.bool),
        torch.tensor([[0.,3.,float('nan')]]),qualification=torch.tensor([[SAFE,UNKNOWN,UNKNOWN]]))
    assert loss == 0 and info['correct'] == 0 and info['unqualified'] == 1


def test_conflicting_alias_measurements_fail_closed():
    with pytest.raises(ValueError, match='conflicting'):
        qualify_actions(np.ones(3,bool), {0:[1],1:[1]}, 0, metrics(), {1:metrics(completed=2199)})


@pytest.mark.parametrize('candidate', [metrics(power=float('nan')), metrics(sla=-1)])
def test_invalid_measurements_cannot_qualify(candidate):
    with pytest.raises(ValueError):
        qualify_actions(np.ones(3,bool), {0:[1],1:[2]}, 0, metrics(), {1:candidate})


def test_capture_records_actual_sla_union_at_episode_end():
    from types import SimpleNamespace
    env=SimpleNamespace(done=True,ep_stats={'completed':2200,'rack_viol':0},
                        sla_stats=lambda:{'sla_viol':1270})
    assert capture_outcome(env)==metrics(sla=1270)
    env.done=False
    with pytest.raises(ValueError):capture_outcome(env)


def test_legacy_partial_bank_preserves_evidence_without_inventing_sla():
    case=dict(mask=np.ones(3,bool),plans={0:[1],1:[2]},reference_action=0,
        reference_stats=dict(completed=2200,rack_viol=0),
        branch_rows=[dict(action=1,completed=2199,power=0)])
    q,reasons=qualify_case(case)
    assert q.tolist()==[SAFE,UNSAFE,UNKNOWN] and reasons[1]==['completed']


def test_new_complete_bank_can_qualify_measured_improvements():
    case=dict(mask=np.ones(3,bool),plans={0:[1],1:[2]},reference_action=0,
        reference_outcome=metrics(),branch_rows=[dict(action=1,constraint_metrics=metrics())])
    q,_=qualify_case(case)
    assert q.tolist()==[SAFE,SAFE,UNKNOWN]
