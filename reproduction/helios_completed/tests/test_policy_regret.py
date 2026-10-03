import numpy as np
import pytest
import torch
from a4_policy_regret import policy_regret_loss, select_candidates, validate_bank, masked_reference_kl


def test_already_best_case_has_zero_gradient():
    logits=torch.tensor([[0.,.2,.4,-2.]],requires_grad=True)
    mask=torch.ones_like(logits,dtype=torch.bool)
    returns=torch.tensor([[-82.85354,-82.81118,-82.00012,float('nan')]])
    loss,info=policy_regret_loss(logits,mask,returns)
    loss.backward()
    assert info['correct']==1 and info['active']==0 and loss==0
    assert torch.equal(logits.grad,torch.zeros_like(logits))


def test_wrong_choice_gets_directional_gradient_and_stops_when_correct():
    logits=torch.tensor([[.4,.1,0.,-2.]],requires_grad=True)
    mask=torch.ones_like(logits,dtype=torch.bool)
    values=torch.tensor([[0.,1.,float('nan'),float('nan')]])
    loss,info=policy_regret_loss(logits,mask,values);loss.backward()
    assert info['active']==1 and logits.grad[0,1]<0 and logits.grad[0,0]>0
    assert logits.grad[0,2]==0 and logits.grad[0,3]==0
    corrected=torch.tensor([[0.,.5,.1,-2.]],requires_grad=True)
    loss,info=policy_regret_loss(corrected,mask,values);loss.backward()
    assert info['active']==0 and torch.equal(corrected.grad,torch.zeros_like(corrected))


def test_unknown_current_action_and_defer_do_not_invent_targets():
    mask=torch.ones(2,4,dtype=torch.bool)
    logits=torch.tensor([[0.,.1,.4,-2.],[0.,.1,.2,5.]],requires_grad=True)
    values=torch.tensor([[0.,1.,float('nan'),float('nan')],[0.,1.,2.,float('nan')]])
    loss,info=policy_regret_loss(logits,mask,values)
    assert info['unqueried']==1 and info['defer']==1 and info['active']==0


def test_candidates_include_deployment_and_distinct_plans():
    mask=np.array([1,1,1,1,1],bool)
    plans={0:[0],1:[1],2:[1],3:[3]}
    projected=np.array([[1.,2.],[3.,0.],[3.,0.],[1.,1.]])
    selected=select_candidates(2,mask,plans,projected,preferred=0)
    assert selected[0]==2 and len(selected)==3 and len({tuple(plans[a]) for a in selected})==3
    with pytest.raises(ValueError):select_candidates(4,mask,plans,projected)


def test_masked_reference_kl_is_finite_with_zero_grad_on_padding():
    masks=torch.tensor([[1,1,0]],dtype=torch.bool)
    reference=torch.tensor([[.2,.1,float('-inf')]])
    logits=torch.tensor([[.1,.2,float('-inf')]],requires_grad=True)
    loss=masked_reference_kl(logits,reference,masks);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all() and logits.grad[0,2]==0


def test_bank_rejects_other_policy_or_missing_reference_action():
    metadata={'schema':'a4-policy-regret-bank-v2','teacher_sha':'policy','source_sha':'source','gamma':.99,'horizon':180,'sla':[1.,1.,1.]}
    case={'mask':np.ones(4,bool),'reference_action':0,'returns':np.array([1.,2.,np.nan,np.nan]),'obs':np.zeros(6),'tape':[]}
    bank={'metadata':metadata,'cases':[case]}
    validate_bank(bank,'policy','source')
    with pytest.raises(ValueError):validate_bank(bank,'other','source')
    case['reference_action']=2
    with pytest.raises(ValueError):validate_bank(bank,'policy','source')
