import numpy as np
import torch
from constraint_routing_loss import constraint_routing_loss

def evaluate(scores,values,qualification,plans):
    logits=torch.tensor([scores],dtype=torch.float64,requires_grad=True)
    masks=torch.isfinite(logits)
    loss,info=constraint_routing_loss(logits,masks,torch.tensor([values],dtype=torch.float64),
        torch.tensor([qualification]),[plans])
    loss.backward()
    return loss.detach(),logits.grad[0],info

def test_known_harm_trains_even_when_deployment_already_chooses_safe_reference():
    loss,grad,info=evaluate([.4,.2,.1,-2.],[-105.,-102.,-102.,np.nan],
        [1,-1,-1,0],{0:[0],1:[1],2:[1]})
    assert loss>0 and info['active']==1 and info['legacy_active']==0
    assert grad[0]<0 and bool((grad[1:3]>0).all()) and grad[3]==0
    assert info['safety_pairs']==1

def test_unknown_and_defer_have_zero_direct_routing_gradient():
    loss,grad,_=evaluate([.4,.2,5.,6.,-np.inf], [0.,3.,np.nan,np.nan,np.nan],
        [1,-1,0,0,0],{0:[0],1:[1],2:[2]})
    assert loss>0 and bool(torch.isfinite(grad).all())
    assert torch.equal(grad[2:],torch.zeros_like(grad[2:]))

def test_all_tied_qualified_targets_receive_gradient():
    _,grad,_=evaluate([.4,.1,.2,-2.],[0.,0.,3.,np.nan],
        [1,1,-1,0],{0:[0],1:[1],2:[2]})
    assert bool((grad[:2]<0).all()) and grad[2]>0

def test_alias_splitting_preserves_physical_plan_loss_and_total_gradient():
    base,grad,_=evaluate([.4,.2,-2.],[0.,3.,np.nan],[1,-1,0],{0:[0],1:[1]})
    split,sgrad,_=evaluate([.4,.2-np.log(2),.2-np.log(2),-2.],
        [0.,3.,3.,np.nan],[1,-1,-1,0],{0:[0],1:[1],2:[1]})
    torch.testing.assert_close(base,split)
    torch.testing.assert_close(grad[0],sgrad[0])
    torch.testing.assert_close(grad[1],sgrad[1:3].sum())

def test_different_ring_order_remains_a_different_plan():
    _,_,info=evaluate([.4,.2,.1,-2.],[0.,3.,3.,np.nan],
        [1,-1,-1,0],{0:[0,1],1:[2,3],2:[3,2]})
    assert info['safety_pairs']==2

def test_qualified_return_regret_still_trains_without_an_unsafe_branch():
    loss,grad,info=evaluate([.4,.1,-2.],[0.,3.,np.nan],
        [1,1,0],{0:[0],1:[1]})
    assert loss>0 and grad[0]>0 and grad[1]<0 and grad[2]==0
    assert info['return_pairs']==1

def test_unknown_metrics_and_tied_safe_states_have_no_invented_credit():
    for values,q in [([0.,3.,np.nan],[1,0,0]),([0.,0.,np.nan],[1,1,0])]:
        loss,grad,info=evaluate([.4,.1,-2.],values,q,{0:[0],1:[1]})
        assert loss==0 and info['active']==0 and torch.equal(grad,torch.zeros_like(grad))
