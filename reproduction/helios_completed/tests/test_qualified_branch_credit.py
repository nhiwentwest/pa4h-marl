import numpy as np
import pytest
import torch
from qualified_branch_credit import make_branch_label,qualified_branch_loss


def case():
    return dict(obs=np.array([.4,.2,.1,-2.],np.float32),mask=np.ones(4,bool),
                plans={0:[0],1:[1],2:[1]})


def test_trainer_boundary_rejects_reward_gain_that_drops_a_job():
    outcomes={0:dict(completed=2200,sla=19,power=0),1:dict(completed=2199,sla=20,power=0)}
    label=make_branch_label(case(),0,outcomes,np.array([-105.,-102.,np.nan,np.nan]))
    actor=torch.nn.Linear(4,4,bias=False)
    with torch.no_grad():actor.weight.copy_(torch.eye(4))
    class Masked(torch.nn.Module):
        def forward(self,x,mask):return actor(x).masked_fill(~mask,-torch.inf)
    loss,info=qualified_branch_loss(Masked(),[label],'cpu')
    loss.backward()
    assert info['active']==1 and loss>0 and info['legacy_active']==0
    assert actor.weight.grad[0,0]<0 and actor.weight.grad[1,0]>0
    assert torch.equal(actor.weight.grad[3],torch.zeros_like(actor.weight.grad[3]))
    assert label['qualification'].tolist()==[1,-1,-1,0]


def test_missing_metadata_is_not_accepted_by_trainer():
    with pytest.raises(ValueError,match='qualification'):
        qualified_branch_loss(None,[dict(obs=np.ones(4))],'cpu')


def test_reference_must_be_a_measured_continuation():
    with pytest.raises(ValueError,match='reference'):
        make_branch_label(case(),0,{},np.array([0.,1.,np.nan,np.nan]))

def test_full_labels_require_all_metrics_for_every_measured_branch():
    with pytest.raises(ValueError,match='complete'):
        make_branch_label(case(),0,{0:dict(completed=2200,power=0)},
                          np.array([0.,np.nan,np.nan,np.nan]))

def test_trainer_rejects_tampered_qualification():
    label=make_branch_label(case(),0,{0:dict(completed=2200,sla=19,power=0),
        1:dict(completed=2199,sla=19,power=0)},np.array([0.,3.,np.nan,np.nan]))
    label['qualification'][1]=1
    with pytest.raises(ValueError,match='qualification'):
        qualified_branch_loss(None,[label],'cpu')

def test_measured_return_requires_corresponding_outcome():
    with pytest.raises(ValueError,match='outcome'):
        make_branch_label(case(),0,{0:dict(completed=2200,sla=19,power=0)},
                          np.array([0.,3.,np.nan,np.nan]))
