import numpy as np
import pytest
from power_constraint import PowerConstraintError, power_action_mask, require_power_feasible


def test_sparse_breach_cannot_be_traded_for_preemption_cost():
    # 43 fine samples are predictable even though the existing reward teacher prefers WAIT.
    mask=power_action_mask([True,True],43/1500,0.)
    assert mask.tolist()==[False,True]


def test_mask_retains_choices_when_safe():
    assert power_action_mask([True,True],0.,0.).tolist()==[True,True]
    assert power_action_mask([False,True],0.,0.).tolist()==[False,True]


@pytest.mark.parametrize('mask,before,after',[([True],.1,0),([False,False],.1,0),
    ([True,False],.1,0),([True,True],float('nan'),0),([True,True],.1,float('inf')),
    ([True,True],-.1,0),([True,True],.1,1.1),([True,True],.1,.1),([True,True],.1,.2)])
def test_invalid_or_noncausal_relief_fails_closed(mask,before,after):
    with pytest.raises((ValueError,PowerConstraintError)):
        power_action_mask(mask,before,after)


class Env:
    num_racks=2
    def __init__(self,risks):self.risks=risks
    def project_rack_violation_fraction(self,rack):return self.risks[rack]


def test_postcondition_rejects_residual_violation():
    require_power_feasible(Env([0.,0.]))
    with pytest.raises(PowerConstraintError,match='1'):
        require_power_feasible(Env([0.,1/1500]))


def test_nonfinite_projection_fails_closed():
    with pytest.raises(PowerConstraintError):require_power_feasible(Env([0.,float('nan')]))
