import copy
import pytest
from power_warmstart import migrate_state


def parent():
    return dict(schema_version=2,semantic_version='helios-a4-reward-cf-v2',episode=24,
        simulator_semantics='physical-replay',relief_credit={'schema':'global-relief-v1'},
        actors={'a1':{'weights':[1,2]}},optimizers={'a1':{'moment':[3]}},
        rng={'numpy':'original','torch':'original'},sla={'lambda':1},critic={},a4_return_critic={})


def test_explicit_migration_preserves_weights_optimizers_rng_and_parent():
    old=parent();snapshot=copy.deepcopy(old)
    descriptor={'schema':'global-relief-v1','power_constraint':{'schema':'next-step-hard-power-v1'}}
    new=migrate_state(old,descriptor,'physical-replay')
    assert old==snapshot
    for key in ('actors','optimizers','rng','sla','critic','a4_return_critic'):
        assert new[key]==old[key]
    assert new['semantic_version']=='helios-a4-reward-cf-v3-hard-power'
    assert new['relief_credit']==descriptor
    assert new['migration']['parent_semantics']=='helios-a4-reward-cf-v2'


@pytest.mark.parametrize('change',[{'episode':132},{'schema_version':1},
    {'semantic_version':'legacy'},{'simulator_semantics':'different'},
    {'relief_credit':{'schema':'different'}},{'rng':None}])
def test_warmstart_rejects_mismatched_or_incomplete_parent(change):
    old=parent();old.update(change)
    with pytest.raises(ValueError):migrate_state(old,{'schema':'global-relief-v1','power_constraint':{}},'physical-replay')


def test_target_must_enable_constraint():
    with pytest.raises(ValueError):migrate_state(parent(),{'schema':'global-relief-v1'},'physical-replay')
