import pytest
from a1_warmstart import migrate_state


def state():
    return dict(schema_version=2,semantic_version='helios-a4-reward-cf-v2',episode=120,
        simulator_semantics='physical',relief_credit={'schema':'global-relief-v1'},
        actors={},optimizers={},rng={},sla={},critic={},a4_return_critic={})


def test_migration_is_explicit_and_preserves_parent_state():
    original=state();target={'schema':'global-relief-v1','a1_delay_credit':{'training_only':True}}
    new=migrate_state(original,target,'physical',120)
    assert original['semantic_version']=='helios-a4-reward-cf-v2'
    assert new['semantic_version']=='helios-a1-delay-credit-v1'
    assert new['actors']==original['actors'] and new['rng']==original['rng']


@pytest.mark.parametrize('change',[{'episode':24},{'schema_version':1},{'semantic_version':'wrong'},
    {'simulator_semantics':'wrong'},{'relief_credit':{}},{'rng':None}])
def test_other_parents_rejected(change):
    original=state();original.update(change)
    with pytest.raises(ValueError):migrate_state(original,{'schema':'global-relief-v1','a1_delay_credit':{}},'physical',120)


def test_target_must_enable_delay_credit():
    with pytest.raises(ValueError):migrate_state(state(),{'schema':'global-relief-v1'},'physical',120)


def test_temporal_feature_recipe_has_its_own_semantics():
    target={'schema':'global-relief-v1','a1_delay_credit':{},'a1_temporal_observation':{'shape_unchanged':True}}
    new=migrate_state(state(),target,'physical',120)
    assert new['semantic_version']=='helios-a1-temporal-credit-v1'
