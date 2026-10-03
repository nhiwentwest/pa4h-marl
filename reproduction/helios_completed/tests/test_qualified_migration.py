import pytest
from a4_branch_rank import migrate_branch_state

def fixture():
    rank=dict(schema='sampled-full-continuation-v1',training_only=True,gamma=.99,
        max_cases_per_batch=2,max_actions=3,gradient_cap=.25,
        suffix='frozen-grouped-pointwise-completion',query_mask_is_not_legality=True)
    target=dict(other='preserved',a4_branch_rank=dict(rank,schema='full-continuation-qualified-risk-v2'))
    state=dict(schema_version=2,episode=24,semantic_version='helios-a4-branch-rank-v1',
        simulator_semantics='sim',relief_credit=dict(other='preserved',a4_branch_rank=rank),
        **{k:{} for k in ('actors','optimizers','rng','sla','critic','a4_return_critic')})
    return state,target

def test_explicit_branch_parent_migration_preserves_full_state():
    state,target=fixture()
    result=migrate_branch_state(state,target,'sim',24)
    assert result['semantic_version']=='helios-a4-constrained-risk-v2'
    assert state['semantic_version']=='helios-a4-branch-rank-v1'
    for k in ('actors','optimizers','rng','sla','critic','a4_return_critic'):
        assert result[k] is state[k]

@pytest.mark.parametrize('change', ['schema_version','episode','semantic_version','simulator_semantics','relief_credit','optimizers'])
def test_migration_rejects_changed_parent(change):
    state,target=fixture();state[change]=None
    with pytest.raises(ValueError):migrate_branch_state(state,target,'sim',24)
