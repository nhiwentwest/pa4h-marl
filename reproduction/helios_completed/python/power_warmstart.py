"""Explicit, bounded migration of the selected episode-24 parent to a new recipe.

This is not an ordinary resume: the legal action set changes. Preserve the old
artifact and record parent semantics in the new state. Normal v3 loading rejects
v2 checkpoints without this conversion.
"""


def migrate_state(state, descriptor, simulator_semantics):
    base=dict(descriptor)
    constraint=base.pop('power_constraint',None)
    if constraint is None:
        raise ValueError('target recipe does not enable the power constraint')
    if (state.get('schema_version')!=2 or state.get('episode')!=24
            or state.get('semantic_version')!='helios-a4-reward-cf-v2'
            or state.get('simulator_semantics')!=simulator_semantics
            or state.get('relief_credit')!=base):
        raise ValueError('parent differs from the selected episode-24 soft-power recipe')
    if not all(isinstance(state.get(k),dict) for k in
               ('actors','optimizers','rng','sla','critic','a4_return_critic')):
        raise ValueError('parent lacks full training state')
    result=dict(state)
    result['semantic_version']='helios-a4-reward-cf-v3-hard-power'
    result['relief_credit']=dict(descriptor)
    result['migration']=dict(parent_semantics=state['semantic_version'],parent_episode=24,
        reason='explicit new hard-power recipe; full-state warm start, not ordinary resume')
    return result
