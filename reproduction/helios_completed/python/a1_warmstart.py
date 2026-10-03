"""Explicit full-state warm start for the separate A1 delay-credit diagnostic."""


def migrate_state(state, descriptor, simulator_semantics, parent_episode):
    base=dict(descriptor)
    if base.pop('a1_delay_credit',None) is None:
        raise ValueError('target does not enable the delay-credit recipe')
    temporal=base.pop('a1_temporal_observation',None) is not None
    if (state.get('schema_version')!=2 or state.get('episode')!=parent_episode
            or state.get('semantic_version')!='helios-a4-reward-cf-v2'
            or state.get('simulator_semantics')!=simulator_semantics
            or state.get('relief_credit')!=base):
        raise ValueError('parent semantics, episode, or credit differs')
    if not all(isinstance(state.get(k),dict) for k in
               ('actors','optimizers','rng','sla','critic','a4_return_critic')):
        raise ValueError('full training state is missing')
    result=dict(state)
    result['semantic_version']='helios-a1-temporal-credit-v1' if temporal else 'helios-a1-delay-credit-v1'
    result['relief_credit']=dict(descriptor)
    result['migration']=dict(parent_semantics=state['semantic_version'],parent_episode=parent_episode,
        reason='explicit training-only A1 delay-credit diagnostic; not ordinary resume')
    return result
