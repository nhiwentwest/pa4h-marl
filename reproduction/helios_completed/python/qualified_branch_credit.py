"""Constraint-aware branch labels used by the isolated Helios PPO trainer."""
import numpy as np
import torch
from constraint_routing_loss import constraint_routing_loss
from branch_constraints import qualify_actions, METRICS
from placement_labels import expand_alias_returns


def make_branch_label(case, reference_action, outcomes, returns):
    """All outcomes must come from the same prefix and continuation policy."""
    if reference_action not in outcomes:
        raise ValueError('missing reference continuation outcome')
    if any(set(METRICS)-set(value) for value in outcomes.values()):
        raise ValueError('full continuation labels require complete outcomes')
    plans=case['plans']
    raw=np.asarray(returns)
    if raw.shape!=np.asarray(case['mask']).shape or not np.isfinite(raw[reference_action]):
        raise ValueError('missing reference return')
    measured_plans={tuple(plans[a]) for a in outcomes}
    if any(a not in plans or tuple(plans[a]) not in measured_plans
           for a in np.flatnonzero(np.isfinite(raw))):
        raise ValueError('measured return lacks corresponding outcome')
    values=expand_alias_returns(returns,case['mask'],plans)
    qualification,reasons=qualify_actions(case['mask'],plans,reference_action,
                                         outcomes[reference_action],outcomes)
    return dict(obs=case['obs'],mask=case['mask'],returns=values,plans=plans,
                reference_action=reference_action,qualification=qualification,
                qualification_reasons=reasons,constraint_outcomes=outcomes,
                schema='full-continuation-qualified-v1')


def qualified_branch_loss(actor, labels, device):
    if any(label.get('schema')!='full-continuation-qualified-v1' for label in labels):
        raise ValueError('branch labels lack measured constraint qualification')
    for label in labels:
        checked=make_branch_label(label,label['reference_action'],label['constraint_outcomes'],label['returns'])
        if not np.array_equal(checked['qualification'],label['qualification']):
            raise ValueError('stored qualification differs from measured outcomes')
    obs=torch.as_tensor(np.stack([x['obs'] for x in labels]),device=device)
    masks=torch.as_tensor(np.stack([x['mask'] for x in labels]),device=device)
    values=torch.as_tensor(np.stack([x['returns'] for x in labels]),device=device)
    qualification=torch.as_tensor(np.stack([x['qualification'] for x in labels]),device=device)
    return constraint_routing_loss(actor(obs,masks),masks,values,qualification,
                                   [x['plans'] for x in labels])
