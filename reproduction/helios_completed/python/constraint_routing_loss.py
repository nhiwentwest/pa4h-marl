"""Observed branch risk for A4, with one comparison per ordered host plan.

The PPO objective remains primary. This auxiliary loss only uses measured
continuations and never turns higher return with failed guardrails into a
positive target. It can learn from a harmful alternative before deployment
selects that alternative.
"""
import torch
import torch.nn.functional as F
from a4_policy_regret import policy_regret_loss


def constraint_routing_loss(logits, masks, returns, qualification, plans, eps=1e-6):
    _,old_info=policy_regret_loss(logits,masks,returns,eps=eps,qualification=qualification)
    if len(plans)!=len(logits):
        raise ValueError('one ordered-plan mapping is required per state')
    losses=[];safety_pairs=0;return_pairs=0
    for row,by_action in enumerate(plans):
        measured=torch.isfinite(returns[row]) & masks[row].bool()
        keys={}
        for action,hosts in by_action.items():
            a=int(action)
            if not 0<=a<len(logits[row])-1 or not bool(masks[row,a]):
                raise ValueError('invalid ordered-plan action')
            if bool(measured[a]):
                keys.setdefault(tuple(int(h) for h in hosts),[]).append(a)
        represented={a for indices in keys.values() for a in indices}
        if represented!={int(a) for a in measured.nonzero().flatten().tolist()}:
            raise ValueError('a measured placement lacks an ordered plan')
        groups=[]
        for actions in keys.values():
            statuses={int(qualification[row,a]) for a in actions}
            values=[float(returns[row,a]) for a in actions]
            if len(statuses)!=1 or max(values)-min(values)>eps:
                raise ValueError('aliases disagree on qualification or return')
            groups.append((actions,statuses.pop(),sum(values)/len(values),
                           torch.logsumexp(logits[row,actions],dim=0)))
        safe=[g for g in groups if g[1]==1]
        if not safe:continue
        best_value=max(g[2] for g in safe)
        best=[g for g in safe if g[2]>=best_value-eps]
        best_logmass=torch.logsumexp(torch.stack([g[3] for g in best]),dim=0)
        penalties=[]
        for actions,status,value,logmass in groups:
            if status==-1:
                weight=1.
                safety_pairs+=1
            elif status==1 and value<best_value-eps:
                weight=min(best_value-value,1.)
                return_pairs+=1
            else:
                continue
            penalties.append(weight*F.softplus(logmass-best_logmass))
        if penalties:losses.append(torch.stack(penalties).mean())
    info=dict(old_info,legacy_active=old_info['active'],active=len(losses),
              safety_pairs=safety_pairs,return_pairs=return_pairs)
    if losses:return torch.stack(losses).mean(),info
    return torch.where(torch.isfinite(logits),logits,torch.zeros_like(logits)).sum()*0,info
