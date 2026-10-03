"""Correct demonstrated deployment regret using frozen-policy return labels."""
import numpy as np
import torch
import torch.nn.functional as F


def deployed_actions(logits,masks):
    masked=logits.masked_fill(~masks.bool(),-torch.inf)
    admit=torch.logsumexp(masked[:,:-1],dim=1)>=masked[:,-1]
    rack=masked[:,:-1].argmax(dim=1)
    return torch.where(admit,rack,torch.full_like(rack,logits.shape[1]-1))


def policy_regret_loss(logits,masks,returns,eps=1e-6,qualification=None):
    if logits.shape!=masks.shape or logits.shape!=returns.shape or logits.ndim!=2:
        raise ValueError('logits, mask and labels must have identical batch/action shapes')
    masks=masks.bool();queried=torch.isfinite(returns)
    if not bool(masks.any(dim=1).all()) or bool((queried & ~masks).any()):
        raise ValueError('empty action set or illegal branch label')
    if bool(queried[:,-1].any()):raise ValueError('routing labels may not supervise DEFER')
    if not bool(torch.isfinite(logits[masks]).all()):raise ValueError('non-finite legal logit')
    current=deployed_actions(logits,masks);rows=torch.arange(len(logits),device=logits.device)
    known=queried[rows,current];defer=current==logits.shape[1]-1
    if qualification is None:
        qualified=queried
        current_safe=known
        current_unsafe=torch.zeros_like(known)
        unqualified=torch.zeros_like(known)
    else:
        if (qualification.shape!=returns.shape or
            not bool(((qualification>=-1)&(qualification<=1)&(qualification==qualification.round())).all())):
            raise ValueError('invalid constraint qualification')
        qualified=queried & (qualification==1)
        current_safe=known & (qualification[rows,current]==1)
        current_unsafe=known & (qualification[rows,current]==-1)
        unqualified=known & (qualification[rows,current]==0) & ~defer
    safe=torch.where(qualified,returns,torch.full_like(returns,-torch.inf))
    best_value=safe.max(dim=1).values
    current_value=torch.where(known,returns[rows,current],torch.zeros_like(best_value))
    gap=best_value-current_value
    active=~defer & qualified.any(1) & (current_unsafe | (current_safe & (gap>eps)))
    info=dict(active=int(active.sum()),correct=int((current_safe & ~defer & (gap<=eps)).sum()),
        unqueried=int((~known & ~defer).sum()),defer=int(defer.sum()))
    if qualification is not None:
        info.update(unsafe=int((current_unsafe & ~defer).sum()),unqualified=int(unqualified.sum()))
    if not bool(active.any()):
        zero=torch.where(torch.isfinite(logits),logits,torch.zeros_like(logits)).sum()*0
        return zero,info
    # Supervise every tied-best measured anchor, not an arbitrary array index.
    # Average within each state so alias multiplicity does not weight states.
    # Unknown choices (including DEFER) remain outside the routing objective.
    a=rows[active];c=current[active]
    best_mask=qualified[a] & (returns[a]>=best_value[a,None]-eps)
    target_logits=torch.where(best_mask,logits[a],logits[a,c,None])
    penalties=F.softplus(logits[a,c,None]-target_logits)
    per_state=torch.where(best_mask,penalties,torch.zeros_like(penalties)).sum(1)/best_mask.sum(1)
    weight=torch.where(current_unsafe[active],torch.ones_like(gap[active]),gap[active].clamp(max=1))
    loss=(weight.to(logits.dtype)*per_state).mean()
    return loss,info


def masked_reference_kl(logits,reference_logits,masks):
    legal=masks.bool()
    lp=F.log_softmax(logits.masked_fill(~legal,-torch.inf),dim=-1)
    ref=F.log_softmax(reference_logits.detach().masked_fill(~legal,-torch.inf),dim=-1)
    safe_lp=torch.where(legal,lp,torch.zeros_like(lp))
    safe_ref=torch.where(legal,ref,torch.zeros_like(ref))
    return (torch.exp(ref)*(safe_ref-safe_lp)).sum(dim=1).mean()


def select_candidates(reference_action,mask,plans,projected,preferred=None,max_actions=3):
    nr=len(mask)-1
    if not 0<=reference_action<nr or not mask[reference_action] or reference_action not in plans:
        raise ValueError('candidate set requires a legal deployment placement')
    selected=[int(reference_action)];used={tuple(plans[reference_action])}
    def add(a):
        if a in plans and a<nr and mask[a] and tuple(plans[a]) not in used:
            selected.append(int(a));used.add(tuple(plans[a]))
    if preferred is not None:add(int(preferred))
    ref=np.asarray(projected[reference_action],dtype=float)
    ranked=sorted((a for a in plans if a<nr and mask[a]),
        key=lambda a:(-float(np.linalg.norm(np.asarray(projected[a])-ref)),a))
    for a in ranked:
        if len(selected)>=max_actions:break
        add(a)
    return selected


def validate_bank(bank,teacher_sha,source_sha):
    meta=bank.get('metadata',{})
    if (meta.get('schema')!='a4-policy-regret-bank-v2' or meta.get('teacher_sha')!=teacher_sha
        or meta.get('source_sha')!=source_sha or meta.get('gamma')!=.99
        or meta.get('horizon')!=180 or meta.get('sla')!=[1.,1.,1.]):
        raise ValueError('branch bank policy, source or return semantics differ')
    for case in bank.get('cases',[]):
        mask=np.asarray(case['mask'],dtype=bool);values=np.asarray(case['returns'])
        a=int(case['reference_action'])
        if (values.shape!=mask.shape or not 0<=a<len(mask)-1 or not mask[a]
            or not np.isfinite(values[a]) or not np.isfinite(case['obs']).all()
            or np.any(np.isfinite(values)&~mask) or np.isfinite(values[-1])
            or not isinstance(case.get('tape'),list)):
            raise ValueError('branch case lacks a legal reference or replay data')
