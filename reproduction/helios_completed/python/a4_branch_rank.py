"""Sampled full-continuation routing labels, used only by the training loss."""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from branch_constraints import capture_outcome


def branch_rank_loss(logits, legal_masks, returns, eps=1e-6):
    legal=legal_masks.bool()
    labeled=torch.isfinite(returns)
    if bool((labeled & ~legal).any()):
        raise ValueError('a branch label names an illegal action')
    if bool((labeled & ~torch.isfinite(logits)).any()):
        raise ValueError('non-finite queried action logit')
    # NaN means unqueried, never illegal. Do not normalize away tiny gaps.
    safe=torch.where(labeled,returns,torch.zeros_like(returns))
    gap=safe[:,:,None]-safe[:,None,:]
    pairs=labeled[:,:,None] & labeled[:,None,:] & (gap>eps)
    n=int(pairs.sum())
    if not n:
        return torch.where(torch.isfinite(logits),logits,torch.zeros_like(logits)).sum()*0,0
    i,j,k=pairs.nonzero(as_tuple=True)
    loss=(gap[i,j,k].clamp(max=1).to(logits.dtype)*F.softplus(-(logits[i,j]-logits[i,k]))).mean()
    return loss,n


def migrate_branch_state(state, descriptor, simulator_semantics, parent_episode):
    base=dict(descriptor)
    rank=base.pop('a4_branch_rank',None)
    if rank is None or rank.get('schema')!='full-continuation-qualified-risk-v2':
        raise ValueError('branch credit missing from target recipe')
    expected=dict(base,a4_branch_rank=dict(rank,schema='sampled-full-continuation-v1'))
    if (state.get('schema_version')!=2 or state.get('episode')!=parent_episode
        or state.get('semantic_version')!='helios-a4-branch-rank-v1'
        or state.get('simulator_semantics')!=simulator_semantics
        or state.get('relief_credit')!=expected):
        raise ValueError('parent semantics or recipe differ')
    if not all(isinstance(state.get(k),dict) for k in ('actors','optimizers','rng','sla','critic','a4_return_critic')):
        raise ValueError('full training state missing')
    out=dict(state,semantic_version='helios-a4-constrained-risk-v2',relief_credit=dict(descriptor))
    out['migration']=dict(parent_semantics=state['semantic_version'],parent_episode=parent_episode,
        reason='explicit full-state warm start with measured completion/SLA/power branch risk')
    return out


def signature(name, obs, mask, step):
    return (name,int(step),hashlib.sha256(np.asarray(obs,dtype=np.float32).tobytes()+
        np.asarray(mask,dtype=bool).tobytes()).hexdigest())


class TapeRecorder:
    """Reservoir-sample a decision without touching the rollout's RNG."""
    def __init__(self, seed, min_step=10):
        self.rng=np.random.RandomState(seed)
        self.tape=[];self.case=None;self.seen=0
        self.min_step=int(min_step)

    def record(self,name,obs,mask,step,action,env):
        index=len(self.tape)
        self.tape.append((signature(name,obs,mask,step),int(action)))
        if name!='a4' or env is None or step<self.min_step or not hasattr(env,'_branch_job'):
            return
        job=env._branch_job
        legal=np.flatnonzero(mask[:-1]);plans={};all_plans={}
        for a in legal:
            plan=env._pick_hosts_pa(int(a),job)
            if plan is not None:
                all_plans[int(a)]=list(plan)
                plans.setdefault(tuple(plan),int(a))
        if len(plans)<2:return
        candidates=list(plans.values())
        if action in legal:
            key=tuple(env._pick_hosts_pa(int(action),job))
            candidates.remove(plans[key]);candidates.insert(0,int(action))
        else:
            # Include deployment's preferred legal anchor, no DEFER label.
            candidates.sort()
        first=candidates[:1];others=candidates[1:]
        self.rng.shuffle(others);candidates=first+others[:2]
        self.seen+=1
        if self.rng.randint(self.seen)==0:
            self.case=dict(index=index,step=int(step),obs=np.asarray(obs,dtype=np.float32).copy(),
                mask=np.asarray(mask,dtype=bool).copy(),candidates=candidates,job_id=str(job.job_id),plans=all_plans)


def deployment_act(trainer,ag,env):
    from grouped_admission import select_grouped_admission
    from completion_victim import completion_preserving_mask
    base=trainer.make_deterministic_act(ag,'pointwise');current={}
    def act(name,obs,mask,step,**kwargs):
        if name=='a4':
            if mask.sum()==1:return int(np.flatnonzero(mask)[0])
            return select_grouped_admission(trainer.actor_probabilities(ag.actors[name],obs,mask),mask)
        if name=='a2':
            action=base(name,obs,mask,step,**kwargs);current['rack']=action;return action
        if name=='a3':
            candidates=env.jobs_on_rack(current['rack'])[:4]
            preferred=np.asarray(completion_preserving_mask(mask,[env.running_remaining(j) for j in candidates],step,env.max_steps),dtype=bool)
            return base(name,obs,preferred,step,**kwargs)
        return base(name,obs,mask,step,**kwargs)
    act.current=current
    return act


def replay_branch(trainer,ag,env,tape,case,chosen,replay_suffix=False):
    """Replay every pre-target actor call, then force exactly one legal action."""
    env.reset();base=deployment_act(trainer,ag,env);cursor=0;hit=False;ret=0.;discount=1.
    def act(name,obs,mask,step,**kwargs):
        nonlocal cursor,hit
        if cursor<=case['index'] or (replay_suffix and cursor<len(tape)):
            sig,original=tape[cursor]
            if signature(name,obs,mask,step)!=sig:raise RuntimeError('branch prefix diverged')
            if cursor==case['index']:
                if name!='a4' or not mask[chosen]:raise ValueError('invalid forced routing action')
                action=chosen;hit=True
            else:action=original
            # Recreate decoder state for an A3/A1 call at the branching step.
            if name=="a2":base.current["rack"]=action
            cursor+=1
            return action
        return base(name,obs,mask,step,**kwargs)
    for step in range(env.max_steps):
        trainer.gang_step(env,ag,step,act,collect_credit=True)
        reward=env.advance()['reward']*trainer.REWARD_SCALE
        if step>=case['step']:
            ret+=discount*reward;discount*=trainer.GAMMA
        if env.done:break
    if not hit:raise RuntimeError('branch decision was not replayed')
    if not np.isfinite(ret):raise FloatingPointError('non-finite branch return')
    return float(ret),dict(env.ep_stats,constraint_metrics=capture_outcome(env))
