"""Memoization and forced-action shortcuts that preserve policy behavior and RNG."""
import numpy as np
import torch


def install_plan_cache(env, verify_hits=64):
    original=env._pick_hosts_pa
    cache={};epoch=[None,None,None];stats=dict(hits=0,misses=0,verified=0)
    def cached(anchor,job):
        free=env.free_host_map()
        budget=getattr(env,'_feas_budget',None)
        if epoch[0]!=env.step_idx or epoch[1] is not free or epoch[2]!=budget:
            cache.clear();epoch[:]=[env.step_idx,free,budget]
        key=(int(anchor),id(job.per_node_trace_w),job.num_nodes,job.progress_steps)
        if key in cache:
            stats['hits']+=1;plan=cache[key]
            if stats['verified']<verify_hits:
                actual=original(anchor,job)
                assert plan==(tuple(actual) if actual is not None else None),'cached placement differs'
                stats['verified']+=1
        else:
            stats['misses']+=1;actual=original(anchor,job)
            plan=tuple(actual) if actual is not None else None;cache[key]=plan
        return list(plan) if plan is not None else None
    env._pick_hosts_pa=cached
    env.plan_cache_stats=stats
    return stats


def forced_action(actor,mask,mode):
    legal=np.flatnonzero(np.asarray(mask,dtype=bool))
    if len(legal)!=1 or mode not in ('train','eval'):
        raise ValueError('shortcut requires one legal action and a known mode')
    if mode=='train':
        device=next(actor.parameters()).device
        probabilities=torch.zeros(len(mask),dtype=torch.float32,device=device)
        probabilities[int(legal[0])]=1.
        # Use the same categorical operation and shape as the original actor:
        # stochastic RNG consumption is retained even for a forced action.
        action=int(torch.distributions.Categorical(probs=probabilities).sample().item())
        assert action==int(legal[0])
    return int(legal[0]),0.


def install_forced_action_shortcut(trainer):
    original=trainer.select_action
    def select(actor,obs,mask=None,mode='train'):
        if mask is not None and int(np.sum(mask))==1:
            return forced_action(actor,mask,mode)
        return original(actor,obs,mask,mode)
    trainer.select_action=select
