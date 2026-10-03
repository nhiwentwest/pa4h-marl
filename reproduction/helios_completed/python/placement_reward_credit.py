"""Full-information pre-relief placement reward contributions.

All coefficients and normalizations match GangEnv.advance. This is a one-step
teacher, not the return after downstream relief, later placements, network delay,
fragmentation or terminal SLA. Those consequences remain in PPO's team return.
No peak-pressure ranking, fixed DEFER bonus or entropy target on tied anchors.
"""
import numpy as np
import torch
from gang_env import (W_SERVE,W_WAIT,W_ENERGY,W_VIOL,W_SLA_BASE,W_DEADLINE,
                      NREL_IDLE_W,NREL_PEAK_W,NET_COMM_TAX,SLA_MAX_WAIT_STEPS,
                      SLA_MAX_RESTART_WAIT_STEPS,ring_cross_rack_fraction)


def placement_reward_utilities(env, job, mask):
    legal=np.asarray(mask,dtype=bool)
    u=np.full(env.num_racks+1,-np.inf,dtype=np.float32)
    u[-1]=0.0
    if not legal[:-1].any():
        return u
    capacity=max(1,env.num_hosts)
    base=[env._rack_base_fine(r) for r in range(env.num_racks)]
    before_risk=sum(np.mean(x>env.rack_budget) for x in base)/env.num_racks
    wait=max(0,env.step_idx-job.arrival_step)
    priority=float(job.priority)
    benefit=W_SERVE*job.num_nodes/capacity+W_WAIT*priority/(capacity*3.)
    # Only newly crossed, one-shot costs are avoided by placing now. No repeated
    # penalty for already recorded admission/restart/deadline breaches.
    first=env._first_wait.get(job.job_id)
    if (job.job_id not in env._admission_breached and
        (first is None or first>SLA_MAX_WAIT_STEPS) and
        wait<=SLA_MAX_WAIT_STEPS<wait+1):
        benefit+=(W_SLA_BASE+env.sla_lambda_adm)*priority/env._sla_priority_total
    if (job.job_id not in env._deadline_crossed_ids and
        wait<=SLA_MAX_WAIT_STEPS<wait+1):
        benefit+=W_DEADLINE/max(1,len(env._all_jobs))
    since=env._restart_wait_since.get(job.job_id)
    if since is not None and job.job_id not in env._restart_breached:
        cumulative=env._restart_wait_total.get(job.job_id,0)+env.step_idx-since
        if cumulative<=SLA_MAX_RESTART_WAIT_STEPS<cumulative+1:
            benefit+=(W_SLA_BASE+env.sla_lambda_restart)*priority/env._sla_priority_total
    for anchor in np.flatnonzero(legal[:-1]):
        hosts=env._pick_hosts_pa(int(anchor),job)
        if hosts is None:
            raise ValueError('legal mask disagrees with candidate placement')
        mult=1.+NET_COMM_TAX*ring_cross_rack_fraction(hosts)
        delta=env._fine_window(job,0)*mult-NREL_IDLE_W
        added=np.bincount([env.rack_of(h) for h in hosts],minlength=env.num_racks)
        risk=sum(np.mean((base[r]+added[r]*delta)>env.rack_budget)
                 for r in range(env.num_racks))/env.num_racks
        energy=job.num_nodes*float(np.mean(delta))/(capacity*NREL_PEAK_W)
        u[anchor]=benefit-W_ENERGY*energy-W_VIOL*(risk-before_risk)
    return u


def expected_reward_regret(logits, utility, masks, epsilon=1e-6):
    legal=masks.bool() & torch.isfinite(utility)
    maximum=torch.where(legal,utility,torch.full_like(utility,-torch.inf)).max(-1).values
    minimum=torch.where(legal,utility,torch.full_like(utility,torch.inf)).min(-1).values
    informative=(legal.sum(-1)>1)&((maximum-minimum)>epsilon)
    if not bool(informative.any()):
        return torch.where(torch.isfinite(logits),logits,torch.zeros_like(logits)).sum()*0.,0
    safe_gap=torch.where(legal,maximum[:,None]-utility,torch.zeros_like(utility))
    probability=torch.softmax(logits.masked_fill(~legal,-torch.inf),dim=-1)
    regret=(probability*safe_gap).sum(-1)
    return regret[informative].mean(),int(informative.sum())
