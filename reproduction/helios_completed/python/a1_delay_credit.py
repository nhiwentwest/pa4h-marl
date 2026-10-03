"""Training-only A1 credit for postponing a likely next-step intervention.

Uses only currently running jobs and their known trace phases. It neither reads
future arrivals nor selects deployment actions. The bounded premium is a credit
approximation and must be qualified by actual scheduler replay before training.
"""
import math
import numpy as np
from relief_credit import ReliefCredit


def delay_adjusted_utility(current, violation_weight, future_delta, discount):
    if not math.isfinite(future_delta) or not 0<=discount<=1:
        raise ValueError('invalid next-step credit or discount')
    delta=current.delta_utility(violation_weight)
    premium=0.
    if delta<=0. and future_delta>0.:
        cost=max(0.,current.immediate_cost+current.sla_proxy_cost)
        premium=discount*min(cost,future_delta)
    return delta+premium,premium


def projected_next_credit(env, victim, current, *, idle_w, peak_w, energy_weight,
                          network_tax, rack_size, rack_filter=None):
    if (env.running_remaining(victim)<=1 or
            env.step_idx+1>=getattr(env,'max_steps',float('inf'))):
        return None
    relief=0.;energy=0.
    racks={env.rack_of(h) for h in victim.placed_hosts}
    if rack_filter is not None:
        racks &= {rack_filter}
    for rack in racks:
        nodes=min(rack_size,env.num_hosts-rack*rack_size)
        elapsed=max(0,env.step_idx-victim.arrival_step_placed)+1
        victim_power=env._fine_window(victim,elapsed)
        total=np.full_like(victim_power,nodes*idle_w,dtype=np.float64)
        for job in env.jobs_on_rack(rack):
            if env.running_remaining(job)<=1:continue
            hosts=list(job.placed_hosts)
            cross=(sum(env.rack_of(h)!=env.rack_of(hosts[(i+1)%len(hosts)])
                       for i,h in enumerate(hosts))/len(hosts)) if len(hosts)>1 else 0.
            power=env._fine_window(job,max(0,env.step_idx-job.arrival_step_placed)+1)
            n=sum(env.rack_of(h)==rack for h in hosts)
            total+=n*(power*(1.+network_tax*cross)-idle_w)
        hosts=list(victim.placed_hosts)
        cross=(sum(env.rack_of(h)!=env.rack_of(hosts[(i+1)%len(hosts)])
                   for i,h in enumerate(hosts))/len(hosts)) if len(hosts)>1 else 0.
        n=sum(env.rack_of(h)==rack for h in hosts)
        removed=n*(victim_power*(1.+network_tax*cross)-idle_w)
        after=np.maximum(0.,total-removed)
        relief+=float(np.mean(total>env.rack_budget)-np.mean(after>env.rack_budget))
        energy+=float(np.mean(np.maximum(0.,removed)))
    return ReliefCredit(relief/env.num_racks,
        energy_weight*energy/max(1e-6,env.num_hosts*peak_w),
        current.immediate_cost,current.sla_proxy_cost)
