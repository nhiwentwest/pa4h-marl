"""Candidate credit for a placement with frozen current occupancy.

This is an analytical hypothesis, not the scheduler's actual counterfactual
return: other placements and relief are excluded. Branch replay must qualify
its ranking before it can supervise a policy. Future arrivals are never read.
"""
import numpy as np
from gang_env import (NREL_IDLE_W, NREL_PEAK_W, NET_COMM_TAX,
                      W_SERVE, W_WAIT, W_ENERGY, W_VIOL,
                      ring_cross_rack_fraction, RACK_SIZE)
from placement_reward_credit import placement_reward_utilities


def horizon_reward_utilities(env, job, mask, horizon=4, gamma=.99):
    if horizon < 1 or not 0 < gamma <= 1:
        raise ValueError('invalid horizon or discount')
    legal = np.asarray(mask, dtype=bool)
    out = placement_reward_utilities(env, job, legal).astype(np.float64)
    anchors = np.flatnonzero(legal[:-1])
    remaining = env.running_remaining(job)
    if not len(anchors) or min(horizon, env.max_steps-env.step_idx, remaining) <= 1:
        return out.astype(np.float32)
    plans = [env._pick_hosts_pa(int(a), job) for a in anchors]
    counts = np.array([np.bincount([env.rack_of(h) for h in plan],
                                  minlength=env.num_racks) for plan in plans])
    multipliers = np.array([1.+NET_COMM_TAX*ring_cross_rack_fraction(p) for p in plans])
    capacity = max(1, env.num_hosts)
    for offset in range(1, min(horizon, env.max_steps-env.step_idx, remaining)):
        idle = np.array([min(RACK_SIZE, env.num_hosts-r*RACK_SIZE)*NREL_IDLE_W
                         for r in range(env.num_racks)])
        base = np.broadcast_to(idle[:, None],
                               (env.num_racks, env._fine_per_step)).copy()
        for running in env.running.values():
            if env.running_remaining(running) <= offset:
                continue
            n = np.bincount([env.rack_of(h) for h in running.placed_hosts],
                            minlength=env.num_racks)
            elapsed = env.step_idx-running.arrival_step_placed+offset
            delta = env._fine_window(running, elapsed)*(
                1.+NET_COMM_TAX*ring_cross_rack_fraction(running.placed_hosts))-NREL_IDLE_W
            base += n[:, None]*delta[None, :]
        before = np.mean(base>env.rack_budget)
        delta = multipliers[:, None]*env._fine_window(job, offset)[None, :]-NREL_IDLE_W
        after = base[None, :, :]+counts[:, :, None]*delta[:, None, :]
        risks = np.mean(after>env.rack_budget, axis=(1,2))-before
        energy = job.num_nodes*np.mean(delta, axis=1)/(capacity*NREL_PEAK_W)
        benefit = W_SERVE*job.num_nodes/capacity+W_WAIT*float(job.priority)/(capacity*3.)
        out[anchors] += gamma**offset*(benefit-W_ENERGY*energy-W_VIOL*risks)
    return out.astype(np.float32)
