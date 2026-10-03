"""Versioned queue-risk events for the existing A2 -> A3 -> A1 actors.

This is an experimental extension, not the paper's power-only detector. No
future arrivals are read. Queue-only decisions receive team-return PPO credit;
the scheduling estimates below are observations, never counterfactual labels.
"""
import numpy as np

SCHEMA = 'queue-capacity-reservation-v1'


def descriptor():
    return dict(schema=SCHEMA, urgent_wait_steps=6, short_duration_steps=12,
                victim_min_nodes=2, victim_min_remaining=12, min_pause_steps=4,
                completion_reserve_steps=2, ageing_steps=36, loans_per_job=1,
                minimum_short_backlog_nodes='one cluster of physical hosts',
                reorder_only_when_pending_nodes_at_least_cluster=True,
                reorder_requires_active_loan_episode=True,
                local_supervision=False, actors_choose_all_relief_actions=True,
                admission_hold='DEFER until latest reserved restart; feasibility still required',
                future_arrivals_used=False, shape_unchanged=True)


def reset(env):
    env.queue_holds = {}
    env.queue_loaned = set()
    env.queue_relief_events = []


def held(env, job):
    return env.step_idx < env.queue_holds.get(job.job_id, -1)


def order_pending(env):
    if not env.queue_holds or sum(job.num_nodes for job in env.pending) < env.num_hosts:
        return
    from gang_env import SLA_MAX_WAIT_STEPS, SLA_COMPLETION_GRACE_STEPS
    def key(job):
        age = env.step_idx - job.arrival_step
        first = env._first_wait.get(job.job_id)
        admission_ok = (first <= SLA_MAX_WAIT_STEPS if first is not None
                        else age <= SLA_MAX_WAIT_STEPS)
        completion_ok = (env.step_idx + env.running_remaining(job) <=
                         job.arrival_step + job.duration_steps + SLA_COMPLETION_GRACE_STEPS)
        return (0 if age >= 36 else (1 if admission_ok and completion_ok else 2),
                job.arrival_step)
    env.pending.sort(key=key)


def eligible(env):
    short = [j for j in env.pending if j.job_id not in env.queue_holds and j.duration_steps <= 12]
    urgent = [j for j in short if env.step_idx-j.arrival_step >= 6]
    if not urgent or sum(j.num_nodes for j in short) < env.num_hosts:
        return []
    return [j for j in env.running.values()
            if j.job_id not in env.queue_loaned and j.num_nodes >= 2
            and env.running_remaining(j) >= 12
            and env.max_steps-2-env.step_idx-env.running_remaining(j) >= 4]


def _context(size):
    # Tied zero utilities provide no local teacher gradient. The explicit flag
    # selects full team GAE for these rows instead of power-local supervision.
    return dict(utility=np.zeros(size, dtype=np.float32), team_only=True,
                event_kind='queue_capacity')


def step(trainer, env, ag, t, act):
    candidates = eligible(env)
    if not candidates:
        return False
    eligible_ids = {j.job_id for j in candidates}
    rf = env.rack_features()
    risk = np.zeros(env.num_racks, dtype=np.float32)
    for job in candidates:
        score = (job.num_nodes * (env.max_steps-2-t-env.running_remaining(job)) /
                 max(1, env.num_hosts*env.max_steps))
        for host in job.placed_hosts:
            rack = env.rack_of(host)
            risk[rack] = max(risk[rack], score)
    mask2 = risk > 0
    rack_obs = (env.rack_history_features(rf).reshape(-1)
                if ag is not None and (getattr(ag, 'use_stgnn', False)
                                       or getattr(ag, 'use_history_mlp', False)) else rf.reshape(-1))
    rack = int(act('a2', np.concatenate([rack_obs, risk]), mask2, t,
                   credit_context=_context(env.num_racks), record=bool(mask2.sum()>1)))
    if not 0 <= rack < env.num_racks or not mask2[rack]:
        raise ValueError('A2 selected an ineligible queue-relief rack')
    jobs = env.jobs_on_rack(rack)[:trainer.RACK_SIZE]
    mask3 = np.zeros(trainer.RACK_SIZE, dtype=bool)
    slots = np.zeros(trainer.RACK_SIZE*trainer.A3_SLOT_DIM, dtype=np.float32)
    feature_map = {}
    for i, job in enumerate(jobs):
        if job.job_id not in eligible_ids:
            continue
        mask3[i] = True
        remaining = env.running_remaining(job)
        resume_at = env.max_steps-2-remaining
        share = job.num_nodes / max(1, env.num_hosts)
        late = float(resume_at+remaining > job.arrival_step+job.duration_steps+12)
        sla = (0.5+env.sla_lambda_comp)*float(job.priority)*late/env._sla_priority_total
        disruption = ((trainer.W_SERVE+trainer.W_CKPT)*share
                      + trainer.W_WAIT*float(job.priority)/max(1, env.num_hosts*3))
        features = np.array([share, trainer.W_ENERGY*env.project_job_energy_saved_normalized(job),
                             disruption, sla], dtype=np.float32)
        feature_map[i] = (features, resume_at, late)
        k = i*trainer.A3_SLOT_DIM
        slots[k:k+trainer.JOB_OBS_DIM] = env.job_features(job)
        slots[k+trainer.JOB_OBS_DIM] = share
        slots[k+trainer.JOB_OBS_DIM+1] = late/2
        slots[k+trainer.JOB_OBS_DIM+2:k+trainer.A3_SLOT_DIM] = features
    victim_index = int(act('a3', np.concatenate([rf[rack], slots]), mask3, t,
                           credit_context=_context(trainer.RACK_SIZE), record=bool(mask3.sum()>1)))
    if not 0 <= victim_index < len(jobs) or not mask3[victim_index]:
        raise ValueError('A3 selected an ineligible queue-relief victim')
    victim = jobs[victim_index]
    features, resume_at, late = feature_map[victim_index]
    pressure = min(1., sum(j.num_nodes for j in env.pending
                          if j.job_id not in env.queue_holds)/max(1, env.num_hosts))
    obs1 = np.concatenate([rf[rack], [len(env.pending)/50., pressure,
                           max(0., pressure-features[0]), features[0], late/2], features])
    decision = int(act('a1', obs1, np.ones(2, dtype=bool), t,
                       credit_context=_context(2), record=True))
    if decision not in (0, 1):
        raise ValueError('invalid A1 action')
    env.ep_stats['queue_relief_decisions'] = env.ep_stats.get('queue_relief_decisions', 0)+1
    if not decision:
        return False
    env.queue_holds[victim.job_id] = resume_at
    env.queue_loaned.add(victim.job_id)
    env.queue_relief_events.append(dict(step=t, job_id=victim.job_id, resume_at=resume_at,
                                       nodes=victim.num_nodes, remaining=env.running_remaining(victim)))
    env.preempt_job(victim)
    env.ep_stats['queue_relief_preemptions'] = env.ep_stats.get('queue_relief_preemptions', 0)+1
    return True


def migrate_state(state, target_descriptor, simulator_semantics, source_sha256):
    """Explicit same-shape warm start; never accept an unrelated checkpoint."""
    import copy
    expected = dict(target_descriptor)
    queue = expected.pop('queue_relief', None)
    if (queue != descriptor() or state.get('semantic_version') != 'helios-a4-completed-diverse-v3'
            or state.get('schema_version') != 2
            or state.get('simulator_semantics') != simulator_semantics
            or state.get('relief_credit') != expected
            or not isinstance(source_sha256, str) or len(source_sha256) != 64):
        raise ValueError('incompatible queue-relief warm start')
    for name in ('actors', 'critic', 'optimizers', 'rng', 'sla', 'a4_return_critic'):
        if not isinstance(state.get(name), dict):
            raise ValueError('incomplete parent training state')
    out = copy.deepcopy(state)
    out['semantic_version'] = 'helios-queue-relief-v4'
    out['relief_credit'] = copy.deepcopy(target_descriptor)
    out['a4_return_critic']['ready'] = False
    out['a4_return_critic']['quality_streak'] = 0
    out['queue_migration'] = dict(parent_episode=state['episode'], parent_semantics=state['semantic_version'],
                                  parent_sha256=source_sha256, fresh_initialization=False,
                                  reason='new queue-risk action contexts and explicit restart reservations')
    return out
