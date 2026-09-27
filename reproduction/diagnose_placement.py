"""Train-only attribution: unchanged seed 1 weights, isolated diagnostic outputs."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from eval_gang import build_eval_env, load_agents, load_sla_multipliers, RPA
from marl_gang_train import gang_step, make_deterministic_act

env = build_eval_env()
load_sla_multipliers(env)
agents, missing = load_agents(env)
assert not missing, missing
results = []
for variant in ('rpa', 'learned', 'a4_oracle', 'no_optional_defer', 'relief_oracle', 'all_oracle'):
    env.reset()
    base = make_deterministic_act(agents, a4_mode='pointwise')
    components = Counter()
    choices = Counter()
    reward = 0.0
    policy = RPA()

    def act(name, obs, mask, step, credit_context=None, **kwargs):
        if variant in ('a4_oracle', 'all_oracle') and name == 'a4' or variant in ('relief_oracle', 'all_oracle') and name != 'a4':
            action = int(np.argmax(np.where(mask, credit_context['utility'], -np.inf)))
        elif variant == 'no_optional_defer' and name == 'a4' and mask[:-1].any():
            restricted = mask.copy()
            restricted[-1] = False
            action = base(name, obs, restricted, step, **kwargs)
        else:
            action = base(name, obs, mask, step, **kwargs)
        if name == 'a4':
            choices['placements' if action < env.num_racks else 'defers'] += 1
            choices['optional_defers'] += int(action == env.num_racks and mask[:-1].any())
            u = credit_context['utility']
            choices['sum_cf_regret'] += float(np.max(u[mask]) - u[action])
        return action

    for step in range(env.max_steps):
        if variant == 'rpa':
            policy.step(env, step)
        else:
            gang_step(env, agents, step, act)
        metrics = env.advance()
        reward += metrics['reward']
        components.update(metrics['comp'])
        if env.done:
            break
    row = dict(variant=variant, reward=reward, components=dict(components),
               choices=dict(choices), stats=dict(env.ep_stats), sla=env.sla_stats())
    results.append(row)
    print(json.dumps(row), flush=True)
Path('outputs/helios_placement_attribution.json').write_text(json.dumps(results, indent=2)+'\n')
