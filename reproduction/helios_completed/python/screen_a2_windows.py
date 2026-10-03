"""Read-only replay of preregistered candidate training windows for A2 signal."""
import json
import os
from pathlib import Path

import numpy as np

from gang_env import SIMULATOR_SEMANTICS
from eval_gang import build_eval_env, load_agents, load_sla_multipliers
from marl_gang_train import gang_step, make_deterministic_act


SCREEN_SET = os.environ.get("A2_SCREEN_SET", "earlier")
if SCREEN_SET == "earlier":
    WINDOWS = (940, 950, 960, 970)
elif SCREEN_SET == "train_load":
    WINDOWS = (1000,)
elif SCREEN_SET == "stress_validation":
    WINDOWS = (980, 990)
else:
    raise ValueError("A2_SCREEN_SET must be earlier, train_load or stress_validation")
DECODER = os.environ.get("A2_SCREEN_DECODER", "sequence")
if DECODER not in ("sequence", "pointwise"):
    raise ValueError("A2_SCREEN_DECODER must be sequence or pointwise")


def main():
    if SIMULATOR_SEMANTICS != "gang-replay-v5-global-relief-credit":
        raise RuntimeError("A2 screen requires simulator v4")
    rows = []
    agents = None
    for start in WINDOWS:
        os.environ["WORKLOAD_START_HOUR"] = str(start)
        os.environ["WORKLOAD_WINDOW_HOURS"] = "10"
        env = build_eval_env()
        load_sla_multipliers(env)
        if agents is None:
            agents, missing = load_agents(env)
            if missing:
                raise FileNotFoundError(f"missing checkpoints: {missing}")
        env.reset()
        base_act = make_deterministic_act(agents, a4_mode=DECODER)
        multi_risk_steps = []
        risk_steps = 0

        def act(name, obs, mask, step, **kwargs):
            nonlocal risk_steps
            if name == "a2":
                risk_steps += 1
                legal = np.flatnonzero(mask).tolist()
                if len(legal) > 1:
                    multi_risk_steps.append(dict(step=step, racks=legal))
            return base_act(name, obs, mask, step, **kwargs)

        for step in range(env.max_steps):
            gang_step(env, agents, step, act)
            env.advance()
            if env.done:
                break
        row = dict(start_hour=start, end_hour=start + 10, decoder=DECODER,
                   subsample=float(os.environ["SUBSAMPLE"]),
                   jobs=len(env._all_jobs), risk_steps=risk_steps,
                   multi_risk_steps=multi_risk_steps,
                   a2_choices=env.ep_stats.get("a2_choices", 0),
                   a3_choices=env.ep_stats.get("a3_choices", 0),
                   rack_viol=env.ep_stats["rack_viol"],
                   preemptions=env.ep_stats["preempted"],
                   completed=env.ep_stats["completed"],
                   max_link_util_pct=100 * env.ep_stats["network_max_link_util"])
        rows.append(row)
        print(json.dumps(row), flush=True)
    output = Path(os.environ["A2_SCREEN_OUTPUT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(screen_set=SCREEN_SET, windows=WINDOWS,
                                     checkpoint_dir=os.environ["GANG_CHECKPOINT_DIR"],
                                     rows=rows), indent=2) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
