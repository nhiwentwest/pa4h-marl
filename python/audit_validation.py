"""Small validation audit; no training and no test-window access.

Run against a frozen code snapshot with the same environment variables as eval.
The temporary CSV only removes rows the original queue builder would exclude.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


def main():
    original_source = Path(os.environ["POD_HOURLY_JOBS"])
    start = int(os.environ["WORKLOAD_START_HOUR"])
    window = int(os.environ["WORKLOAD_WINDOW_HOURS"])
    if start not in (980, 990, 1010) or window != 10:
        raise ValueError("This diagnostic is restricted to the declared validation windows")
    frame = pd.read_csv(original_source)
    subset = frame[(frame.arrival_hour >= start) & (frame.arrival_hour < start + window)]
    with tempfile.TemporaryDirectory(prefix="pa4h_validation_audit_") as td:
        subset_path = Path(td) / "window.csv"
        subset.to_csv(subset_path, index=False)
        os.environ["POD_HOURLY_JOBS"] = str(subset_path)
        # Import after selecting the equivalent window CSV, before env constants load.
        import eval_gang as evaluation
        evaluation.TAG = os.environ.get("AUDIT_TAG", evaluation.TAG)
        evaluation.DECODER = os.environ.get("AUDIT_DECODER", evaluation.DECODER)
        import gang_env as ge
        from marl_gang_train import gang_step, make_deterministic_act

        env = evaluation.build_eval_env()
        evaluation.load_sla_multipliers(env)
        agents, missing = evaluation.load_agents(env)
        if missing:
            raise RuntimeError(f"Missing checkpoints: {missing}")
        horizon = env.max_steps
        jobs = [dict(id=j.job_id, arrival=j.arrival_step, duration=j.duration_steps,
                     nodes=j.num_nodes, earliest_finish=j.arrival_step + j.duration_steps,
                     profile=j.profile_path) for j in env._all_jobs]
        impossible = {j["id"] for j in jobs if j["earliest_finish"] > horizon}
        metadata = {j["id"]: j for j in jobs}
        events, predictions, decisions = [], [], []
        original_reset, original_place = env.reset, env.place_job
        original_preempt, original_advance = env.preempt_job, env.advance

        def shadow_peaks():
            """Mirror Java's ring-boundary power multiplier, for diagnostic comparison."""
            fine = np.zeros((env.num_racks, env._fine_per_step))
            for rack in range(env.num_racks):
                fine[rack] = min(ge.RACK_SIZE, env.num_hosts - rack * ge.RACK_SIZE) * ge.NREL_IDLE_W
            for job in env.running.values():
                hosts = list(job.placed_hosts)
                boundaries = sum(hosts[i] // ge.RACK_SIZE != hosts[(i + 1) % len(hosts)] // ge.RACK_SIZE
                                 for i in range(len(hosts)))
                multiplier = 1.0 + ge.NET_COMM_TAX * boundaries / len(hosts)
                power = env._fine_window(job, env.step_idx - job.arrival_step_placed) * multiplier
                for host in hosts:
                    fine[host // ge.RACK_SIZE] += power - ge.NREL_IDLE_W
            return fine.max(axis=1)

        def reset():
            events.clear()
            predictions.clear()
            decisions.clear()
            return original_reset()

        def place(job, anchor_rack, power_aware=True):
            ok = original_place(job, anchor_rack, power_aware=power_aware)
            if ok:
                events.append(dict(kind="place", step=env.step_idx, job=job.job_id,
                                   hosts=list(job.placed_hosts), duration=job.duration_steps,
                                   remaining_duration=getattr(job, "remaining_duration_steps", job.duration_steps),
                                   progress_steps=getattr(job, "progress_steps", 0)))
            return ok

        def preempt(job):
            projected = np.array([env.project_rack_peak(r) for r in range(env.num_racks)])
            shadow = shadow_peaks()
            events.append(dict(kind="preempt", step=env.step_idx, job=job.job_id,
                               elapsed=env.step_idx - job.arrival_step_placed,
                               duration=job.duration_steps, nodes=job.num_nodes,
                               projected_risky_racks=int((projected > env.rack_budget).sum()),
                               shadow_risky_racks=int((shadow > env.rack_budget).sum()),
                               projected_peak=float(projected.max()), shadow_peak=float(shadow.max())))
            return original_preempt(job)

        def advance():
            projected = np.array([env.project_rack_peak(r) for r in range(env.num_racks)])
            shadow = shadow_peaks()
            result = original_advance()
            observed = np.array(env.bridge.getRackPeakW(), dtype=float)
            predictions.append(dict(step=env.step_idx - 1,
                                    violation_samples=int(result["rack_viol"]),
                                    violation_samples_by_rack=[int(x) for x in result["rack_viol_vec"]],
                                    shadow_error_w=float(np.max(np.abs(shadow - observed))),
                                    projected_error_w=float(np.max(np.abs(projected - observed))),
                                    violated_racks=np.flatnonzero(observed > env.rack_budget).tolist(),
                                    max_budget_excess_w=float(max(0.0, observed.max() - env.rack_budget)),
                                    false_positive_racks=int(((projected > env.rack_budget) &
                                                             (observed <= env.rack_budget)).sum()),
                                    false_negative_racks=int(((projected <= env.rack_budget) &
                                                             (observed > env.rack_budget)).sum())))
            return result

        env.reset, env.place_job = reset, place
        env.preempt_job, env.advance = preempt, advance
        decoder = evaluation.resolve_decoder()

        class DecisionPolicy:
            def __init__(self, mode):
                self.mode = mode
                self.base = make_deterministic_act(agents, a4_mode=decoder)

            def step(self, current_env, t):
                selected_rack = None
                selected_job = None
                def act(name, obs, mask, step, credit_context=None, **kwargs):
                    nonlocal selected_rack, selected_job
                    if self.mode == "cf_greedy" or (self.mode == "a2_greedy" and name == "a2"):
                        utility = np.asarray(credit_context["utility"])
                        chosen = int(np.argmax(np.where(mask, utility, -np.inf)))
                    elif self.mode == "a1_wait" and name == "a1":
                        chosen = 0
                    elif self.mode == "a1_preempt" and name == "a1":
                        chosen = 1
                    else:
                        chosen = self.base(name, obs, mask, step, credit_context=credit_context, **kwargs)
                    if name == "a2":
                        selected_rack = int(chosen)
                    if name in ("a2", "a3", "a1"):
                        item = dict(step=step, agent=name, chosen=int(chosen),
                                    legal=np.flatnonzero(mask).tolist(),
                                    utility=np.asarray(credit_context["utility"]).tolist())
                        if name == "a3":
                            item["rack"] = selected_rack
                            candidates = current_env.jobs_on_rack(selected_rack)[:ge.RACK_SIZE]
                            item["candidate_jobs"] = [j.job_id for j in candidates]
                            selected_job = candidates[chosen]
                        if (name == "a1" and selected_job is not None and
                                os.environ.get("AUDIT_RELIEF_DETAILS") == "1"):
                            import marl_gang_train as training
                            job = selected_job
                            before = np.array([current_env.project_rack_violation_fraction(r)
                                               for r in range(current_env.num_racks)])
                            after, credit = training.build_relief_candidate_credit(
                                current_env, job, before, current_env.preemption_sla_cost(job))
                            item["credit_details"] = dict(
                                rack=selected_rack, job=job.job_id, nodes=job.num_nodes,
                                hosts=list(job.placed_hosts), before=before.tolist(), after=after.tolist(),
                                features=credit.features().tolist(),
                                delta_utility=credit.delta_utility(training.A1_VIOL_WEIGHT),
                                local_relief=float(before[selected_rack]-after[selected_rack]),
                                a1_action=int(chosen), a1_utility=np.asarray(credit_context["utility"]).tolist(),
                                immediate_cost=credit.immediate_cost,
                                energy_benefit=credit.energy_benefit,
                                sla_proxy_cost=credit.sla_proxy_cost,
                                actor_observation=np.asarray(obs).tolist(),
                                flagged_power_gain=training.A1_VIOL_WEIGHT * (before[selected_rack]-after[selected_rack]) / current_env.num_racks,
                                global_power_gain=training.A1_VIOL_WEIGHT * float((before-after).sum()) / current_env.num_racks,
                                global_team_weight_power_gain=ge.W_VIOL * float((before-after).sum()) / current_env.num_racks,
                                immediate_net_without_power_or_sla=credit.energy_benefit-credit.immediate_cost,
                                completion_already_breached=job.job_id in current_env._completion_late_ids,
                                restart_already_breached=job.job_id in current_env._restart_breached)
                        decisions.append(item)
                    return chosen
                gang_step(current_env, agents, t, act)

        output = dict(source=str(original_source), source_sha256=hashlib.sha256(original_source.read_bytes()).hexdigest(),
                      simulator_semantics=getattr(ge, "SIMULATOR_SEMANTICS", "gang-replay-v2-full-restart"),
                      validation_start_hour=start, horizon=horizon, jobs=jobs,
                      unavoidable_unfinished=sorted(impossible), completion_upper_bound=len(jobs)-len(impossible),
                      decoder=decoder, policies={})
        policies = [("MARL", DecisionPolicy("trained")), ("RPA", evaluation.RPA()),
                    ("EPOBF", evaluation.EPOBF()),
                    ("CF_GREEDY_DIAGNOSTIC", DecisionPolicy("cf_greedy")),
                    ("MARL_A1_WAIT_DIAGNOSTIC", DecisionPolicy("a1_wait")),
                    ("MARL_A1_PREEMPT_DIAGNOSTIC", DecisionPolicy("a1_preempt")),
                    ("MARL_A2_GREEDY_DIAGNOSTIC", DecisionPolicy("a2_greedy"))]
        requested = os.environ.get("AUDIT_POLICIES", "").strip()
        if requested:
            labels = requested.split(",")
            available = {label for label, _ in policies}
            if any(label not in available for label in labels):
                raise ValueError(f"Unknown audit policies: {labels}")
            policies = [(label, policy) for label, policy in policies if label in labels]
        for label, policy in policies:
            reward, stats, resource = evaluation.run_policy(env, policy)
            preemptions = [e for e in events if e["kind"] == "preempt"]
            dropped = set(metadata) - env._completed_ids
            record = dict(reward=reward, stats=stats, resource=resource,
                          actual_steps=env.step_idx, dropped_jobs=sorted(dropped),
                          avoidable_unfinished=sorted(dropped-impossible),
                          admission_late_jobs=sorted(env._admission_breached),
                          restart_late_jobs=sorted(env._restart_breached),
                          completion_late_jobs=sorted(env._completion_late_ids),
                          preemptions_per_job=dict(collections.Counter(e["job"] for e in preemptions)),
                          decisions=list(decisions),
                          preempts_without_shadow_power_risk=sum(e["shadow_risky_racks"] == 0 for e in preemptions),
                          elapsed_node_steps_before_preempt=sum(e["elapsed"] * e["nodes"] for e in preemptions),
                          max_shadow_error_w=max(p["shadow_error_w"] for p in predictions),
                          max_projection_error_w=max(p["projected_error_w"] for p in predictions),
                          projection_false_positive_rack_steps=sum(p["false_positive_racks"] for p in predictions),
                          projection_false_negative_rack_steps=sum(p["false_negative_racks"] for p in predictions),
                          events=list(events), predictions=list(predictions))
            output["policies"][label] = record
            print(json.dumps(dict(policy=label, completed=stats["sla"]["served"], sla=stats["sla"],
                                  violations=stats["rack_viol"], preempts=len(preemptions),
                                  false_risk_preempts=record["preempts_without_shadow_power_risk"],
                                  avoidable_unfinished=record["avoidable_unfinished"],
                                  max_shadow_error_w=record["max_shadow_error_w"])), flush=True)
        destination = Path(os.environ["AUDIT_OUTPUT"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(output, indent=2))
        print(f"Audit saved to {destination}", flush=True)


if __name__ == "__main__":
    main()
