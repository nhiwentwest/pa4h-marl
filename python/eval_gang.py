"""
eval_gang.py — ONE harness, all policies, identical workload, full metric set.

Every policy runs on the SAME gang stepGang loop and the SAME env, so the numbers
are directly comparable. Metrics reported per policy:
  reward     — cooperative team reward (goodput − energy − viol − wait − ckpt)
  admit/comp — jobs placed / completed (goodput)
  preempt    — checkpoint-restart preemptions (churn)
  rackViol   — fine-grained rack power-budget violations ; viol% of rack-time
  SLA%       — priority-weighted deadline-miss rate (wait-to-place > SLA bound)
  wait       — avg steps a job waits before running (SLA latency)
  E_kWh      — cluster energy
  xrack      — CROSS-RACK FRACTION: mean fraction of a gang's nodes off its home
               rack (the network-locality KPI; lower = more rack-local, less
               cross-rack traffic + network-tax power)

Policies:
  RANDOM  — place on a random power-feasible rack (floor)
  SPREAD  — the ≤3-node rule (power-safe by count, phase/priority/network blind)
  EPOBF   — Best-Fit by power residual  (port of baselines/epobf_ft.py)
  RPA     — First-Fit + LIFO preempt on breach  (port of baselines/rpa.py)
  SAFERES  — lookahead reservation, LA steps  (port of baselines/safe_res_la50.py)
  MARL    — trained 4-agent CTDE policy (deterministic argmax), *_gang_best.pt,
            run through the SAME gang_step() the trainer uses (no divergence)

  PYTHONPATH=python python python/eval_gang.py <port> [--tag gang_best]
"""
import os
import sys
import json
import csv
import numpy as np
import torch

from gang_env import GangEnv, RACK_SIZE, JOB_OBS_DIM, NET_COMM_TAX, SIMULATOR_SEMANTICS
from marl_gang_train import (Agents, build_global, gang_step, make_deterministic_act,
                             relief_credit_metadata)
from models import select_action
from resource_metrics import ResourceMeter
from nrel_injection_bridge import STEPS_PER_HOUR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else \
    int(os.environ.get("BRIDGE_PORT", 25333))
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "gang_best"
DECODER = (sys.argv[sys.argv.index("--decoder") + 1]
           if "--decoder" in sys.argv else "auto")
SUB = float(os.environ.get("SUBSAMPLE", 0.05))
STEPS = int(os.environ.get("STEPS", 120))
SAFERES_LA = int(os.environ.get("SAFERES_LOOKAHEAD", 20))


def build_eval_env():
    """Build the shared evaluation environment for Alibaba or a selected window."""
    start_raw = os.environ.get("WORKLOAD_START_HOUR", "").strip()
    if not start_raw:
        return GangEnv(port=PORT, subsample=SUB, max_arrival_step=48,
                       max_steps=STEPS)

    window_raw = os.environ.get("WORKLOAD_WINDOW_HOURS", "").strip()
    if not window_raw or int(window_raw) <= 0:
        raise ValueError("WORKLOAD_WINDOW_HOURS must be positive")
    start_hour = int(start_raw)
    window_hours = int(window_raw)
    start_step = start_hour * STEPS_PER_HOUR
    end_step = (start_hour + window_hours) * STEPS_PER_HOUR
    env = GangEnv(port=PORT, subsample=SUB, max_steps=STEPS,
                  min_arrival_step=start_step, arrival_end_step=end_step)
    for job in env._all_jobs:
        job.arrival_step -= start_step
    print(f"[eval-workload] source={os.environ.get('POD_HOURLY_JOBS', 'default')} "
          f"hours=[{start_hour},{start_hour + window_hours}) jobs={len(env._all_jobs)}")
    return env


def resolve_decoder():
    if DECODER != "auto":
        return DECODER
    path = os.path.join(os.environ.get("GANG_CHECKPOINT_DIR", "."),
                        "deployment_decoder.json")
    if os.path.exists(path):
        with open(path) as f:
            name = json.load(f).get("name")
        if name in ("sequence", "pointwise"):
            return name
    # Backward-compatible checkpoints were trained/promoted by pointwise argmax.
    return "pointwise"


def est_power_w(job):
    """Conservative gang power estimate (peak-per-node × nodes), as the baselines use."""
    return job.peak_per_node_w * job.num_nodes


def shared_candidate_plans(env, job):
    """Legal A4 anchors under the same spill-enabled allocator as MARL.

    The returned host distribution is used by reservation baselines for their
    per-rack accounting.  Actual submission must still call ``place_job(...,
    power_aware=True)`` so every policy is subject to the same current NREL
    feasibility check at actuation time.
    """
    plans = []
    for anchor in range(env.num_racks):
        hosts = env._pick_hosts_pa(anchor, job)
        if hosts is None:
            continue
        counts = np.zeros(env.num_racks, dtype=np.int32)
        for host in hosts:
            counts[env.rack_of(host)] += 1
        plans.append((anchor, hosts, counts))
    return plans


# ----------------------- baselines (ported to gang_env) ------------------- #
class EPOBF:
    """Best-Fit by power residual: place on the rack with the smallest non-negative
    (budget − current_power − est_power) headroom. Coarse (peak estimate, no phase
    foresight) — that coarseness is why MARL/SafeRes can beat it. baselines/epobf_ft.py."""
    def step(self, env, t):
        for job in sorted(env.pending, key=lambda j: (j.arrival_step, -j.num_nodes)):
            best, best_res = None, 1e18
            for anchor, hosts, counts in shared_candidate_plans(env, job):
                peaks = [env.project_rack_peak(r, job, int(counts[r]), add_hosts=hosts)
                         if counts[r] else env.project_rack_peak(r)
                         for r in range(env.num_racks)]
                residual = sum(max(0.0, env.rack_budget - p) for p in peaks)
                if residual < best_res:
                    best, best_res = anchor, residual
            if best is not None:
                env.place_job(job, best, power_aware=True)


class RPA:
    """First-Fit on node availability (power-blind), avoiding the last-preempted rack;
    then LIFO-preempt newest jobs on any rack over budget. baselines/rpa.py."""
    def __init__(self):
        self.not_before, self.last_pre, self.start = {}, {}, {}

    def step(self, env, t):
        # mitigate first: preempt newest-first on racks currently over budget
        for r in range(env.num_racks):
            p = env.project_rack_peak(r)
            if p <= env.rack_budget:
                continue
            for v in sorted(env.jobs_on_rack(r),
                            key=lambda j: self.start.get(j.job_id, -1), reverse=True):
                env.preempt_job(v)
                self.last_pre[v.job_id] = r
                self.not_before[v.job_id] = t + 2      # cooldown
                p = env.project_rack_peak(r)
                if p <= env.rack_budget:
                    break
        # place: first-fit, avoid last-preempted rack
        for job in sorted(env.pending, key=lambda j: (j.arrival_step, -j.num_nodes)):
            if t < self.not_before.get(job.job_id, 0):
                continue
            cands = [anchor for anchor, _hosts, _counts
                     in shared_candidate_plans(env, job)]
            others = [r for r in cands if r != self.last_pre.get(job.job_id)]
            r = others[0] if others else (cands[0] if cands else None)
            if r is not None and env.place_job(job, r, power_aware=True):
                self.start[job.job_id] = t


class SafeRes:
    """Lookahead reservation: reserve a (rack × time) block where the job fits on both
    nodes and power for its whole duration; execute when the reserved start arrives.
    Uses current free/power as the future proxy. baselines/safe_res_la50.py."""
    def __init__(self, la):
        self.la = la
        self.rn = {r: {} for r in range(64)}   # reserved nodes  [rack][step]
        self.rp = {r: {} for r in range(64)}   # reserved power  [rack][step]
        self.plan = {}                          # job_id -> (start, anchor rack)

    def step(self, env, t):
        for job in sorted(env.pending, key=lambda j: (j.arrival_step, -j.num_nodes)):
            if job.job_id in self.plan:
                continue
            dur = job.duration_steps
            power = env.rack_power_w()
            candidates = shared_candidate_plans(env, job)
            placed = False
            for s0 in range(t, t + self.la):
                for anchor, _hosts, counts in candidates:
                    ok = True
                    for r, n_on_r in enumerate(counts):
                        if n_on_r == 0:
                            continue
                        free_r, pow_r = env.rack_free_nodes(r), power[r]
                        est_r = job.peak_per_node_w * int(n_on_r)
                        for s in range(s0, s0 + dur):
                            if free_r - self.rn[r].get(s, 0) < n_on_r:
                                ok = False; break
                            if pow_r + self.rp[r].get(s, 0) + est_r > env.rack_budget:
                                ok = False; break
                        if not ok:
                            break
                    if ok:
                        for r, n_on_r in enumerate(counts):
                            if n_on_r == 0:
                                continue
                            est_r = job.peak_per_node_w * int(n_on_r)
                            for s in range(s0, s0 + dur):
                                self.rn[r][s] = self.rn[r].get(s, 0) + int(n_on_r)
                                self.rp[r][s] = self.rp[r].get(s, 0) + est_r
                        self.plan[job.job_id] = (s0, anchor); placed = True; break
                if placed:
                    break
        # execute plans whose start == now
        for job in list(env.pending):
            pl = self.plan.get(job.job_id)
            if pl and pl[0] <= t:
                env.place_job(job, pl[1], power_aware=True)


class RandomPol:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def step(self, env, t):
        for job in list(env.pending):
            feas = [anchor for anchor, _hosts, _counts
                    in shared_candidate_plans(env, job)]
            if feas:
                env.place_job(job, int(self.rng.choice(feas)))


class Spread:
    def step(self, env, t):
        for job in list(env.pending):
            candidates = shared_candidate_plans(env, job)
            if not candidates:
                continue
            # Naive spread rule: choose the least-loaded legal anchor.  Unlike
            # the previous <=3-node gate, it may spill exactly like MARL.
            power = env.rack_power_w()
            anchor, _hosts, _counts = min(candidates, key=lambda x: power[x[0]])
            env.place_job(job, anchor, power_aware=True)


def run_policy(env, pol):
    env.reset()
    meter = ResourceMeter.for_bridge_port(PORT)
    total_r = 0.0
    for t in range(env.max_steps):
        pol.step(env, t)
        m = env.advance()
        meter.sample()
        total_r += m["reward"]
        if env.done:
            break
    s = dict(env.ep_stats); s["sla"] = env.sla_stats()
    return total_r, s, meter.finish()


def run_marl(env, ag, decoder="sequence"):
    env.reset()
    meter = ResourceMeter.for_bridge_port(PORT)
    act = make_deterministic_act(ag, a4_mode=decoder)

    total_r = 0.0
    for t in range(env.max_steps):
        gang_step(env, ag, t, act)
        m = env.advance()
        meter.sample()
        total_r += m["reward"]
        if env.done:
            break
    s = dict(env.ep_stats); s["sla"] = env.sla_stats()
    return total_r, s, meter.finish()


def warmup_environment(env, steps=5):
    """Trigger Java JIT/Py4J/data caches before per-policy resource accounting."""
    env.reset()
    pol = Spread()
    for t in range(min(int(steps), env.max_steps)):
        pol.step(env, t)
        env.advance()
        if env.done:
            break
    env.reset()


def load_agents(env):
    ckpt_dir = os.environ.get("GANG_CHECKPOINT_DIR", ".")
    manifest_path = os.path.join(ckpt_dir, "metric_manifest.json")
    if not os.path.exists(manifest_path):
        raise ValueError("missing metric_manifest.json with v5 relief_credit descriptor")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if (manifest.get("simulator_semantics") != SIMULATOR_SEMANTICS or
            manifest.get("relief_credit") != relief_credit_metadata()):
        raise ValueError("metric manifest simulator semantics or relief_credit descriptor mismatch")
    ag = Agents(env.num_racks, len(build_global(env)))
    missing = []
    for k in ag.actors:
        p = os.path.join(ckpt_dir, f"{k}_{TAG}.pt")
        if os.path.exists(p):
            ag.actors[k].load_state_dict(torch.load(p, map_location=DEVICE))
            ag.actors[k].eval()
        else:
            missing.append(p)
    return ag, missing


def load_sla_multipliers(env):
    path = os.path.join(os.environ.get("GANG_CHECKPOINT_DIR", "."),
                        "sla_multipliers.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        state = json.load(f)
    env.sla_lambda_adm = float(state["admission"])
    env.sla_lambda_restart = float(state.get("restart", env.sla_lambda_restart))
    env.sla_lambda_comp = float(state["completion"])


def main():
    decoder = resolve_decoder()
    env = build_eval_env()
    load_sla_multipliers(env)
    n_jobs = len(env._all_jobs)
    cannot_finish_by_horizon = sum(
        job.arrival_step + job.duration_steps > env.max_steps
        for job in env._all_jobs)
    completion_upper_bound = n_jobs - cannot_finish_by_horizon
    print(f"port={PORT} sub={SUB} steps={env.max_steps} jobs={n_jobs} "
          f"racks={env.num_racks} budget={env.rack_budget:.0f}W safe_res_LA={SAFERES_LA} tag={TAG}\n")
    print(f"Earliest-finish bound: {completion_upper_bound}/{n_jobs} jobs can finish "
          f"within the horizon even with immediate placement; "
          f"{cannot_finish_by_horizon} cannot.\n")
    print(f"MARL deterministic decoder={decoder} (alternate is always audited)\n")
    warmup_environment(env, int(os.environ.get("EVAL_WARMUP_STEPS", 5)))

    rows = []
    for name, pol in [("RANDOM", RandomPol()), ("SPREAD", Spread()),
                      ("EPOBF", EPOBF()), ("RPA", RPA()), ("SAFERES", SafeRes(SAFERES_LA))]:
        rows.append((name,) + run_policy(env, pol))

    ag, missing = load_agents(env)
    if missing:
        print(f"[warn] MARL skipped, missing: {missing}")
    else:
        rows.append(("MARL",) + run_marl(env, ag, decoder))
        alternate = "pointwise" if decoder == "sequence" else "sequence"
        alt_label = "MARL_POINT" if alternate == "pointwise" else "MARL_SEQ"
        rows.append((alt_label,) + run_marl(env, ag, alternate))

    def vpct(rv):
        return 100.0 * rv / max(1.0, 1500.0 * env.num_racks * env.max_steps)

    # served% (goodput) + job-level SLA (dropped OR late), both bounded [0,100].
    hdr = (f"{'policy':10s} {'reward':>7s} {'srv%':>5s} {'SLA%':>6s} {'adm%':>5s} "
           f"{'comp%':>6s} {'drop':>4s} {'late':>4s} {'preempt':>7s} "
           f"{'rackViol':>9s} {'viol%':>6s} {'wait':>5s} "
           f"{'E_kWh':>6s} {'xrack':>6s} {'netGB':>7s} {'netX%':>6s} "
           f"{'netD_s':>7s} {'link%':>6s} {'time_s':>7s} {'cpu%':>6s} {'RAMMB':>7s}")
    print(hdr); print("-" * len(hdr))
    csv_rows = []
    for name, R, s, resource in rows:
        xr = s["cross_rack_sum"] / max(1, s["cross_rack_n"])
        avg_wait = s["place_wait_sum"] / max(1, s["place_n"])
        sla = s["sla"]
        served = 100.0 * sla["served"] / max(1, sla["n"])
        sla_pct = 100.0 * sla["sla_viol"] / max(1, sla["n"])
        adm_pct = 100.0 * sla["admission_late"] / max(1, sla["n"])
        restart_pct = 100.0 * sla.get("restart_late", 0) / max(1, sla["n"])
        completion_viol = sla["completion_late"] + sla["dropped"]
        comp_pct = 100.0 * completion_viol / max(1, sla["n"])
        completion_late_pct = 100.0 * sla["completion_late"] / max(1, sla["n"])
        net_gb = s["network_total_bytes"] / 1e9
        net_cross_pct = 100.0 * s["network_cross_bytes"] / max(
            1.0, s["network_total_bytes"])
        print(f"{name:10s} {R:7.2f} {served:5.0f} {sla_pct:6.0f} "
              f"{adm_pct:5.0f} {comp_pct:6.0f} {sla['dropped']:4d} "
              f"{sla['late']:4d} {s['preempted']:7d} {s['rack_viol']:9d} "
              f"{vpct(s['rack_viol']):6.1f} {avg_wait:5.1f} {s['energy']:6.1f} {xr:6.3f} "
              f"{net_gb:7.2f} {net_cross_pct:6.1f} {s['network_delay_sec']:7.1f} "
              f"{100.0*s['network_max_link_util']:6.2f} "
              f"{resource['wall_time_s']:7.2f} {resource['avg_cpu_pct']:6.1f} "
              f"{resource['peak_rss_mb']:7.1f}")
        csv_rows.append(dict(
            policy=name,
            reward=R,
            jobs_total=sla["n"],
            earliest_finish_impossible_jobs=cannot_finish_by_horizon,
            feasible_completion_upper_bound=completion_upper_bound,
            admitted_jobs=s["admitted"],
            completed_jobs=s["completed"],
            served_jobs=sla["served"],
            served_pct=served,
            sla_viol_jobs=sla["sla_viol"],
            sla_pct=sla_pct,
            admission_late_jobs=sla["admission_late"],
            restart_late_jobs=sla.get("restart_late", 0),
            restart_late_pct=restart_pct,
            admission_pct=adm_pct,
            completion_late_jobs=sla["completion_late"],
            completion_late_pct=completion_late_pct,
            completion_viol_jobs=completion_viol,
            completion_pct=comp_pct,
            dropped=sla["dropped"],
            dropped_jobs=sla["dropped"],
            late=sla["late"],
            late_jobs=sla["late"],
            preempt=s["preempted"],
            preempted_jobs=s["preempted"],
            wait=avg_wait,
            avg_wait_steps=avg_wait,
            wait_cost=s["wait_cost"],
            rack_viol=s["rack_viol"],
            rack_viol_pct=vpct(s["rack_viol"]),
            xrack=xr,
            cross_rack_frac=xr,
            cross_rack_sum=s["cross_rack_sum"],
            cross_rack_samples=s["cross_rack_n"],
            network_cross_bytes=s["network_cross_bytes"],
            network_local_bytes=s["network_local_bytes"],
            network_total_bytes=s["network_total_bytes"],
            network_cross_gb=s["network_cross_bytes"] / 1e9,
            network_local_gb=s["network_local_bytes"] / 1e9,
            network_total_gb=net_gb,
            network_cross_byte_pct=net_cross_pct,
            network_delay_sec=s["network_delay_sec"],
            network_max_job_delay_sec=s["network_max_job_delay_sec"],
            network_max_link_util_pct=100.0 * s["network_max_link_util"],
            network_mean_link_util_pct=(100.0 * s["network_mean_link_util_sum"]
                                        / max(1, s["cross_rack_n"])),
            network_extra_steps=s["network_extra_steps"],
            network_metric_source="simulated_ring_packets_on_fat_tree_links",
            net_comm_tax=NET_COMM_TAX,
            simulated_packet_telemetry=True,
            energy_kwh=s["energy"],
            **resource,
        ))

    csv_path = os.environ.get("GANG_EVAL_CSV", "gang_eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    print(f"\nresource+metric CSV -> {csv_path}")

    d = {n: (R, s) for n, R, s, _resource in rows}
    if "MARL" in d:
        best_heur = max((n for n in d if not n.startswith("MARL")),
                        key=lambda n: d[n][0])
        print(f"\nMARL reward={d['MARL'][0]:+.2f} | best heuristic={best_heur} "
              f"{d[best_heur][0]:+.2f} | SAFERES {d.get('SAFERES',(float('nan'),))[0]:+.2f}")
        print(f"MARL − {best_heur} = {d['MARL'][0] - d[best_heur][0]:+.2f}")


if __name__ == "__main__":
    main()
