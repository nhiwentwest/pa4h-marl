"""
GangEnv — GPU-exclusive rack-power scheduling environment (hướng A).

A self-contained Py4J client over the CloudSim gang bridge (Py4jBridge). Unlike
the legacy CloudSimGymEnv (VM live-migration), the actuator here is gang
submit/preempt with NREL power replay and hard rack power budgets. It exposes the
primitives the 4-agent CTDE policy acts on, plus a greedy power-aware baseline
(HeuristicScheduler) used both as a smoke driver and as a non-RL comparison.

Control flow per RL step t:
  1. reap finished jobs (Java), ingest arrivals(t) into the pending backlog
  2. PLACE  (A4): for each pending job (priority order) pick an anchor rack; env
                  fills its N nodes from free hosts (rack-local first, spill on).
                  Jobs that don't fit stay queued (wait-cost accrues).
  3. DETECT (A2): flag the rack most at risk of a future power-budget violation.
  4. RELIEVE (A1 trigger + A3 victim): if flagged, preempt one running job on that
                  rack (checkpoint cost); the job is requeued (checkpoint-restart).
  5. advance one step (stepGang) -> rack stats, NREL energy, rack-budget reward.

Everything is data-derived: arrivals/durations/power from Alibaba+NREL, budgets
from NREL nameplate. No synthetic spikes, no forecasting model.
"""
import os
import base64

QUEUE_RELIEF_ENABLED = os.environ.get("QUEUE_RELIEF_ENABLED", "0") == "1"
from functools import lru_cache
import numpy as np
from py4j.java_gateway import JavaGateway, GatewayParameters

from nrel_injection_bridge import build_job_queue

RACK_SIZE = int(os.environ.get("NETWORK_RACK_SIZE", 4))
HORIZON = int(os.environ.get("GANG_HORIZON", 6))          # future-trajectory steps (30 min)
RACK_OBS_DIM = 7
JOB_OBS_DIM = 8
NETWORK_OBS_DIM = 4
RACK_HISTORY_LEN = int(os.environ.get("RACK_HISTORY_LEN", 6))
SIMULATOR_SEMANTICS = "gang-replay-v5-global-relief-credit"
A4_DEFER_OPPORTUNITY_COST = float(os.environ.get("A4_DEFER_OPPORTUNITY_COST", 0.12))

NREL_PEAK_W = 3062.0
NREL_IDLE_W = 612.0
INTERVAL_SEC = int(os.environ.get("INTERVAL_SEC", 300))
FINE_DT_SEC = 0.2
NET_COMM_TAX = float(os.environ.get("NET_COMM_TAX", 0.30))
# Placement is legal only if the projected rack peak stays under this fraction of
# the budget — a small safety margin covering the untaxed-projection approximation.
FEAS_MARGIN = float(os.environ.get("FEAS_MARGIN", 0.98))
# SLA: a job whose wait-to-placement (queue + checkpoint-restart delay, measured
# from arrival) exceeds this many RL steps is an SLA (deadline) violation. 12 steps
# = 1h at 300s/step. Priority-weighted so online-serving misses count more.
SLA_MAX_WAIT_STEPS = int(os.environ.get("SLA_MAX_WAIT_STEPS", 12))
SLA_MAX_RESTART_WAIT_STEPS = int(os.environ.get("SLA_MAX_RESTART_WAIT_STEPS", 12))
SLA_COMPLETION_GRACE_STEPS = int(os.environ.get("SLA_COMPLETION_GRACE_STEPS", 12))
SLA_DROP_SEVERITY = float(os.environ.get("SLA_DROP_SEVERITY", 2.0))

# Cooperative TEAM reward weights (gang-native). The objective is to SERVE GPU
# jobs (goodput) at low energy, under hard rack power budgets, with low SLA impact
# (queue wait + checkpoint churn). The +serve term is what makes "place nothing"
# NON-optimal; the −viol/−wait terms make greedy packing and starvation costly.
# All terms are normalized to ~O(1)/step so none dominates by scale.
W_SERVE = float(os.environ.get("W_SERVE", 1.0))
W_ENERGY = float(os.environ.get("W_ENERGY", 0.3))
W_VIOL = float(os.environ.get("W_VIOL", 2.0))
W_WAIT = float(os.environ.get("W_WAIT", 1.0))
W_CKPT = float(os.environ.get("W_CKPT", 0.15))
W_SLA_BASE = float(os.environ.get("W_SLA_BASE", 0.5))
W_DEADLINE = float(os.environ.get("W_DEADLINE", 2.0))
W_NETWORK_DELAY = float(os.environ.get("W_NETWORK_DELAY", 0.10))
W_NETWORK_CONGESTION = float(os.environ.get("W_NETWORK_CONGESTION", 0.10))
W_NETWORK_PLACEMENT = float(os.environ.get("W_NETWORK_PLACEMENT", 0.10))
# Start from a feasible SLA budget, then tighten by an explicit curriculum only
# after a stable policy exists.  Aggressive infeasible targets make dual ascent
# diverge and can turn the relief hierarchy off.
SLA_TARGET_ADMISSION = float(os.environ.get("SLA_TARGET_ADMISSION", 0.20))
SLA_TARGET_RESTART = float(os.environ.get("SLA_TARGET_RESTART", 0.20))
SLA_TARGET_COMPLETION = float(os.environ.get("SLA_TARGET_COMPLETION", 0.20))
SLA_DUAL_LR = float(os.environ.get("SLA_DUAL_LR", 0.10))
SLA_DUAL_WARMUP_UPDATES = int(os.environ.get("SLA_DUAL_WARMUP_UPDATES", 2))
SLA_LAMBDA_MAX = float(os.environ.get("SLA_LAMBDA_MAX", 3.0))


def ring_cross_rack_fraction(hosts, rack_size=RACK_SIZE):
    """Fraction of ring all-reduce edges crossing a rack boundary."""
    hosts = list(hosts)
    if len(hosts) < 2:
        return 0.0
    cross = sum(1 for i, src in enumerate(hosts)
                if src // rack_size != hosts[(i + 1) % len(hosts)] // rack_size)
    return cross / len(hosts)


def network_reward_penalty(delay_sec, max_link_util):
    """Normalized per-step communication delay + congestion penalty."""
    delay_n = max(0.0, float(delay_sec)) / max(1.0, INTERVAL_SEC)
    util_n = np.clip(float(max_link_util), 0.0, 1.0)
    return W_NETWORK_DELAY * delay_n + W_NETWORK_CONGESTION * util_n


class GangEnv:
    def __init__(self, port=None, jobs=None, subsample=None, max_arrival_step=None,
                 min_arrival_step=None, arrival_end_step=None, max_steps=None, seed=0):
        port = int(port if port is not None else os.environ.get("BRIDGE_PORT", 25333))
        self.gw = JavaGateway(gateway_parameters=GatewayParameters(
            port=port, auto_convert=True))
        self.bridge = self.gw.entry_point
        self._IntArr = self.gw.jvm.int
        self._DblArr = self.gw.jvm.double

        self.num_hosts = int(self.bridge.getNumHosts())
        self.num_racks = int(self.bridge.getNumRacks())
        self.rack_budget = float(self.bridge.getRackBudgetW())
        self.per_node_budget = self.rack_budget / RACK_SIZE
        self._fine_per_step = max(1, int(round(INTERVAL_SEC / FINE_DT_SEC)))  # 1500
        self._feas_budget = self.rack_budget * FEAS_MARGIN
        self._base_cache = {}       # rack -> phase-aligned fine-power sum (per RL step)
        self._win_cache = {}        # (id(job), elapsed) -> fine-power window (per episode)
        self._free_cache = None     # getHostFreeMap() cached per step (Py4J round-trip)
        self._rackpy_cache = None   # (power,peak,viol,traj) cached per step (Py4J)

        self._all_jobs = jobs if jobs is not None else build_job_queue(
            subsample=subsample, max_arrival_step=max_arrival_step,
            min_arrival_step=min_arrival_step, arrival_end_step=arrival_end_step,
            verbose=True)
        self.max_steps = int(max_steps if max_steps is not None else
                             (max(j.arrival_step for j in self._all_jobs) + 60))

        self._trace_cache = {}      # profile_path -> java double[]
        self.sla_lambda_adm = float(os.environ.get("SLA_LAMBDA_ADMISSION", 1.0))
        self.sla_lambda_restart = float(os.environ.get("SLA_LAMBDA_RESTART", 1.0))
        self.sla_lambda_comp = float(os.environ.get("SLA_LAMBDA_COMPLETION", 1.0))
        self._sla_rate_ema = np.zeros(3, dtype=np.float64)
        self._sla_dual_updates = 0
        self._sla_ema_initialized = False
        self.reset()

    # ---------------- lifecycle ----------------
    def _packed_trace_for(self, job):
        key = job.profile_path
        a = self._trace_cache.get(key)
        if a is None:
            arr = np.ascontiguousarray(job.per_node_trace_w, dtype="<f8")
            a = base64.b64encode(arr.tobytes()).decode("ascii")
            self._trace_cache[key] = a
        return a

    def _jarr_int(self, lst):
        a = self.gw.new_array(self._IntArr, len(lst))
        for i, v in enumerate(lst):
            a[i] = int(v)
        return a

    def reset(self):
        self.bridge.reset()
        self.step_idx = 0
        # arrivals bucketed by step
        self._arrivals = {}
        for j in self._all_jobs:
            j.progress_steps = 0
            j.remaining_duration_steps = j.duration_steps
            j.placed_hosts = None
            j.arrival_step_placed = -1
            self._arrivals.setdefault(j.arrival_step, []).append(j)
        self.pending = []           # list[Job] backlog (priority-sorted each step)
        self.running = {}           # job_id -> Job
        if QUEUE_RELIEF_ENABLED:
            from queue_relief import reset
            reset(self)
        self.wait_steps = {}        # job_id -> steps spent waiting (SLA proxy)
        self._prev_viol = np.zeros(self.num_racks)
        self._prev_energy = 0.0
        self._last_network = np.zeros(12, dtype=np.float64)
        self._last_xrack = 0.0
        self._rack_history = []
        self._rack_history_step = None
        self._ckpt_nodes_step = 0   # nodes checkpoint-preempted in the current step
        self._base_cache = {}
        self._win_cache = {}
        self._free_cache = None
        self._rackpy_cache = None
        self._completed_ids = set()   # unique jobs that finished (goodput)
        self._first_wait = {}         # job_id -> wait (steps) at its FIRST placement
        self._admission_breached = set()
        self._restart_breached = set()
        self._restart_wait_total = {}
        self._restart_wait_since = {}
        self._completion_late_ids = set()
        self._terminal_sla_applied = False
        self._deadline_crossed_ids = set()   # jobs that newly crossed SLA deadline
        self._sla_adm_cost_step = 0.0
        self._sla_restart_cost_step = 0.0
        self._sla_comp_cost_step = 0.0
        self._sla_priority_total = max(1.0, sum(float(j.priority) for j in self._all_jobs))
        self.done = False
        self.ep_stats = dict(admitted=0, preempted=0, completed=0, wait_cost=0.0,
                             rack_viol=0, energy=0.0, cross_rack_sum=0.0, cross_rack_n=0,
                             # SLA: placement latency + priority-weighted deadline misses
                             place_wait_sum=0.0, place_n=0, sla_late=0, sla_late_w=0.0,
                             admission_breach=0, admission_breach_w=0.0,
                             restart_breach=0, restart_breach_w=0.0,
                             completion_late=0, completion_late_w=0.0,
                             drop_w=0.0, sla_adm_cost=0.0,
                             sla_restart_cost=0.0, sla_comp_cost=0.0,
                             a2_ranked=0, a3_candidate=0, a1_decisions=0,
                             a1_preempt=0, a3_executed=0,
                             relief_opportunities=0, preempt_on_risk=0,
                             risk_steps=0, a2_choices=0, a2_forced=0,
                             a2_correct=0, a2_choice_correct=0,
                             a3_choices=0, a3_forced=0,
                             a3_correct=0, a3_choice_correct=0,
                             a1_oracle_preempt=0,
                             a1_oracle_wait=0, a1_correct=0,
                             network_cross_bytes=0.0, network_local_bytes=0.0,
                             network_total_bytes=0.0, network_delay_sec=0.0,
                             network_max_job_delay_sec=0.0,
                             network_max_link_util=0.0,
                             network_mean_link_util_sum=0.0,
                             network_extra_steps=0.0)
        self._ingest()
        return self.observe()

    def _ingest(self):
        # reap finished jobs first (Java frees their hosts)
        before = set(self.running)
        reaped = int(self.bridge.reapFinishedJobs())
        if reaped:
            free = self.free_host_map()
            for jid in list(self.running):
                hosts = self.running[jid].placed_hosts
                if hosts and all(free[h] for h in hosts):
                    self.running[jid].remaining_duration_steps = 0
                    self.running.pop(jid, None)
                    self._completed_ids.add(jid)
                    self.ep_stats["completed"] += 1
                    job = next((x for x in self._all_jobs if x.job_id == jid), None)
                    if job is not None:
                        deadline = (job.arrival_step + job.duration_steps
                                    + SLA_COMPLETION_GRACE_STEPS)
                        if self.step_idx > deadline and jid not in self._completion_late_ids:
                            self._completion_late_ids.add(jid)
                            self._sla_comp_cost_step += float(job.priority)
                            self.ep_stats["completion_late"] += 1
                            self.ep_stats["completion_late_w"] += float(job.priority)
        for job in self.running.values():
            job.remaining_duration_steps = int(self.bridge.getJobProgress(job.job_id)[1])
        # ingest this step's arrivals
        for j in self._arrivals.get(self.step_idx, []):
            j.placed_hosts = None
            self.pending.append(j)
            self.wait_steps.setdefault(j.job_id, 0)

    # ---------------- world queries ----------------
    def free_host_map(self):
        if self._free_cache is None:
            self._free_cache = list(self.bridge.getHostFreeMap())
        return self._free_cache

    def _rack_py_arrays(self):
        """Py4J rack arrays (power, peak, viol, future-traj) — stable within an RL
        step (they update only on stepGang), fetched once and cached. This is what
        makes the per-job A4 mask loop cheap instead of hammering the bridge."""
        if self._rackpy_cache is None:
            power = np.array(self.bridge.getRackPowerW(), dtype=np.float64)
            peak = np.array(self.bridge.getRackPeakW(), dtype=np.float64)
            viol = np.array(self.bridge.getRackViolCountStep(), dtype=np.float64)
            traj = self._future_traj()
            self._rackpy_cache = (power, peak, viol, traj)
        return self._rackpy_cache

    def rack_of(self, host):
        return host // RACK_SIZE

    def rack_power_w(self):
        """Current per-rack average power (W) this step — cheap, cached read."""
        return self._rack_py_arrays()[0]

    def rack_free_nodes(self, r):
        free = self.free_host_map()
        lo, hi = r * RACK_SIZE, min(self.num_hosts, (r + 1) * RACK_SIZE)
        return sum(1 for h in range(lo, hi) if free[h])

    # ---------------- foresight power projection ----------------
    def _fine_window(self, job, elapsed_steps):
        """This job's per-node fine-power (W) over the NEXT RL step, phase-aligned to
        how long it has already run (traces loop with their own period). Memoized."""
        elapsed_steps = job.progress_steps + int(elapsed_steps)
        key = (id(job), elapsed_steps)
        w = self._win_cache.get(key)
        if w is not None:
            return w
        tr = job.per_node_trace_w
        L = len(tr)
        if L == 0:
            w = np.full(self._fine_per_step, NREL_IDLE_W, dtype=np.float64)
        else:
            base = int((elapsed_steps * self._fine_per_step) % L)
            idx = (base + np.arange(self._fine_per_step)) % L
            w = np.asarray(tr, dtype=np.float64)[idx]
        self._win_cache[key] = w
        return w

    def _rack_base_fine(self, r):
        """Phase-aligned fine-power sum over all nodes currently on rack r, cached
        within the RL step (invalidated on place / preempt / advance)."""
        c = self._base_cache.get(r)
        if c is None:
            lo, hi = r * RACK_SIZE, min(self.num_hosts, (r + 1) * RACK_SIZE)
            total = np.full(self._fine_per_step, (hi - lo) * NREL_IDLE_W,
                            dtype=np.float64)
            for j in self.jobs_on_rack(r):
                n_on_r = sum(1 for h in j.placed_hosts if self.rack_of(h) == r)
                if n_on_r:
                    mult = 1.0 + NET_COMM_TAX * ring_cross_rack_fraction(j.placed_hosts)
                    active = self._fine_window(j, self.step_idx - j.arrival_step_placed) * mult
                    total = total + n_on_r * (active - NREL_IDLE_W)
            self._base_cache[r] = total
            c = total
        return c

    def project_rack_peak(self, r, add_job=None, add_n=0, add_hosts=None):
        """Projected rack peak power (W) over the next step, phase-aware from the REAL
        NREL traces, if add_n nodes of add_job were placed on rack r now. Anti-phase
        jobs don't stack peaks (a 4th safe node is allowed); in-phase jobs do (refused).
        This is the temporal foresight the ≤3-node rule is blind to."""
        total = self._rack_base_fine(r)
        if add_job is not None and add_n > 0:
            if add_hosts is None:
                raise ValueError("Power projection requires the candidate host plan")
            mult = 1.0 + NET_COMM_TAX * ring_cross_rack_fraction(add_hosts)
            total = total + add_n * (self._fine_window(add_job, 0) * mult - NREL_IDLE_W)
        return float(total.max()) if total.size else 0.0

    def project_rack_violation_fraction(self, r, remove_job=None):
        """Exact next-step fine-sample violation rate, optionally without one job.

        This is the counterfactual A1/A3 need: a victim is useful only when its
        removal actually reduces rack-budget violations, not merely because it
        has a low restart cost.
        """
        total = self._rack_base_fine(r).copy()
        if remove_job is not None and remove_job.placed_hosts:
            n_on_r = sum(1 for h in remove_job.placed_hosts if self.rack_of(h) == r)
            if n_on_r:
                elapsed = max(0, self.step_idx - remove_job.arrival_step_placed)
                mult = 1.0 + NET_COMM_TAX * ring_cross_rack_fraction(remove_job.placed_hosts)
                active_delta = self._fine_window(remove_job, elapsed) * mult - NREL_IDLE_W
                total = np.maximum(0.0, total - n_on_r * active_delta)
        return float(np.mean(total > self.rack_budget)) if total.size else 0.0

    def project_job_energy_saved_normalized(self, job):
        """Next-step energy removed by preempting ``job``, normalized exactly like
        the team reward's energy term. Idle host power remains after removal."""
        if not job.placed_hosts:
            return 0.0
        removed_mean_w = 0.0
        elapsed = max(0, self.step_idx - job.arrival_step_placed)
        for r in {self.rack_of(h) for h in job.placed_hosts}:
            n_on_r = sum(1 for h in job.placed_hosts if self.rack_of(h) == r)
            mult = 1.0 + NET_COMM_TAX * ring_cross_rack_fraction(job.placed_hosts)
            delta = self._fine_window(job, elapsed) * mult - NREL_IDLE_W
            removed_mean_w += n_on_r * float(np.mean(np.maximum(0.0, delta)))
        return removed_mean_w / max(1e-6, self.num_hosts * NREL_PEAK_W)

    def _future_traj(self):
        traj = self.bridge.getRackFutureTrajectory(HORIZON)
        return np.array([[traj[r][t] for t in range(HORIZON)]
                         for r in range(self.num_racks)], dtype=np.float64)

    # ---------------- observations ----------------
    def rack_features(self):
        """[num_racks, RACK_OBS_DIM] normalized rack state."""
        free = self.free_host_map()
        bridge_power, bridge_peak, viol, traj = self._rack_py_arrays()
        # Java's telemetry clock advances only in stepGang().  A4 makes several
        # placements before that call, so use the Python phase-aligned replay for
        # power/peak in the observation.  This makes the second placement see the
        # first one immediately; the bridge trajectory remains a conservative
        # longer-horizon signal.
        live = np.array([self._rack_base_fine(r) for r in range(self.num_racks)])
        power = live.mean(axis=1) if live.size else bridge_power
        peak = live.max(axis=1) if live.size else bridge_peak
        bridge_fut = traj.max(axis=1) if traj.size else bridge_peak
        fut_peak = np.maximum(bridge_fut, peak)
        feat = np.zeros((self.num_racks, RACK_OBS_DIM), dtype=np.float32)
        for r in range(self.num_racks):
            lo, hi = r * RACK_SIZE, min(self.num_hosts, (r + 1) * RACK_SIZE)
            free_r = sum(1 for h in range(lo, hi) if free[h])
            n_jobs = sum(1 for j in self.running.values() if j.placed_hosts
                         and any(self.rack_of(h) == r for h in j.placed_hosts))
            feat[r] = [
                free_r / RACK_SIZE,
                power[r] / self.rack_budget,
                peak[r] / self.rack_budget,
                fut_peak[r] / self.rack_budget,
                (self.rack_budget - power[r]) / self.rack_budget,
                viol[r] / 1500.0,
                min(1.0, n_jobs / RACK_SIZE),
            ]
        return feat

    def network_features(self):
        """Last-step packet telemetry visible to A4 at decentralized execution."""
        n = self._last_network
        return np.array([
            n[3],
            n[5],
            np.clip(n[6] / max(1.0, INTERVAL_SEC), 0.0, 1.0),
            self._last_xrack,
        ], dtype=np.float32)

    def rack_history_features(self, current=None):
        """Return [rack, time, feature] history, updating the current step once."""
        current = self.rack_features() if current is None else np.asarray(current)
        current = current.astype(np.float32, copy=True)
        if self._rack_history_step == self.step_idx and self._rack_history:
            self._rack_history[-1] = current
        else:
            self._rack_history.append(current)
            self._rack_history = self._rack_history[-RACK_HISTORY_LEN:]
            self._rack_history_step = self.step_idx
        frames = list(self._rack_history)
        while len(frames) < RACK_HISTORY_LEN:
            frames.insert(0, frames[0].copy())
        return np.stack(frames[-RACK_HISTORY_LEN:], axis=1)

    def a4_counterfactuals(self, job):
        """Exact, one-step counterfactual utility for every A4 action.

        Each legal anchor is replayed against the *current* Python phase-aligned
        rack state.  Utility is negative projected peak pressure, so a higher
        value is safer.  DEFER is deliberately worse than the best feasible
        placement by a fixed opportunity cost; it remains legal, but is no
        longer treated as equally safe merely because it adds no power.
        """
        utility = np.full(self.num_racks + 1, -np.inf, dtype=np.float32)
        mask = np.zeros(self.num_racks + 1, dtype=bool)
        projected = np.full((self.num_racks, self.num_racks), np.nan, dtype=np.float32)
        for anchor in range(self.num_racks):
            hosts = self._pick_hosts_pa(anchor, job)
            if hosts is None:
                continue
            added = np.zeros(self.num_racks, dtype=np.int32)
            for h in hosts:
                added[self.rack_of(h)] += 1
            peaks = np.array([
                self.project_rack_peak(r, job, int(added[r]), add_hosts=hosts) if added[r]
                else self.project_rack_peak(r)
                for r in range(self.num_racks)
            ], dtype=np.float32)
            projected[anchor] = peaks
            # The maximum governs a rack/PDU cap; mean pressure breaks ties in
            # favour of a placement that preserves cluster-wide headroom.
            ratios = peaks / max(1e-6, self.rack_budget)
            cross_edges = ring_cross_rack_fraction(hosts)
            utility[anchor] = -float(ratios.max() + 0.10 * ratios.mean()
                                     + W_NETWORK_PLACEMENT * cross_edges)
            mask[anchor] = True
        mask[self.num_racks] = True
        feasible = utility[:self.num_racks][mask[:self.num_racks]]
        utility[self.num_racks] = (float(feasible.max()) - A4_DEFER_OPPORTUNITY_COST
                                   if feasible.size else 0.0)
        utility[self.num_racks] -= self.admission_defer_penalty(job)
        return utility, mask, projected

    def a4_counterfactual_details(self, job):
        """Full-information A4 labels consumed by the trainer and eval loop."""
        utility, mask, projected = self.a4_counterfactuals(job)
        if QUEUE_RELIEF_ENABLED:
            from queue_relief import held
            if held(self, job):
                mask[:self.num_racks] = False
        legal_racks = mask[:self.num_racks]
        horizon_risk = np.ones(self.num_racks, dtype=np.float32)
        if legal_racks.any():
            horizon_risk[legal_racks] = np.mean(
                projected[legal_racks] > self.rack_budget, axis=1).astype(np.float32)
        forced_defer = not bool(legal_racks.any())
        if forced_defer:
            self.ep_stats["forced_defer_states"] = self.ep_stats.get("forced_defer_states", 0) + 1
        return {"utility": utility, "mask": mask, "projected": projected,
                "forced_defer": forced_defer, "horizon_risk": horizon_risk}

    def job_features(self, job):
        wait = max(0, self.step_idx - job.arrival_step)
        adm_slack = SLA_MAX_WAIT_STEPS - wait
        expected_finish = self.step_idx + self.running_remaining(job)
        comp_deadline = job.arrival_step + job.duration_steps + SLA_COMPLETION_GRACE_STEPS
        comp_scale = max(1.0, job.duration_steps + SLA_COMPLETION_GRACE_STEPS)
        return np.array([
            job.num_nodes / RACK_SIZE,
            job.peak_per_node_w / self.per_node_budget,
            job.priority / 3.0,
            min(1.0, self.running_remaining(job) / 50.0),
            job.mean_per_node_w / self.per_node_budget,
            min(2.0, wait / max(1.0, SLA_MAX_WAIT_STEPS)),
            np.clip(adm_slack / max(1.0, SLA_MAX_WAIT_STEPS), -1.0, 1.0),
            np.clip((comp_deadline - expected_finish) / comp_scale, -1.0, 1.0),
        ], dtype=np.float32)

    def admission_defer_cost(self, job):
        """Potential increase from waiting one more step; steep near the deadline."""
        wait = max(0, self.step_idx - job.arrival_step)
        d = max(1.0, float(SLA_MAX_WAIT_STEPS))
        before = min(1.0, wait / d) ** 2
        after = min(1.0, (wait + 1) / d) ** 2
        crossing = 1.0 if wait <= SLA_MAX_WAIT_STEPS < wait + 1 else 0.0
        return (float(job.priority) / 3.0) * (after - before + crossing)

    def admission_defer_penalty(self, job):
        """SLA-aware A4 shaping that remains active when the dual term is zero."""
        return (W_SLA_BASE + self.sla_lambda_adm) * self.admission_defer_cost(job)

    def completion_slack(self, job):
        deadline = job.arrival_step + job.duration_steps + SLA_COMPLETION_GRACE_STEPS
        return deadline - (self.step_idx + self.running_remaining(job))

    def preemption_sla_cost(self, job):
        """Priority-weighted risk that checkpoint-restart causes completion SLA harm."""
        deadline = job.arrival_step + job.duration_steps + SLA_COMPLETION_GRACE_STEPS
        projected_after_restart = self.step_idx + self.running_remaining(job) + 1
        slack = deadline - projected_after_restart
        scale = max(1.0, job.duration_steps + SLA_COMPLETION_GRACE_STEPS)
        risk = (1.0 - np.clip(slack / scale, 0.0, 1.0)) ** 2
        return (float(job.priority) / 3.0) * (0.10 + float(risk))

    def _record_admission_breach(self, job):
        if job.job_id in self._admission_breached:
            return False
        first_wait = self._first_wait.get(job.job_id)
        if first_wait is not None and first_wait <= SLA_MAX_WAIT_STEPS:
            return False
        if self.step_idx - job.arrival_step <= SLA_MAX_WAIT_STEPS:
            return False
        self._admission_breached.add(job.job_id)
        cost = float(job.priority)
        self._sla_adm_cost_step += cost
        self.ep_stats["admission_breach"] += 1
        self.ep_stats["admission_breach_w"] += cost
        return True

    def _record_restart_breach(self, job):
        jid = job.job_id
        since = self._restart_wait_since.get(jid)
        if since is None or jid in self._restart_breached:
            return False
        cumulative = self._restart_wait_total.get(jid, 0) + self.step_idx - since
        if cumulative <= SLA_MAX_RESTART_WAIT_STEPS:
            return False
        self._restart_breached.add(jid)
        cost = float(job.priority)
        self._sla_restart_cost_step += cost
        self.ep_stats["restart_breach"] += 1
        self.ep_stats["restart_breach_w"] += cost
        return True

    def sla_global_features(self):
        pending_risk = ([min(1.0, max(0, self.step_idx - j.arrival_step)
                                    / max(1.0, SLA_MAX_WAIT_STEPS)) for j in self.pending]
                        or [0.0])
        running_risk = ([min(1.0, max(0.0, -self.completion_slack(j))
                                    / max(1.0, j.duration_steps)) for j in self.running.values()]
                        or [0.0])
        return np.array([
            float(np.mean(pending_risk)),
            len(self._admission_breached) / max(1, len(self._all_jobs)),
            float(np.mean(running_risk)),
            len(self._completion_late_ids) / max(1, len(self._all_jobs)),
        ], dtype=np.float32)

    def update_sla_multipliers(self, admission_rate, restart_rate, completion_rate):
        rates = np.clip([admission_rate, restart_rate, completion_rate], 0.0, 1.0)
        self._sla_dual_updates = getattr(self, "_sla_dual_updates", 0) + 1
        # Initialize EMA with first real rate instead of from zero
        if not hasattr(self, "_sla_ema_initialized") or not self._sla_ema_initialized:
            self._sla_rate_ema = rates.copy()
            self._sla_ema_initialized = True
        else:
            self._sla_rate_ema = 0.7 * self._sla_rate_ema + 0.3 * rates
        if self._sla_dual_updates <= SLA_DUAL_WARMUP_UPDATES:
            return
        for i, (name, target, attr) in enumerate((
            ("adm", SLA_TARGET_ADMISSION, "sla_lambda_adm"),
            ("restart", SLA_TARGET_RESTART, "sla_lambda_restart"),
            ("comp", SLA_TARGET_COMPLETION, "sla_lambda_comp"),
        )):
            before = getattr(self, attr)
            delta = SLA_DUAL_LR * (self._sla_rate_ema[i] - target)
            setattr(self, attr, float(np.clip(before + delta, 0.0, SLA_LAMBDA_MAX)))
            print(f"     sla_dual_{name}: raw={rates[i]:.3f} ema={self._sla_rate_ema[i]:.3f} "
                  f"target={target:.2f} lambda {before:.3f}+{delta:+.4f}={getattr(self, attr):.3f}")

    def running_remaining(self, job):
        return max(0, job.remaining_duration_steps)

    def observe(self):
        self.pending.sort(key=lambda j: (-j.priority, j.arrival_step, j.job_id))
        return dict(
            racks=self.rack_features(),
            pending=self.pending,
            running=list(self.running.values()),
            free=self.free_host_map(),
        )

    # ---------------- actuation ----------------
    def _pick_hosts(self, anchor_rack, k):
        """k free hosts, anchor rack first then spill to nearest racks. None if <k free."""
        free = self.free_host_map()
        order = list(range(anchor_rack * RACK_SIZE, self.num_hosts)) + \
            list(range(0, anchor_rack * RACK_SIZE))
        chosen = [h for h in order if free[h]][:k]
        return chosen if len(chosen) == k else None

    def _pick_hosts_pa(self, anchor_rack, job):
        """Find a safe rack-contiguous ring plan, preferring fewer occupied racks.

        For m occupied racks and k nodes, the ring has m/k crossing edges (zero
        for m=1). Search rack counts under that exact multiplier. Anchor-first
        choices preserve locality without getting stuck in an unsafe 3+1 split
        when a safe 2+2 split exists. Each fixed-m search has at most racks × k × k states.
        """
        free = self.free_host_map()
        others = sorted((r for r in range(self.num_racks) if r != anchor_rack),
                        key=lambda r: self.project_rack_peak(r))
        order = [anchor_rack] + others
        hosts = {r: [h for h in range(r * RACK_SIZE, min(self.num_hosts, (r + 1) * RACK_SIZE))
                     if free[h]] for r in order}
        if sum(map(len, hosts.values())) < job.num_nodes:
            return None
        power = self._fine_window(job, 0)
        for rack_count in range(1, min(job.num_nodes, self.num_racks) + 1):
            cross = 0.0 if rack_count == 1 else rack_count / job.num_nodes
            added_power = power * (1.0 + NET_COMM_TAX * cross) - NREL_IDLE_W
            allowed = {
                r: tuple(n for n in range(min(len(hosts[r]), job.num_nodes), 0, -1)
                         if float((self._rack_base_fine(r) + n * added_power).max()) <= self._feas_budget)
                for r in order
            }

            @lru_cache(maxsize=None)
            def select(index, slots, nodes):
                if slots == 0:
                    return () if nodes == 0 else None
                if index == len(order) or nodes < slots:
                    return None
                capacities = sorted((max(allowed[r]) for r in order[index:] if allowed[r]), reverse=True)
                if len(capacities) < slots or sum(capacities[:slots]) < nodes:
                    return None
                rack = order[index]
                for n in allowed[rack]:
                    if n > nodes - slots + 1:
                        continue
                    suffix = select(index + 1, slots - 1, nodes - n)
                    if suffix is not None:
                        return ((rack, n),) + suffix
                return select(index + 1, slots, nodes)

            counts = select(0, rack_count, job.num_nodes)
            if counts is not None:
                return [host for rack, n in counts for host in hosts[rack][:n]]
        return None

    def place_job(self, job, anchor_rack, power_aware=True):
        if QUEUE_RELIEF_ENABLED:
            from queue_relief import held
            if held(self, job):
                return False
        # power_aware=True (default, used by RL + safe baselines): reject placements
        # that would push the rack's projected peak over budget. False: legacy
        # host-availability packing (the GREEDY over-pack baseline, for contrast).
        hosts = (self._pick_hosts_pa(anchor_rack, job) if power_aware
                 else self._pick_hosts(anchor_rack, job.num_nodes))
        if hosts is None:
            return False
        trace = self._packed_trace_for(job)
        ok = self.bridge.submitJobPacked(job.job_id, self._jarr_int(hosts), trace,
                                         float(job.dt_sec), int(job.duration_steps),
                                         float(job.priority))
        if ok:
            progress = self.bridge.getJobProgress(job.job_id)
            job.progress_steps, job.remaining_duration_steps = int(progress[0]), int(progress[1])
            job.placed_hosts = hosts
            job.arrival_step_placed = self.step_idx
            self.running[job.job_id] = job
            self._base_cache.clear()   # rack occupancy changed
            self._free_cache = None
            self._rackpy_cache = None  # next A4 decision must see this placement
            if job in self.pending:
                self.pending.remove(job)
            self.ep_stats["admitted"] += 1
            # SLA latency: steps waited from arrival to THIS placement (queue +
            # any checkpoint-restart delay from prior preemptions).
            wait = max(0, self.step_idx - job.arrival_step)
            self.ep_stats["place_wait_sum"] += wait
            self.ep_stats["place_n"] += 1
            self._first_wait.setdefault(job.job_id, wait)   # per-job, first placement
            since = self._restart_wait_since.get(job.job_id)
            if since is not None:
                self._record_restart_breach(job)
                self._restart_wait_total[job.job_id] = (
                    self._restart_wait_total.get(job.job_id, 0) + self.step_idx - since)
                del self._restart_wait_since[job.job_id]
            if wait > SLA_MAX_WAIT_STEPS:
                self.ep_stats["sla_late"] += 1
                self.ep_stats["sla_late_w"] += job.priority
                self._record_admission_breach(job)
        return bool(ok)

    def preempt_job(self, job):
        progress = self.bridge.getJobProgress(job.job_id)
        job.progress_steps, job.remaining_duration_steps = int(progress[0]), int(progress[1])
        self.bridge.preemptJob(job.job_id)
        self.running.pop(job.job_id, None)
        self._base_cache.clear()   # rack occupancy changed
        self._free_cache = None
        self._rackpy_cache = None  # A4/A2 cannot reuse pre-preemption rack state
        self._ckpt_nodes_step += job.num_nodes   # checkpoint cost ∝ nodes
        job.placed_hosts = None
        self._restart_wait_since[job.job_id] = self.step_idx
        self.pending.append(job)                 # checkpoint-restart: requeue
        self.ep_stats["preempted"] += 1

    def jobs_on_rack(self, r):
        return [j for j in self.running.values()
                if j.placed_hosts
                and any(self.rack_of(h) == r for h in j.placed_hosts)]

    def sla_stats(self):
        """Per-job, bounded SLA accounting (call after an episode). A job breaches SLA
        if it never completed (dropped) OR completed but was first placed past the
        deadline. served = goodput. All counts are over the unique job set."""
        n = len(self._all_jobs)
        served = len(self._completed_ids)
        dropped_ids = {j.job_id for j in self._all_jobs} - self._completed_ids
        late_ids = ((self._admission_breached | self._restart_breached |
                     self._completion_late_ids)
                    & self._completed_ids)
        late = len(late_ids)
        dropped = n - served
        return dict(n=n, served=served, dropped=dropped, late=late,
                    admission_late=len(self._admission_breached),
                    restart_late=len(self._restart_breached),
                    completion_late=len(self._completion_late_ids),
                    sla_viol=len(dropped_ids | late_ids))

    def advance(self):
        """Advance one step (pure trace replay) and compute the gang-native TEAM
        reward in Python (the Java scalar is ignored). Returns metrics + components."""
        res = np.array(self.bridge.stepGang())
        gdim = int(self.bridge.getGlobalStateDim())
        done_flag = float(res[gdim + 1]) > 0.5
        # served nodes = GPU nodes actively computing this step (goodput proxy)
        served = sum(j.num_nodes for j in self.running.values())
        energy_now = float(self.bridge.getEnergyKwh())
        e_delta = max(0.0, energy_now - self._prev_energy)
        self._prev_energy = energy_now
        viol = np.array(self.bridge.getRackViolCountStep(), dtype=np.float64)
        cross_rack = float(self.bridge.getGangCrossRackFraction())
        net = np.array(self.bridge.getGangNetworkMetrics(), dtype=np.float64)
        if net.size < 12:
            raise RuntimeError("gang bridge returned incomplete network telemetry")
        self._last_network = net
        self._last_xrack = cross_rack
        ckpt_nodes = self._ckpt_nodes_step
        self._ckpt_nodes_step = 0

        self.step_idx += 1
        self._base_cache.clear()   # phases advanced -> projections stale
        self._free_cache = None
        self._rackpy_cache = None
        self._ingest()
        wcost = 0.0
        for j in self.pending:
            self.wait_steps[j.job_id] = self.wait_steps.get(j.job_id, 0) + 1
            wcost += j.priority
            self._record_admission_breach(j)
            self._record_restart_breach(j)

        # --- normalized reward components (each ~O(1)/step) ---
        e_full = self.num_hosts * NREL_PEAK_W * INTERVAL_SEC / 3.6e6
        serve_n = served / max(1, self.num_hosts)
        energy_n = e_delta / max(1e-6, e_full)
        viol_n = viol.sum() / max(1.0, 1500.0 * self.num_racks)
        wait_n = wcost / max(1.0, self.num_hosts * 3.0)
        ckpt_n = ckpt_nodes / max(1, self.num_hosts)
        network_n = network_reward_penalty(net[6], net[5])
        self.done = done_flag or self.step_idx >= self.max_steps
        if self.done and not self._terminal_sla_applied:
            dropped = [j for j in self._all_jobs if j.job_id not in self._completed_ids]
            drop_cost = SLA_DROP_SEVERITY * sum(float(j.priority) for j in dropped)
            self._sla_comp_cost_step += drop_cost
            self.ep_stats["drop_w"] += drop_cost
            self._terminal_sla_applied = True
        # Deadline-crossing penalty: one-shot hit when a job first passes SLA deadline
        newly_late = 0
        for j in self.pending:
            if j.job_id not in self._deadline_crossed_ids:
                if self.step_idx - j.arrival_step > SLA_MAX_WAIT_STEPS:
                    self._deadline_crossed_ids.add(j.job_id)
                    newly_late += 1
        deadline_n = newly_late / max(1, len(self._all_jobs))
        sla_adm_n = self._sla_adm_cost_step / self._sla_priority_total
        sla_restart_n = self._sla_restart_cost_step / self._sla_priority_total
        sla_comp_n = self._sla_comp_cost_step / self._sla_priority_total
        reward = (W_SERVE * serve_n - W_ENERGY * energy_n - W_VIOL * viol_n
                  - W_WAIT * wait_n - W_CKPT * ckpt_n
                  - network_n
                  - W_DEADLINE * deadline_n
                  - (W_SLA_BASE + self.sla_lambda_adm) * sla_adm_n
                  - (W_SLA_BASE + self.sla_lambda_restart) * sla_restart_n
                  - (W_SLA_BASE + self.sla_lambda_comp) * sla_comp_n)
        self.ep_stats["sla_adm_cost"] += self._sla_adm_cost_step
        self.ep_stats["sla_restart_cost"] += self._sla_restart_cost_step
        self.ep_stats["sla_comp_cost"] += self._sla_comp_cost_step
        self._sla_adm_cost_step = 0.0
        self._sla_restart_cost_step = 0.0
        self._sla_comp_cost_step = 0.0

        self.ep_stats["wait_cost"] += wcost
        self.ep_stats["rack_viol"] += int(viol.sum())
        self.ep_stats["energy"] = energy_now
        self.ep_stats["cross_rack_sum"] += cross_rack
        self.ep_stats["cross_rack_n"] += 1
        self.ep_stats["network_cross_bytes"] += net[0]
        self.ep_stats["network_local_bytes"] += net[1]
        self.ep_stats["network_total_bytes"] += net[2]
        self.ep_stats["network_delay_sec"] += net[6]
        self.ep_stats["network_max_job_delay_sec"] = max(
            self.ep_stats["network_max_job_delay_sec"], net[7])
        self.ep_stats["network_max_link_util"] = max(
            self.ep_stats["network_max_link_util"], net[5])
        self.ep_stats["network_mean_link_util_sum"] += net[4]
        self.ep_stats["network_extra_steps"] += net[9]
        # Fixed-horizon episodes for stable RL: end at max_steps (or the Java
        # GANG_MAX_STEPS cap). Unfinished jobs simply don't complete this episode.
        metrics = dict(
            reward=float(reward), served=int(served),
            energy_kwh=energy_now, rack_viol=int(viol.sum()), rack_viol_vec=viol,
            wait_cost=wcost, ckpt_nodes=int(ckpt_nodes), cross_rack=cross_rack,
            network_cross_bytes=net[0], network_local_bytes=net[1],
            network_total_bytes=net[2], network_cross_frac=net[3],
            network_mean_link_util=net[4], network_max_link_util=net[5],
            network_delay_sec=net[6], network_max_job_delay_sec=net[7],
            network_extra_steps=net[9],
            n_pending=len(self.pending), n_running=len(self.running),
            comp=dict(serve=serve_n, energy=energy_n, viol=viol_n,
                      wait=wait_n, ckpt=ckpt_n, sla_admission=sla_adm_n,
                      sla_restart=sla_restart_n,
                      sla_completion=sla_comp_n, network=network_n),
        )
        self._prev_viol = viol
        return metrics


# --------------------------------------------------------------------------- #
# Greedy power-aware baseline (smoke driver + non-RL comparison)
# --------------------------------------------------------------------------- #
class HeuristicScheduler:
    """Place each pending job on the rack whose predicted post-placement peak is
    lowest (best-fit under budget); when a rack's future peak exceeds budget,
    preempt its lowest-priority job. Pure rules, no learning."""

    def __init__(self, env):
        self.env = env

    def _best_rack(self, job):
        e = self.env
        feat = e.rack_features()
        best, best_score = None, 1e18
        for r in range(e.num_racks):
            hosts = e._pick_hosts(r, job.num_nodes)
            if hosts is None:
                continue
            # predicted added peak on anchor rack
            added = job.peak_per_node_w * sum(1 for h in hosts if e.rack_of(h) == r)
            proj = feat[r, 2] * e.rack_budget + added
            score = proj  # lower is better; prefer staying under budget
            if score < best_score:
                best, best_score = r, score
        return best

    def act(self):
        e = self.env
        # PLACE
        for job in list(e.pending):
            r = self._best_rack(job)
            if r is not None:
                e.place_job(job, r)
        # DETECT + RELIEVE
        feat = e.rack_features()
        traj = e._future_traj()
        fut_peak = traj.max(axis=1) if traj.size else np.zeros(e.num_racks)
        at_risk = int(np.argmax(fut_peak))
        if fut_peak[at_risk] > e.rack_budget:
            victims = e.jobs_on_rack(at_risk)
            if victims:
                victim = min(victims, key=lambda j: (j.priority, -j.peak_per_node_w))
                e.preempt_job(victim)


def smoke(port, steps=60):
    env = GangEnv(port=port, subsample=float(os.environ.get("SUBSAMPLE", 0.05)),
                  max_arrival_step=48, max_steps=steps)
    print(f"hosts={env.num_hosts} racks={env.num_racks} budget={env.rack_budget:.0f}W "
          f"jobs={len(env._all_jobs)} max_steps={env.max_steps}")
    sched = HeuristicScheduler(env)
    peak_viol = 0
    for t in range(steps):
        sched.act()
        m = env.advance()
        peak_viol = max(peak_viol, m["rack_viol"])
        if t % 8 == 0 or m["rack_viol"] > 0:
            print(f"  t={t:2d} run={m['n_running']:2d} pend={m['n_pending']:3d} "
                  f"served={m['served']:2d} viol={m['rack_viol']:4d} "
                  f"E={m['energy_kwh']:6.1f}kWh ckpt={m['ckpt_nodes']:2d} "
                  f"reward={m['reward']:+.3f}")
        if env.done:
            break
    s = env.ep_stats
    print(f"\nEP: admitted={s['admitted']} preempted={s['preempted']} "
          f"completed={s['completed']} rack_viol={s['rack_viol']} "
          f"wait_cost={s['wait_cost']:.0f} energy={s['energy']:.1f}kWh")
    assert s["admitted"] > 0, "no jobs admitted"
    print("SMOKE OK: gang loop live (arrivals->place->detect->preempt->step)")
    return env


if __name__ == "__main__":
    import sys
    smoke(int(sys.argv[1]) if len(sys.argv) > 1 else 25333)
