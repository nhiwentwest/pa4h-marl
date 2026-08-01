"""
safe_res_la50.py — SAFERES: safe look-ahead reservation baseline.

NOTE: reconstructed (2026-08-01) from oracle_la50.py (same reservation
machinery) and the call site in eval_baselines.py, after the original stayed
on the Lightning workspace. SAFERES reserves a rack-time block only when both
node capacity and estimated mean power remain feasible for the whole job
duration inside a bounded look-ahead window (default 20 steps in
eval_baselines.py). The self-contained port used for the paper's final
numbers lives in eval_gang.py (class SafeRes); this module keeps the phase-2
adapter-based harness runnable.
"""
from typing import Dict

from baselines.common import SchedulerBridgeAdapter
from nrel_injection_bridge import Job


class SafeRes_LA50_Scheduler:
    def __init__(self, adapter: SchedulerBridgeAdapter, lookahead_steps: int = 20):
        self.adapter = adapter
        self.lookahead_steps = lookahead_steps
        self.pending_queue = []

        # 2D reservation matrices (rack -> step -> reserved amount)
        self.reserved_nodes: Dict[int, Dict[int, int]] = {
            r: {} for r in range(self.adapter.num_racks)}
        self.reserved_power_w: Dict[int, Dict[int, float]] = {
            r: {} for r in range(self.adapter.num_racks)}

    def add_jobs(self, jobs):
        self.pending_queue.extend(jobs)
        self.pending_queue.sort(key=lambda j: (j.arrival_step, -j.num_nodes))

    def _get_reserved_nodes(self, rack_id: int, step: int) -> int:
        return self.reserved_nodes[rack_id].get(step, 0)

    def _get_reserved_power(self, rack_id: int, step: int) -> float:
        return self.reserved_power_w[rack_id].get(step, 0.0)

    def plan_job(self, job: Job, current_step: int) -> bool:
        """Find a rack-time block where the job fits without violating the
        rack power budget for its entire duration."""
        max_search_step = current_step + self.lookahead_steps

        for candidate_start in range(current_step, max_search_step):
            for rack_id in range(self.adapter.num_racks):
                can_fit = True
                for s in range(candidate_start,
                               candidate_start + job.duration_steps):
                    actual_free_nodes = self.adapter.get_rack_free_nodes(rack_id)
                    future_free_nodes = (actual_free_nodes
                                         - self._get_reserved_nodes(rack_id, s))
                    if future_free_nodes < job.num_nodes:
                        can_fit = False
                        break

                    actual_power = self.adapter.get_effective_rack_power(rack_id)
                    future_power = (actual_power
                                    + self._get_reserved_power(rack_id, s)
                                    + job.estimated_power_w)
                    if future_power > self.adapter.rack_budget_w:
                        can_fit = False
                        break

                if can_fit:
                    job.planned_start_step = candidate_start
                    job.planned_rack = rack_id
                    for s in range(candidate_start,
                                   candidate_start + job.duration_steps):
                        self.reserved_nodes[rack_id][s] = (
                            self._get_reserved_nodes(rack_id, s) + job.num_nodes)
                        self.reserved_power_w[rack_id][s] = (
                            self._get_reserved_power(rack_id, s)
                            + job.estimated_power_w)
                    return True
        return False

    def schedule(self, step: int, gateway):
        """Plan pending jobs inside the look-ahead window; execute the ones
        whose planned start is THIS step."""
        still_pending = []
        for job in self.pending_queue:
            if job.planned_start_step is None:
                if not self.plan_job(job, step):
                    still_pending.append(job)

            if job.planned_start_step == step:
                ok = self.adapter.submit_job_to_rack(
                    job, job.planned_rack, step, gateway)
                if not ok:
                    # Reality diverged from the reservation — replan later.
                    job.planned_start_step = None
                    job.planned_rack = None
                    still_pending.append(job)
            elif job.planned_start_step is not None and job.planned_start_step > step:
                still_pending.append(job)

        self.pending_queue = still_pending

        # Drop expired reservation rows to bound memory.
        for r in range(self.adapter.num_racks):
            self.reserved_nodes[r].pop(step - 1, None)
            self.reserved_power_w[r].pop(step - 1, None)
