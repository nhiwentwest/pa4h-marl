import sys
from typing import Dict
from baselines.common import SchedulerBridgeAdapter
from nrel_injection_bridge import Job

class Oracle_LA50_Scheduler:
    def __init__(self, adapter: SchedulerBridgeAdapter, lookahead_steps: int = 50):
        self.adapter = adapter
        self.lookahead_steps = lookahead_steps
        self.pending_queue = []

        # Constraint 9: 2D Time-Series Matrix for reservation (Rack -> Step -> Val)
        self.reserved_nodes: Dict[int, Dict[int, int]] = {r: {} for r in range(self.adapter.num_racks)}
        self.reserved_power_w: Dict[int, Dict[int, float]] = {r: {} for r in range(self.adapter.num_racks)}

    def add_jobs(self, jobs):
        self.pending_queue.extend(jobs)
        # Sort pending queue: earliest arrival first, then largest jobs
        self.pending_queue.sort(key=lambda j: (j.arrival_step, -j.num_nodes))

    def _get_reserved_nodes(self, rack_id: int, step: int) -> int:
        return self.reserved_nodes[rack_id].get(step, 0)

    def _get_reserved_power(self, rack_id: int, step: int) -> float:
        return self.reserved_power_w[rack_id].get(step, 0.0)

    def plan_job(self, job: Job, current_step: int) -> bool:
        """Finds a 2D block (rack x time) where the job perfectly fits without violating budgets."""
        # Search space: current_step to current_step + lookahead
        max_search_step = current_step + self.lookahead_steps

        for candidate_start in range(current_step, max_search_step):
            for rack_id in range(self.adapter.num_racks):

                can_fit = True
                # Check entire duration
                for s in range(candidate_start, candidate_start + job.duration_steps):
                    # Estimate nodes
                    actual_free_nodes = self.adapter.get_rack_free_nodes(rack_id)
                    # Node capacity for future steps = currently free - future reservations
                    future_free_nodes = actual_free_nodes - self._get_reserved_nodes(rack_id, s)

                    if future_free_nodes < job.num_nodes:
                        can_fit = False
                        break

                    # Estimate power
                    actual_power = self.adapter.get_effective_rack_power(rack_id)
                    future_power = actual_power + self._get_reserved_power(rack_id, s) + job.estimated_power_w

                    if future_power > self.adapter.rack_budget_w:
                        can_fit = False
                        break

                if can_fit:
                    # Plan found!
                    job.planned_start_step = candidate_start
                    job.planned_rack = rack_id

                    # Lock reservations
                    for s in range(candidate_start, candidate_start + job.duration_steps):
                        self.reserved_nodes[rack_id][s] = self._get_reserved_nodes(rack_id, s) + job.num_nodes
                        self.reserved_power_w[rack_id][s] = self._get_reserved_power(rack_id, s) + job.estimated_power_w

                    return True

        return False

    def schedule(self, step: int, gateway):
        """Plans all pending jobs in Lookahead window, and executes those starting at THIS step."""
        still_pending = []

        for job in self.pending_queue:
            if job.planned_start_step is None:
                # Try to plan it
                success = self.plan_job(job, step)
                if not success:
                    still_pending.append(job) # Wait until space opens up in Lookahead window

            if job.planned_start_step == step:
                # Time to execute!
                success = self.adapter.submit_job_to_rack(job, job.planned_rack, step, gateway)
                if not success:
                    # Execution failed! Free map diverges from prediction. Need to replan!
                    job.planned_start_step = None
                    job.planned_rack = None
                    still_pending.append(job)
            elif job.planned_start_step is not None and job.planned_start_step > step:
                still_pending.append(job)

        self.pending_queue = still_pending

        # Cleanup past matrices to save memory
        for r in range(self.adapter.num_racks):
            if step - 1 in self.reserved_nodes[r]:
                del self.reserved_nodes[r][step - 1]
            if step - 1 in self.reserved_power_w[r]:
                del self.reserved_power_w[r][step - 1]
