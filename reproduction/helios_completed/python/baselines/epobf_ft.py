import sys
from baselines.common import SchedulerBridgeAdapter
from nrel_injection_bridge import Job

class EPOBF_Scheduler:
    def __init__(self, adapter: SchedulerBridgeAdapter):
        self.adapter = adapter
        self.pending_queue = []

    def add_jobs(self, jobs):
        self.pending_queue.extend(jobs)
        # Sort pending queue: earliest arrival first, then largest node req
        self.pending_queue.sort(key=lambda j: (j.arrival_step, -j.num_nodes))

    def select_rack(self, job: Job) -> int:
        """Constraint 8: EPOBF compares Watt with Watt and selects minimum non-negative residual."""
        candidates = []

        for rack_id in range(self.adapter.num_racks):
            free_nodes = self.adapter.get_rack_free_nodes(rack_id)
            if free_nodes < job.num_nodes:
                continue

            effective_power = self.adapter.get_effective_rack_power(rack_id)
            headroom_w = self.adapter.rack_budget_w - effective_power
            residual_w = headroom_w - job.estimated_power_w

            if residual_w >= 0: # 0 means exact fit
                # candidates tuple: (residual_w, remaining_nodes, rack_id)
                candidates.append((residual_w, free_nodes - job.num_nodes, rack_id))

        if not candidates:
            return None

        candidates.sort() # Sorts by residual_w first (Best-Fit)
        return candidates[0][2] # return rack_id

    def schedule(self, step: int, gateway):
        """Attempts to schedule all pending jobs using EPOBF logic."""
        still_pending = []
        for job in self.pending_queue:
            rack_id = self.select_rack(job)
            if rack_id is not None:
                success = self.adapter.submit_job_to_rack(job, rack_id, step, gateway)
                if not success:
                    # Rare race condition if free_nodes changed unexpectedly
                    still_pending.append(job)
            else:
                still_pending.append(job)

        self.pending_queue = still_pending
