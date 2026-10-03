import sys
from baselines.common import SchedulerBridgeAdapter, RunningJob
from nrel_injection_bridge import Job

class RPA_Scheduler:
    def __init__(self, adapter: SchedulerBridgeAdapter):
        self.adapter = adapter
        self.pending_queue = []

    def add_jobs(self, jobs):
        self.pending_queue.extend(jobs)
        # Sort pending queue: earliest arrival first
        self.pending_queue.sort(key=lambda j: (j.arrival_step, -j.num_nodes))

    def select_first_fit(self, job: Job, step: int) -> int:
        """First Fit focusing only on Node Availability + Anti-Oscillation."""
        # Constraint 5: Not before step
        if step < job.not_before_step:
            return None

        candidate_racks = []
        for rack_id in range(self.adapter.num_racks):
            free_nodes = self.adapter.get_rack_free_nodes(rack_id)
            if free_nodes >= job.num_nodes:
                candidate_racks.append(rack_id)

        if not candidate_racks:
            return None

        # Constraint 5: Avoid last preempted rack on FIRST available reschedule
        # We try to pick any rack that is NOT the last preempted rack
        valid_others = [r for r in candidate_racks if r != job.last_preempted_rack]

        if valid_others:
            return valid_others[0] # First-fit among other racks

        # Fallback to the same rack only if it's the ONLY available rack cluster-wide
        return candidate_racks[0]

    def schedule(self, step: int, gateway):
        """Places jobs First-Fit blindly, irrespective of power."""
        still_pending = []
        for job in self.pending_queue:
            rack_id = self.select_first_fit(job, step)
            if rack_id is not None:
                success = self.adapter.submit_job_to_rack(job, rack_id, step, gateway)
                if not success:
                    still_pending.append(job)
            else:
                still_pending.append(job)
        self.pending_queue = still_pending

    def mitigate_violations(self, step: int):
        """Constraint 7: Monitor power and mitigate via LIFO Preemption."""
        for rack_id in range(self.adapter.num_racks):
            effective_power = self.adapter.get_effective_rack_power(rack_id)

            if effective_power <= self.adapter.rack_budget_w:
                continue

            # Rack violated budget! Get running jobs
            running_jobs = self.adapter.get_running_jobs_on_rack(rack_id)

            # Constraint 4: Sort LIFO based on start_step (newest killed first)
            running_jobs.sort(key=lambda rj: rj.start_step if rj.start_step is not None else -1, reverse=True)

            for victim in running_jobs:
                if victim.job_id not in self.adapter.job_registry:
                    continue # Should not happen unless ghost job

                job_obj = self.adapter.job_registry[victim.job_id]

                # Preempt job
                self.adapter.preempt_job(job_obj, step, cooldown_steps=2) # 2 steps cooldown
                self.pending_queue.append(job_obj) # put back in queue

                # Constraint 7: Update power without calling getRackPowerW() again
                effective_power -= job_obj.estimated_power_w

                if effective_power <= self.adapter.rack_budget_w:
                    break # Safely mitigated
