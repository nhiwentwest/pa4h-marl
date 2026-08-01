import sys
from collections import defaultdict
from nrel_injection_bridge import Job

class RunningJob:
    def __init__(self, job_id, start_step, rack_id, host_ids, power_w):
        self.job_id = job_id
        self.start_step = start_step
        self.rack_id = rack_id
        self.host_ids = host_ids
        self.power_w = power_w

class SchedulerBridgeAdapter:
    def __init__(self, bridge, num_hosts: int, hosts_per_rack: int, rack_budget_w: float):
        self.bridge = bridge
        self.num_hosts = num_hosts
        self.hosts_per_rack = hosts_per_rack
        self.rack_budget_w = rack_budget_w
        self.num_racks = (num_hosts + hosts_per_rack - 1) // hosts_per_rack
        
        # Constraint 1: Topology safely handles boundary
        self.rack_hosts = defaultdict(list)
        for start in range(0, num_hosts, hosts_per_rack):
            rack_id = start // hosts_per_rack
            end = min(num_hosts, start + hosts_per_rack)
            self.rack_hosts[rack_id] = list(range(start, end))
            
        self.job_registry = {}
        
        # Constraint 6: Safe Telemetry Delta
        self.reserved_power_w = defaultdict(float)
        self.estimated_relief_w = defaultdict(float)
        
        # Py4j Optimization caching
        self._cached_free_map = None
        self._cached_rack_power = None
        
    def reset_telemetry_delta(self):
        """Called at start of each step to clear deltas."""
        self.reserved_power_w.clear()
        self.estimated_relief_w.clear()
        self._cached_free_map = None
        self._cached_rack_power = None

    def register_jobs(self, jobs: list[Job]):
        for j in jobs:
            self.job_registry[j.job_id] = j
            
    def _get_free_map(self):
        if self._cached_free_map is None:
            self._cached_free_map = list(self.bridge.getHostFreeMap())
        return self._cached_free_map

    def get_free_host_ids(self, rack_id: int) -> list[int]:
        free_map = self._get_free_map()
        return [h for h in self.rack_hosts[rack_id] if bool(free_map[h])]

    def get_rack_free_nodes(self, rack_id: int) -> int:
        return len(self.get_free_host_ids(rack_id))

    def get_effective_rack_power(self, rack_id: int) -> float:
        if self._cached_rack_power is None:
            self._cached_rack_power = list(self.bridge.getRackPowerW())
        base_power = self._cached_rack_power[rack_id]
        return base_power + self.reserved_power_w[rack_id] - self.estimated_relief_w[rack_id]

    def submit_job_to_rack(self, job: Job, rack_id: int, step: int, gateway) -> bool:
        free_hosts = self.get_free_host_ids(rack_id)
        if len(free_hosts) < job.num_nodes:
            return False
            
        selected_hosts = free_hosts[:job.num_nodes]
        
        # Convert to Java types
        j_host_array = gateway.new_array(gateway.jvm.int, len(selected_hosts))
        for i, h in enumerate(selected_hosts):
            j_host_array[i] = int(h)
            
        j_trace_array = gateway.new_array(gateway.jvm.double, len(job.per_node_trace_w))
        for i, val in enumerate(job.per_node_trace_w):
            j_trace_array[i] = float(val)
            
        # Submit
        self.bridge.submitJob(
            str(job.job_id), 
            j_host_array, 
            j_trace_array, 
            float(job.dt_sec),
            int(job.duration_steps), 
            float(job.priority)
        )
        
        # Update cache locally
        for h in selected_hosts:
            self._cached_free_map[h] = False
            
        job.start_step = step
        job.assigned_rack = rack_id
        job.assigned_host_ids = tuple(selected_hosts)
        job.status = "running"
        
        self.reserved_power_w[rack_id] += job.estimated_power_w
        return True

    def preempt_job(self, job: Job, step: int, cooldown_steps: int = 0):
        self.bridge.preemptJob(str(job.job_id))
        
        job.last_preempted_rack = job.assigned_rack
        job.not_before_step = step + cooldown_steps
        job.preemption_count += 1
        job.status = "pending"
        
        # Update cache locally
        if self._cached_free_map is not None:
            for h in job.assigned_host_ids:
                self._cached_free_map[h] = True
                
        self.estimated_relief_w[job.assigned_rack] += job.estimated_power_w
        
        job.start_step = None
        job.assigned_rack = None
        job.assigned_host_ids = ()

    def get_running_jobs_on_rack(self, rack_id: int) -> list[RunningJob]:
        # bridge.getRunningJobIds() returns a list of strings
        # We must filter those that are actually on this rack
        all_ids = list(self.bridge.getRunningJobIds())
        
        rack_jobs = []
        for jid in all_ids:
            if jid in self.job_registry:
                j = self.job_registry[jid]
                if j.assigned_rack == rack_id:
                    rack_jobs.append(RunningJob(j.job_id, j.start_step, rack_id, j.assigned_host_ids, j.estimated_power_w))
                    
        return rack_jobs

    def advance_step(self, gateway):
        # We need to create empty java int[] arrays for overloaded/underloaded
        j_over = gateway.new_array(gateway.jvm.int, 0)
        j_under = gateway.new_array(gateway.jvm.int, 0)
        j_place = gateway.new_array(gateway.jvm.int, 0)
        
        # Advance simulation by 1 RL step (300 seconds)
        # We bypass the RL action by passing empty arrays
        self.bridge.step(j_over, j_under, -1, j_place)
