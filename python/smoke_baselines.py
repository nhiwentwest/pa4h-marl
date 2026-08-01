import os
import sys
import time
from py4j.java_gateway import JavaGateway
from nrel_injection_bridge import build_job_queue
from baselines.common import SchedulerBridgeAdapter
from baselines.epobf_ft import EPOBF_Scheduler
from baselines.rpa import RPA_Scheduler
from baselines.oracle_la50 import Oracle_LA50_Scheduler

def run_smoke_test(baseline_name, queue):
    gateway = JavaGateway()
    bridge = gateway.entry_point
    bridge.reset()
    
    # In smoke test, NUM_HOSTS is 20
    adapter = SchedulerBridgeAdapter(bridge, num_hosts=20, hosts_per_rack=4, rack_budget_w=16000.0)
    adapter.register_jobs(queue)
    
    if baseline_name == "EPOBF":
        scheduler = EPOBF_Scheduler(adapter)
    elif baseline_name == "RPA":
        scheduler = RPA_Scheduler(adapter)
    elif baseline_name == "Oracle":
        scheduler = Oracle_LA50_Scheduler(adapter, lookahead_steps=10)
    
    print(f"\n--- Running {baseline_name} Baseline ---")
    
    total_steps = 100
    job_idx = 0
    
    for step in range(total_steps):
        # 1. Reset telemetry delta
        adapter.reset_telemetry_delta()
        
        # 2. Get arrived jobs
        arrived = []
        while job_idx < len(queue) and queue[job_idx].arrival_step <= step:
            arrived.append(queue[job_idx])
            job_idx += 1
            
        if arrived:
            scheduler.add_jobs(arrived)
            
        # 3. Schedule jobs
        scheduler.schedule(step, gateway)
        
        # 4. Advance step (bypass RL)
        adapter.advance_step(gateway)
        
        # 5. Post-step mitigation (only for RPA)
        if baseline_name == "RPA":
            scheduler.mitigate_violations(step)
            
        # Print metrics every 20 steps
        if step % 20 == 0:
            power = bridge.getRackPowerW()
            viols = bridge.getRackViolCountStep()
            print(f"Step {step}: Pending Jobs={len(scheduler.pending_queue)}, Running Jobs={len(bridge.getRunningJobIds())}")
            print(f"  Rack Power: {[round(p, 1) for p in power]}")

if __name__ == '__main__':
    csv_path = os.environ.get("POD_HOURLY_JOBS", "data/alibaba/pod_hourly_jobs.csv")
    # load only first 100 jobs for smoke test to be fast
    full_queue = build_job_queue(csv_path)
    
    for b in ["EPOBF", "RPA", "Oracle"]:
        # Give each baseline a fresh slice of the first 20 jobs
        import copy
        q = copy.deepcopy(full_queue[:20])
        # Force them to arrive within the first 50 steps for the test
        for i, j in enumerate(q):
            j.arrival_step = i * 2 
            
        run_smoke_test(b, q)
        time.sleep(1)
