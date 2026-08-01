import os
import sys
import time
import pandas as pd
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters
from nrel_injection_bridge import build_job_queue
from baselines.common import SchedulerBridgeAdapter
from baselines.epobf_ft import EPOBF_Scheduler
from baselines.rpa import RPA_Scheduler
from baselines.safe_res_la50 import SafeRes_LA50_Scheduler

def evaluate_baseline(baseline_name, queue, gateway):
    bridge = gateway.entry_point
    bridge.reset()
    
    # Setup
    adapter = SchedulerBridgeAdapter(bridge, num_hosts=20, hosts_per_rack=4, rack_budget_w=bridge.getRackBudgetW())
    adapter.register_jobs(queue)
    
    if baseline_name == "EPOBF":
        scheduler = EPOBF_Scheduler(adapter)
    elif baseline_name == "RPA":
        scheduler = RPA_Scheduler(adapter)
    elif baseline_name == "SafeRes":
        scheduler = SafeRes_LA50_Scheduler(adapter, lookahead_steps=20) # Keep lookahead reasonable
        
    print(f"\n[EVAL] Starting {baseline_name} Baseline with {len(queue)} jobs.")
    
    # State tracking
    step = 0
    job_idx = 0
    total_viol = 0
    total_energy_wh = 0.0
    
    start_time = time.time()
    
    while True:
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
        
        # 5. Post-step mitigation (RPA only)
        if baseline_name == "RPA":
            scheduler.mitigate_violations(step)
            
        # 6. Collect Step Metrics
        power = bridge.getRackPowerW()
        viols = bridge.getRackViolCountStep()
        
        # Aggregate Energy (Wh) = Power (W) * (Interval / 3600)
        # We assume 300s per step
        total_energy_wh += sum(power) * (300.0 / 3600.0)
        total_viol += sum(viols)
        
        # Check termination condition
        active_jobs_count = len(bridge.getRunningJobIds())
        pending_jobs_count = len(scheduler.pending_queue)
        
        # If all jobs have arrived, and no jobs are pending, and no jobs are running
        if job_idx >= len(queue) and active_jobs_count == 0 and pending_jobs_count == 0:
            break
            
        # Safety limit for simulation
        if step >= 1500:
            print(f"[WARN] Hit hard limit of 1500 steps!")
            break
            
        if step % 50 == 0:
            print(f"  Step {step}: Pending={pending_jobs_count}, Running={active_jobs_count}, Viols={total_viol}")
            
        step += 1
        
    # Calculate Summary Metrics
    total_preemptions = sum(j.preemption_count for j in queue)
    makespan = step
    
    avg_wait = sum((j.start_step - j.arrival_step) for j in queue if j.start_step is not None) / len(queue) if len(queue) > 0 else 0
    
    elapsed_real = time.time() - start_time
    print(f"[EVAL] {baseline_name} Finished in {elapsed_real:.1f}s.")
    print(f"       Makespan={makespan}, AvgWait={avg_wait:.2f}, Viols={total_viol}, Preempts={total_preemptions}, Energy={total_energy_wh/1000:.2f}kWh")
    
    return {
        "Baseline": baseline_name,
        "Makespan (Steps)": makespan,
        "Avg Wait (Steps)": round(avg_wait, 2),
        "Total Preemptions": total_preemptions,
        "Power Violations": total_viol,
        "Energy (kWh)": round(total_energy_wh / 1000.0, 2)
    }


if __name__ == "__main__":
    csv_path = os.environ.get(
        "POD_HOURLY_JOBS",
        "data/alibaba/pod_hourly_jobs.csv")
    # Use full queue for evaluation
    full_queue = build_job_queue(csv_path)
    
    results = []
    
    gateway = JavaGateway(
        gateway_parameters=GatewayParameters(port=25333, auto_convert=True),
        callback_server_parameters=CallbackServerParameters(port=0)
    )
    
    for b in ["EPOBF", "RPA", "SafeRes"]:
        import copy
        q = copy.deepcopy(full_queue)
        res = evaluate_baseline(b, q, gateway)
        results.append(res)
        
    gateway.shutdown()
        
    # Save to CSV
    df = pd.DataFrame(results)
    out_path = os.environ.get("BASELINE_RESULTS_CSV", "baseline_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved full results to {out_path}")
    print(df.to_markdown(index=False))
