import os
import sys
import time
from py4j.java_gateway import JavaGateway, GatewayParameters
from nrel_injection_bridge import build_job_queue

try:
    print('Connecting to Py4jGateway...')
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=25333))
    sim = gateway.entry_point

    sim.reset()

    num_hosts = sim.getNumHosts()
    print(f'Simulator initialized with {num_hosts} hosts.')

    csv_path = os.environ.get("POD_HOURLY_JOBS", "data/alibaba/pod_hourly_jobs.csv")
    queue = build_job_queue(csv_path)

    # Take first 10 jobs
    queue = queue[:10]

    print('Starting simulation loop for Phase 2 Smoke Test...')
    DoubleArray = gateway.jvm.double
    IntArray = gateway.jvm.int

    empty_int_arr = gateway.new_array(IntArray, 0)

    current_step = 0
    jobs_submitted = 0

    # We will step until all 10 jobs are submitted (their arrival_step might be e.g. 100, 200...)
    # But since they all arrived in hour=0, offset_sec is 0-3599.
    # interval is 300s. So max arrival_step is 3599/300 = 11.
    # We can just simulate 15 steps.
    for step in range(15):
        print(f'\n--- Step {step} ---')

        # Check arrivals
        arrived_jobs = [j for j in queue if j.arrival_step == step]

        for job in arrived_jobs:
            print(f'Job Arrived: {job.job_id} requesting {job.num_nodes} nodes for {job.duration_h:.2f}h')

            # Find free hosts
            free_map = sim.getHostFreeMap()
            free_indices = [i for i, is_free in enumerate(free_map) if is_free]

            if len(free_indices) >= job.num_nodes:
                selected_hosts = free_indices[:job.num_nodes]

                # Create java arrays
                host_ids = gateway.new_array(IntArray, job.num_nodes)
                for i, h in enumerate(selected_hosts): host_ids[i] = h

                # Create trace (for now just 4000W peak synth trace to test injection)
                dur_steps = job.duration_steps
                trace = gateway.new_array(DoubleArray, min(1500, dur_steps * 1500)) # Just enough for a few steps to prevent memory bloat in test
                for i in range(len(trace)): trace[i] = 4000.0

                success = sim.submitJob(job.job_id, host_ids, trace, 0.2, dur_steps, 1.0)
                if success:
                    print(f'  => Successfully submitted to hosts: {selected_hosts}')
                    jobs_submitted += 1
                else:
                    print('  => Failed to submit (Java returned false)')
            else:
                print(f'  => Not enough free nodes! (Free: {len(free_indices)}, Req: {job.num_nodes})')

        # Advance simulation
        sim.step(empty_int_arr, empty_int_arr, -1, empty_int_arr)

        # Print power
        rack_power = sim.getRackPowerW()
        viol = sim.getRackViolCountStep()

        print('Rack powers: ', end='')
        for r in range(sim.getNumRacks()):
            print(f'R{r}: {rack_power[r]:.0f}W (V:{viol[r]})  ', end='')
        print()

    print(f'\nSmoke test 2.5 completed. Total jobs submitted: {jobs_submitted}')

except Exception as e:
    print(f'Error during smoke test: {e}')
