"""
Convert HeliosData cluster_log.csv → pod_hourly_jobs.csv (Alibaba schema).

Alibaba schema: workload_id, arrival_hour, total_gpu_request, num_pods, job_type, model_type

Helios schema:  job_id, user, vc, gpu_num, cpu_num, node_num, state, submit_time,
                start_time, end_time, duration, queue

Mapping:
  workload_id       = job_id (string)
  arrival_hour      = (submit_time - t0) in hours
  total_gpu_request = gpu_num
  num_pods          = node_num   (closest analogue)
  job_type          = "training" (all Helios DL jobs → training)
  model_type        = vc         (virtual-cluster used as proxy model category)
"""
import pandas as pd
import sys
import os

def convert(input_csv, output_csv, max_jobs=None):
    df = pd.read_csv(input_csv)
    print(f"[helios] loaded {len(df)} rows from {input_csv}")
    print(f"[helios] columns: {list(df.columns)}")
    print(f"[helios] state distribution:\n{df['state'].value_counts().to_string()}")

    # Filter: only GPU jobs that actually ran (COMPLETED or FAILED with gpu_num > 0)
    df = df[df["gpu_num"] > 0].copy()
    print(f"[helios] after gpu_num > 0 filter: {len(df)} rows")

    # Filter: only jobs that have a valid submit_time
    df = df.dropna(subset=["submit_time"])
    df["submit_time"] = pd.to_datetime(df["submit_time"])
    print(f"[helios] date range: {df['submit_time'].min()} → {df['submit_time'].max()}")

    # Compute arrival_hour relative to earliest submit
    t0 = df["submit_time"].min()
    df["arrival_hour"] = ((df["submit_time"] - t0).dt.total_seconds() / 3600.0).astype(int)

    # Map to Alibaba schema
    out = pd.DataFrame({
        "workload_id": df["job_id"].astype(str),
        "arrival_hour": df["arrival_hour"],
        "total_gpu_request": df["gpu_num"].astype(float),
        "num_pods": df["node_num"].astype(int),
        "job_type": "training",   # all Helios DL jobs
        "model_type": "unknown",  # VC names don't map to known Alibaba types (cv/genai);
                                  # matcher falls through to mixture case regardless
    })

    # Sort by arrival
    out = out.sort_values("arrival_hour").reset_index(drop=True)

    if max_jobs is not None:
        out = out.head(max_jobs)

    out.to_csv(output_csv, index=False)
    print(f"\n[helios→alibaba] wrote {len(out)} jobs → {output_csv}")
    print(f"[helios→alibaba] arrival_hour range: {out['arrival_hour'].min()} → {out['arrival_hour'].max()}")
    print(f"[helios→alibaba] total_gpu_request stats:\n{out['total_gpu_request'].describe().to_string()}")
    print(f"[helios→alibaba] model_type (VC) distribution:\n{out['model_type'].value_counts().head(10).to_string()}")

    return out

if __name__ == "__main__":
    cluster = sys.argv[1] if len(sys.argv) > 1 else "Earth"
    max_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "HeliosData", "data", cluster, "cluster_log.csv")
    output_csv = os.path.join(script_dir, f"helios_{cluster.lower()}_pod_hourly_jobs.csv")
    
    convert(input_csv, output_csv, max_jobs)
