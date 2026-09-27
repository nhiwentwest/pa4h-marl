"""Convert official Alibaba GPU v2020 job/task tables to the gang queue schema.

Job start_time is submission. Runtime is the span of successful task execution,
which avoids treating queue wait as GPU service time. plan_gpu is a percentage
of one GPU per task instance in the v2020 trace.
"""
import argparse
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


JOB_COLUMNS = ["job_name", "inst_id", "user", "status", "start_time", "end_time"]
TASK_COLUMNS = ["job_name", "task_name", "inst_num", "status", "start_time",
                "end_time", "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"]
TAG_COLUMNS = ["inst_id", "user", "gpu_type_spec", "group", "workload"]


def read_table(path, member_name, columns, usecols):
    """Read headerless official CSV directly from tar.gz, without extracting it."""
    path = Path(path)
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            member = next((m for m in archive.getmembers()
                           if Path(m.name).name == member_name), None)
            if member is None:
                raise ValueError(f"{member_name} missing from {path}")
            with archive.extractfile(member) as stream:
                return pd.read_csv(stream, header=None, names=columns, usecols=usecols)
    return pd.read_csv(path, header=None, names=columns, usecols=usecols)


def convert(job_path, task_path, output_csv, tag_path=None):
    jobs = read_table(job_path, "pai_job_table.csv", JOB_COLUMNS,
                      ["job_name", "inst_id", "status", "start_time"])
    tasks = read_table(task_path, "pai_task_table.csv", TASK_COLUMNS,
                       ["job_name", "inst_num", "status", "start_time",
                        "end_time", "plan_gpu"])
    jobs = jobs.loc[jobs.status == "Terminated"].copy()
    tasks = tasks.loc[tasks.status == "Terminated"].copy()
    for col in ("inst_num", "start_time", "end_time", "plan_gpu"):
        tasks[col] = pd.to_numeric(tasks[col], errors="coerce")
    jobs["start_time"] = pd.to_numeric(jobs["start_time"], errors="coerce")
    tasks = tasks.dropna(subset=["job_name", "inst_num", "start_time",
                                 "end_time", "plan_gpu"])
    tasks = tasks.loc[(tasks.inst_num > 0) & (tasks.plan_gpu > 0)
                      & (tasks.end_time > tasks.start_time)].copy()
    tasks["gpu_request"] = tasks.inst_num * tasks.plan_gpu / 100.0
    grouped = tasks.groupby("job_name", sort=False).agg(
        total_gpu_request=("gpu_request", "sum"),
        num_pods=("inst_num", "sum"),
        task_start=("start_time", "min"),
        task_end=("end_time", "max"))
    merged = jobs.dropna(subset=["job_name", "inst_id", "start_time"])
    merged = merged.drop_duplicates(subset=["job_name"]).join(grouped, on="job_name", how="inner")
    if tag_path is not None:
        tags = read_table(tag_path, "pai_group_tag_table.csv", TAG_COLUMNS,
                          ["inst_id", "workload"])
        tags = tags.dropna(subset=["inst_id", "workload"])
        tags = tags.drop_duplicates(subset=["inst_id"]).set_index("inst_id")
        merged = merged.join(tags[["workload"]], on="inst_id", how="left")
        model_type = merged["workload"].fillna("unknown").astype(str)
    else:
        model_type = pd.Series("unknown", index=merged.index)
    merged["duration_hours"] = (merged.task_end - merged.task_start) / 3600.0
    merged = merged.loc[(merged.total_gpu_request > 0)
                        & (merged.duration_hours > 0)
                        & np.isfinite(merged.duration_hours)].copy()
    t0 = merged.start_time.min()
    merged["arrival_hour"] = np.floor((merged.start_time - t0) / 3600).astype(int)
    out = pd.DataFrame({
        "workload_id": merged.inst_id.astype(str),
        "arrival_hour": merged.arrival_hour,
        "total_gpu_request": merged.total_gpu_request,
        "num_pods": merged.num_pods.astype(int),
        "job_type": "training",
        "model_type": model_type.loc[merged.index],
        "duration_hours": merged.duration_hours,
    }).sort_values(["arrival_hour", "workload_id"]).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"[alibaba-v2020] {len(out)} successful GPU jobs, "
          f"hours={out.arrival_hour.min()}..{out.arrival_hour.max()}, "
          f"output={output_csv}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_path")
    parser.add_argument("task_path")
    parser.add_argument("output_csv")
    parser.add_argument("--tags", help="Optional pai_group_tag_table.tar.gz for workload labels")
    args = parser.parse_args()
    convert(args.job_path, args.task_path, args.output_csv, tag_path=args.tags)
