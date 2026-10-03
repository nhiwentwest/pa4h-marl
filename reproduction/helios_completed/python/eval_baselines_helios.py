"""Evaluate all canonical policies on HeliosData Earth + NREL power traces."""
import os

# These must be set before eval_gang imports GangEnv and the workload adapter.
os.environ.setdefault("POD_HOURLY_JOBS", "data/helios_earth_pod_hourly_jobs.csv")
os.environ.setdefault("WORKLOAD_START_HOUR", "1000")
os.environ.setdefault("WORKLOAD_WINDOW_HOURS", "10")
os.environ.setdefault("SUBSAMPLE", "0.05")
os.environ.setdefault("STEPS", "120")
os.environ.setdefault("USE_STGNN", "1")
os.environ.setdefault(
    "GANG_CHECKPOINT_DIR",
    "outputs/gang_v2_helios_nrel_300eps_20260715",
)
os.environ.setdefault(
    "GANG_EVAL_CSV",
    "outputs/gang_v2_helios_nrel_300eps_20260715/eval_baselines_helios.csv",
)

from eval_gang import main


if __name__ == "__main__":
    main()
