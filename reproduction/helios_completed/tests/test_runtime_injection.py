import pandas as pd
import numpy as np
import nrel_injection_bridge as injection


def test_row_runtime_ignores_unrelated_duration_cache_and_preserves_boundaries(tmp_path,monkeypatch):
    # Find a stable ID whose offset reaches the non-hour-aligned start.
    wid=next(str(v) for v in range(100) if injection._hash_int(str(v))%injection.STEPS_PER_HOUR>=3)
    offset=injection._hash_int(wid)%injection.STEPS_PER_HOUR
    frame=pd.DataFrame(dict(workload_id=[wid,'outside'],arrival_hour=[1,3],
        total_gpu_request=[8,4],job_type=['training','training'],
        model_type=['unknown','unknown'],duration_hours=[1.,100.]))
    path=tmp_path/'jobs.csv';frame.to_csv(path,index=False)
    monkeypatch.setattr(injection.M,'build_nrel_registry',lambda _:['profile'])
    monkeypatch.setattr(injection.M,'match_job',lambda *_: (None,{'path':'profile'},{'nodes':2}))
    monkeypatch.setattr(injection.M,'per_node_power',lambda *_:(np.arange(4),np.ones(4)*500))
    def unrelated_cache():raise AssertionError('row runtime must not join Alibaba ID cache')
    monkeypatch.setattr(injection,'_load_duration_map',unrelated_cache)
    start=injection.STEPS_PER_HOUR+3;end=injection.STEPS_PER_HOUR+offset+1
    jobs=injection.build_job_queue(str(path),'unused',min_arrival_step=start,
                                  arrival_end_step=end,verbose=False)
    assert len(jobs)==1 and jobs[0].job_id==wid
    assert jobs[0].duration_steps==3600//injection.INTERVAL_SEC
    assert jobs[0].num_nodes==2
    assert injection.build_job_queue(str(path),'unused',min_arrival_step=start,
        arrival_end_step=end-1,verbose=False)==[]
