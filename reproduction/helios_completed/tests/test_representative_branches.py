import pytest
from representative_branches import priority
from representative_branches import RepresentativeRecorder
from types import SimpleNamespace
import numpy as np

def test_selects_long_gang_and_congested_states_without_reward():
    long=dict(duration=90,nodes=2,queued_demand=4,pending=2)
    gang=dict(duration=20,nodes=8,queued_demand=8,pending=1)
    queue=dict(duration=15,nodes=2,queued_demand=128,pending=64)
    cases=[long,gang,queue]
    assert max(cases,key=lambda c:priority(c,'long')) is long
    assert max(cases,key=lambda c:priority(c,'gang')) is gang
    assert max(cases,key=lambda c:priority(c,'queue')) is queue
    with pytest.raises(ValueError):priority(long,'reward')

def test_recorder_keeps_defer_tape_and_selects_measured_placement_states():
    recorder=RepresentativeRecorder(42,'long')
    job=SimpleNamespace(job_id=1,duration_steps=20,num_nodes=2)
    env=SimpleNamespace(_branch_job=job,pending=[job],_pick_hosts_pa=lambda a,j:[a])
    rng_before=np.random.get_state()
    recorder.record('a4',np.ones(3),np.ones(3,bool),10,2,env)
    assert len(recorder.tape)==1 and recorder.selected is None
    recorder.record('a4',np.ones(3),np.ones(3,bool),11,0,env)
    assert recorder.selected['duration']==20
    job.duration_steps=90
    recorder.record('a4',np.ones(3),np.ones(3,bool),12,1,env)
    assert recorder.selected['duration']==90 and recorder.selected['index']==2
    job.duration_steps=10
    recorder.record('a4',np.ones(3),np.ones(3,bool),13,0,env)
    assert recorder.selected['duration']==90 and recorder.eligible==3
    recorder.record('a2',np.ones(3),np.ones(3,bool),14,0,env)
    assert len(recorder.tape)==5
    rng_after=np.random.get_state()
    assert rng_before[0]==rng_after[0] and np.array_equal(rng_before[1],rng_after[1])
    assert rng_before[2:]==rng_after[2:]
