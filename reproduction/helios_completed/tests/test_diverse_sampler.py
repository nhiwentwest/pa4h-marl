from types import SimpleNamespace
import numpy as np
from representative_branches import DiverseRepresentativeRecorder


def environment(job):
    return SimpleNamespace(_branch_job=job,pending=[job],num_hosts=2,
                           _pick_hosts_pa=lambda anchor,job:[anchor])


def test_cooldown_excludes_repeated_jobs_and_band_selects_early_states():
    job=SimpleNamespace(job_id='old',duration_steps=90,num_nodes=2)
    env=environment(job)
    recorder=DiverseRepresentativeRecorder(1,'long',180,0,['old'])
    recorder.record('a4',np.ones(3),np.ones(3,bool),0,0,env)
    assert recorder.selected is None and len(recorder.tape)==1
    job.job_id='new'
    recorder.record('a4',np.ones(3),np.ones(3,bool),2,0,env)
    assert recorder.selected['step']==2 and recorder.selected['time_band']==0
    recorder.record('a4',np.ones(3),np.ones(3,bool),50,0,env)
    assert recorder.eligible==1  # repeated same job in same band
    job.duration_steps=20
    recorder.record('a4',np.ones(3),np.ones(3,bool),150,0,env)
    assert recorder.selected['step']==2 and recorder.eligible==2


def test_strata_and_private_rng_keep_short_single_node_out_of_gang_labels():
    job=SimpleNamespace(job_id='a',duration_steps=1,num_nodes=1)
    env=environment(job);recorder=DiverseRepresentativeRecorder(42,'gang',180,2)
    before=np.random.get_state()
    recorder.record('a4',np.ones(3),np.ones(3,bool),10,0,env)
    assert recorder.selected is None
    job.num_nodes=2
    recorder.record('a4',np.ones(3),np.ones(3,bool),20,2,env)  # DEFER remains on tape
    assert recorder.selected is None
    recorder.record('a4',np.ones(3),np.ones(3,bool),30,0,env)
    after=np.random.get_state()
    assert recorder.selected['nodes']==2
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]


def test_collector_avoids_jobs_whose_completion_cannot_be_observed():
    job=SimpleNamespace(job_id='tail',duration_steps=180,num_nodes=2)
    env=environment(job);recorder=DiverseRepresentativeRecorder(42,'long',180,0)
    recorder.record('a4',np.ones(3),np.ones(3,bool),10,0,env)
    assert recorder.selected is None
    job.duration_steps=20
    recorder.record('a4',np.ones(3),np.ones(3,bool),11,0,env)
    assert recorder.selected['duration']==20
