"""Bounded coverage sampling, without using reward to choose states."""
from a4_branch_rank import TapeRecorder

def priority(case, stratum):
    if stratum=='long':return (case['duration'],case['queued_demand'],case['nodes'])
    if stratum=='gang':return (case['nodes'],case['duration'],case['queued_demand'])
    if stratum=='queue':return (case['queued_demand'],case['pending'],case['duration'])
    raise ValueError('unknown state stratum')

class RepresentativeRecorder(TapeRecorder):
    def __init__(self,seed,stratum):
        super().__init__(seed)
        self.stratum=stratum;self.selected=None;self.eligible=0

    def record(self,name,obs,mask,step,action,env):
        # A private reservoir of size one visits every eligible state. It does
        # not change NumPy/Torch RNG used by the rollout or deployment policy.
        self.seen=0
        super().record(name,obs,mask,step,action,env)
        if self.case is None or self.case['index']!=len(self.tape)-1:return
        if action==len(mask)-1:return
        job=env._branch_job
        candidate=dict(self.case,duration=int(job.duration_steps),nodes=int(job.num_nodes),
            pending=len(env.pending),queued_demand=sum(int(j.num_nodes) for j in env.pending))
        self.eligible+=1
        if self.selected is None or priority(candidate,self.stratum)>priority(self.selected,self.stratum):
            self.selected=candidate


class DiverseRepresentativeRecorder(TapeRecorder):
    """Uniform reservoir over distinct job/time-band cases in a fixed stratum.

    Cooldown is supplied by the run ledger. Reward never enters selection, and
    the private RNG does not alter PPO rollouts. A missing stratum yields no
    label rather than repeatedly selecting the same maximum-duration job.
    """
    def __init__(self, seed, stratum, horizon, time_band, excluded_job_ids=()):
        super().__init__(seed, min_step=0)
        if stratum not in ('long', 'gang', 'queue') or time_band not in (0, 1, 2):
            raise ValueError('invalid sampler stratum or time band')
        self.stratum=stratum;self.horizon=horizon;self.time_band=time_band
        self.excluded=set(map(str, excluded_job_ids))
        self.selected=None;self.eligible=0;self.unique=set()
        self.reservoirs={};self.counts={}

    def record(self,name,obs,mask,step,action,env):
        self.seen=0
        super().record(name,obs,mask,step,action,env)
        if self.case is None or self.case['index']!=len(self.tape)-1 or action==len(mask)-1:
            return
        job=env._branch_job
        queued=sum(int(j.num_nodes) for j in env.pending)
        if (str(job.job_id) in self.excluded or
            job.duration_steps > self.horizon-int(step) or
            (self.stratum=='long' and job.duration_steps < 12) or
            (self.stratum=='gang' and job.num_nodes < 2) or
            (self.stratum=='queue' and queued < getattr(env,'num_hosts',64))):
            return
        band=min(2, int(step)*3//self.horizon)
        entity=(str(job.job_id),band)
        if entity in self.unique:return
        self.unique.add(entity);self.eligible+=1
        candidate=dict(self.case,duration=int(job.duration_steps),nodes=int(job.num_nodes),
            pending=len(env.pending),queued_demand=queued,time_band=band,stratum=self.stratum)
        self.counts[band]=self.counts.get(band,0)+1
        if self.rng.randint(self.counts[band])==0:self.reservoirs[band]=candidate
        preferred=[self.time_band,(self.time_band+1)%3,(self.time_band+2)%3]
        self.selected=next((self.reservoirs[b] for b in preferred if b in self.reservoirs),None)
