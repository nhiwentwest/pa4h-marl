"""Fresh Helios run on a versioned completed-only, stratified workload."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import eval_gang as evaluation
import marl_gang_train as trainer
from gang_env import GangEnv
from nrel_injection_bridge import STEPS_PER_HOUR, JOBS_CSV
from pilot_protocol import recipe_digest,validate_resume
from longrun_checkpoint import publish_bundle
from checkpoint_quality import validation_quality,baseline_comparison
from workload_windows import validate_splits
from branch_label_cache import actor_digest,cache_key,load_label,store_label
from a4_branch_rank import replay_branch
from completion_victim import completion_preserving_mask
from grouped_admission import select_grouped_admission
from placement_runtime import install_plan_cache,install_forced_action_shortcut
from qualified_branch_credit import make_branch_label
from representative_branches import DiverseRepresentativeRecorder

parser=argparse.ArgumentParser()
parser.add_argument('--credit',choices=['immediate','horizon'],required=True)
parser.add_argument('--seed',type=int,default=1)
parser.add_argument('--episodes',type=int,default=300)
parser.add_argument('--output',required=True)
parser.add_argument('--protocol',default='scripts/helios_completed_protocol.json')
parser.add_argument('--tag',default='gang')
parser.add_argument('--stop-after',type=int,default=300)
args=parser.parse_args()
root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
protocol=json.loads(Path(args.protocol).read_text())
if (protocol.get('schema')!='helios-queue-relief-fresh300-v1'
    or protocol['total_episodes']!=300 or protocol['batch_episodes']!=4
    or protocol['heldout_hours']!=[1020,1030]
    or protocol['heldout_used'] or not protocol['collect_new_states']):
    raise ValueError('fresh 300-episode protocol differs')
train_windows=protocol['train_windows'];dev_windows=protocol['development_windows']
validate_splits(train_windows,dev_windows,protocol['heldout_hours'],protocol['window_hours'])
if protocol['data']['outcome_policy']!='completed':raise ValueError('invalid primary outcome policy')
if hashlib.sha256(Path(JOBS_CSV).read_bytes()).hexdigest()!=protocol['data']['sha256']:
    raise ValueError('workload checksum differs')
if (os.environ.get('HELIOS_DATASET_REVISION')!='completed-diverse-v1' or
    os.environ.get('HELIOS_DATASET_SHA256')!=protocol['data']['sha256']):
    raise ValueError('checkpoint dataset contract differs')
if args.stop_after%4 or not 4<=args.stop_after<=300:raise ValueError('stop must be a batch boundary')
if protocol['sampler'] != dict(schema='distinct-job-timeband-reservoir-v1',cooldown_batches=8,
    max_cases_per_batch=2,max_plans_per_case=3,
    completion_observable_within_horizon=True,
    reuse_requires='identical prefix state, policy, recipe and SLA multipliers'):
    raise ValueError('sampler differs from implementation')
install_forced_action_shortcut(trainer)
if not trainer.A1_DELAY_CREDIT or not trainer.A1_TEMPORAL_OBSERVATION:
    raise ValueError('long run requires temporal victim observation and delay-aware training credit')
if args.seed != 1 or args.credit != 'immediate' or args.episodes != 300:
    raise ValueError('fresh run requires exactly seed 1 and 300 episodes')
recipe=dict(protocol=protocol,credit=args.credit,seed=args.seed,label_semantics='online-observed-risk-plan-mass-v1')
env_names=('NUM_HOSTS','MAX_JOB_NODES','MAX_STEPS','GANG_MAX_STEPS','BATCH_EPISODES',
    'USE_STGNN','USE_HISTORY_MLP','RACK_HISTORY_LEN','USE_CF_SUPERVISION','LR','PPO_EPOCHS',
    'GAMMA','GAE_LAMBDA','A1_DELAY_CREDIT','A1_TEMPORAL_OBSERVATION','A4_BRANCH_RANK',
    'A1_VIOL_WEIGHT','RELIEF_TEAM_ADV_COEF','A4_TEAM_ADV_COEF','CF_AUX_COEF',
    'W_SERVE','W_ENERGY','W_VIOL','W_WAIT','W_CKPT','W_SLA_BASE','W_DEADLINE',
    'W_NETWORK_DELAY','W_NETWORK_CONGESTION','W_NETWORK_PLACEMENT',
    'SLA_TARGET_ADMISSION','SLA_TARGET_RESTART','SLA_TARGET_COMPLETION','SLA_DUAL_LR',
    'SLA_MAX_WAIT_STEPS','SLA_MAX_RESTART_WAIT_STEPS','SLA_COMPLETION_GRACE_STEPS','SLA_DROP_SEVERITY',
    'NET_COMM_TAX','RACK_OVERSUB','GANG_LINK_MBPS','NETWORK_BACKGROUND_UTIL','FEAS_MARGIN',
    'INTERVAL_SEC','POD_HOURLY_JOBS','NREL_ROOT','HELIOS_DATASET_REVISION','HELIOS_DATASET_SHA256')
effective_env={name:os.environ.get(name) for name in env_names}
base_recipe_sha=recipe_digest(dict(protocol=protocol,environment=effective_env),args.credit,args.seed)
source_files=[p for d in ('python','scripts','target/classes') for p in Path(d).rglob('*')
              if p.is_file() and p.suffix in ('.py','.sh','.class')]
source_sha=hashlib.sha256(b''.join(str(p).encode()+p.read_bytes()
                               for p in sorted(source_files))).hexdigest()
if not trainer.QUEUE_RELIEF_ENABLED or not protocol.get('queue_relief_enabled'):
    raise ValueError('queue relief must be enabled for this versioned fresh run')
recipe['launch_validation'] = 'locked workload checksum, disjoint windows, versioned semantics and source/recipe hashes'
script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
recipe_sha=hashlib.sha256((base_recipe_sha+script_sha).encode()).hexdigest()
recipe.update(script_sha256=script_sha,environment=effective_env,
              initialization='random seed 1; no imported weights or labels')
recipe_path=root/'recipe.json'
if recipe_path.exists():
    previous=json.loads(recipe_path.read_text())
    if previous['recipe_sha256']!=recipe_sha or previous['source_sha256']!=source_sha:
        raise ValueError('existing recipe or source changed')
recipe_path.write_text(json.dumps(dict(recipe,recipe_sha256=recipe_sha,
    source_sha256=source_sha),indent=2)+'\n')
os.environ['GANG_CHECKPOINT_DIR']=str(root/'live');(root/'live').mkdir(exist_ok=True)
np.random.seed(args.seed);torch.manual_seed(args.seed)
port=int(os.environ['BRIDGE_PORT'])
def environment(window,bridge_port=None):
    hour=window['hour']
    start=hour*STEPS_PER_HOUR
    e=GangEnv(port=port if bridge_port is None else bridge_port,subsample=window['subsample'],max_steps=protocol['horizon'],
              min_arrival_step=start,arrival_end_step=start+protocol['window_hours']*STEPS_PER_HOUR)
    for j in e._all_jobs:j.arrival_step-=start
    if (len(e._all_jobs)!=window['jobs'] or
        sum(j.arrival_step+j.duration_steps>e.max_steps for j in e._all_jobs)!=window['earliest_finish_impossible']):
        raise ValueError('injected workload differs from locked window statistics')
    install_plan_cache(e)
    return e
train_envs=[environment(w) for w in train_windows]
dev_envs=[environment(w) for w in dev_windows]
branch_envs=[environment(w,int(os.environ['BRANCH_PORT'])) for w in train_windows]
for e in train_envs:
    original=e.a4_counterfactual_details
    def details(job,e=e,original=original):
        e._branch_job=job
        return original(job)
    e.a4_counterfactual_details=details
ag=trainer.Agents(train_envs[0].num_racks,len(trainer.build_global(train_envs[0])))
init_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
    for actor in ag.actors.values() for v in actor.state_dict().values())).hexdigest()
(root/'initial_actor.sha256').write_text(init_hash+'\n')
training_path=root/'checkpoint'/'training_state.pt'
metadata_path=root/'checkpoint'/'run_metadata.json'
evaluations=[];baselines=[];episode_rows=[];ep=0;best=None;pending_evaluation=False
best_reward=None;best_qualified=None;sampler_ledger={}
if training_path.exists():
    meta=json.loads(metadata_path.read_text())
    validate_resume(meta,recipe_sha,source_sha,meta['episode'])
    state=trainer.load_training_state(str(training_path),ag,train_envs[0]);ep=int(state['episode'])
    validate_resume(meta,recipe_sha,source_sha,ep)
    evaluations=meta['evaluations'];baselines=meta['baselines'];episode_rows=meta['episodes']
    best=meta['best'];pending_evaluation=meta['pending_evaluation']
    best_reward=meta['best_reward'];best_qualified=meta['best_qualified'];sampler_ledger=meta['sampler_ledger']
    print(f'Resumed long run at episode {ep}',flush=True)
else:
    print(json.dumps(dict(initialization='fresh',episode=0,actor_sha256=init_hash,
                          imported_checkpoints=0,imported_branch_labels=0)),flush=True)

def apply_dual(e,leader):
    for name in ('sla_lambda_adm','sla_lambda_restart','sla_lambda_comp'):
        setattr(e,name,getattr(leader,name))

def dev_row(hour,decoder,reward,stats,sla):
    return dict(hour=hour,decoder=decoder,reward=float(reward),
        violations=int(stats['rack_viol']),completed=int(stats['completed']),
        sla=int(sla['sla_viol']),wait=float(stats['wait_cost']),preempt=int(stats['preempted']),
        forced_preempt=int(stats.get('power_constraint_forced_preempt',0)))

def evaluate():
    global best,best_reward,best_qualified,pending_evaluation
    numpy_rng=np.random.get_state();torch_rng=torch.get_rng_state()
    results=[]
    try:
        for window,e in zip(dev_windows,dev_envs):
            hour=window['hour']
            e.sla_lambda_adm,e.sla_lambda_restart,e.sla_lambda_comp=protocol['evaluation_sla_multipliers']
            e.reset();current={};base=trainer.make_deterministic_act(ag,'pointwise')
            def act(name,obs,mask,step,**kwargs):
                if name=='a4':
                    if mask.sum()==1:return int(np.flatnonzero(mask)[0])
                    probs=trainer.actor_probabilities(ag.actors[name],obs,mask)
                    return select_grouped_admission(probs,mask)
                if name=='a2':
                    action=base(name,obs,mask,step,**kwargs);current['rack']=action;return action
                if name=='a3':
                    candidates=e.jobs_on_rack(current['rack'])[:4]
                    preferred=np.asarray(completion_preserving_mask(mask,
                        [e.running_remaining(j) for j in candidates],step,e.max_steps),dtype=bool)
                    return base(name,obs,preferred,step,**kwargs)
                return base(name,obs,mask,step,**kwargs)
            reward=0.
            for step in range(e.max_steps):
                trainer.gang_step(e,ag,step,act);reward+=e.advance()['reward']
                if e.done:break
            row=dev_row(hour,protocol['primary_decoder'],reward,e.ep_stats,e.sla_stats())
            results.append(row)
            print(json.dumps(dict(episode=ep,development=row)),flush=True)
        quality=validation_quality(results,dev_windows,protocol['horizon'],protocol['selection_constraints'])
        comparison=baseline_comparison(results,baselines)
        evaluations.append(dict(episode=ep,results=results,quality=quality,rpa_comparison=comparison))
        candidate=dict(episode=ep,score=quality['score'],qualified=quality['qualified'],
            checkpoint=f'checkpoint_ep{ep:03d}_eval',decoder=protocol['primary_decoder'],
            selection_uses_rpa=False,metric=quality['metric'])
        if best_reward is None or candidate['score']>best_reward['score']:best_reward=candidate
        if quality['qualified'] and (best_qualified is None or candidate['score']>best_qualified['score']):
            best_qualified=candidate
        best=best_qualified if best_qualified is not None else best_reward
        pending_evaluation=False
    finally:
        np.random.set_state(numpy_rng);torch.set_rng_state(torch_rng)
    (root/'development.json').write_text(json.dumps(dict(baselines=baselines,evaluations=evaluations),indent=2)+'\n')
    print(json.dumps(dict(episode=ep,best=best)),flush=True)


def save_checkpoint(phase):
    meta=dict(episode=ep,recipe_sha256=recipe_sha,source_sha256=source_sha,
              evaluations=evaluations,baselines=baselines,episodes=episode_rows,
              best=best,pending_evaluation=pending_evaluation,
              best_reward=best_reward,best_qualified=best_qualified,sampler_ledger=sampler_ledger,
              initialization='fresh seed 1; no checkpoint or static label import')
    def save(path):
        trainer.save_training_state(str(path),ag,train_envs[0],ep,-1e18,-1e18,None,None,0)
    publish_bundle(root,f'checkpoint_ep{ep:03d}_{phase}',save,meta)
    if best:
        pointer=root/'best.next';pointer.unlink(missing_ok=True)
        pointer.symlink_to(best['checkpoint'],target_is_directory=True)
        os.replace(pointer,root/'best')
        (root/'best.json').write_text(json.dumps(best,indent=2)+'\n')
    (root/'selection.json').write_text(json.dumps(dict(best_reward=best_reward,
        best_qualified=best_qualified,selected=best,selection_uses_rpa=False),indent=2)+'\n')
    ag.save(tag='gang');trainer.save_sla_multipliers(train_envs[0])
    (root/'episodes.json').write_text(json.dumps(episode_rows,indent=2)+'\n')

if not baselines:
    for window,e in zip(dev_windows,dev_envs):
        hour=window['hour']
        e.sla_lambda_adm,e.sla_lambda_restart,e.sla_lambda_comp=protocol['evaluation_sla_multipliers']
        r,stats,_=evaluation.run_policy(e,evaluation.RPA())
        baselines.append(dev_row(hour,'RPA',r,stats,stats['sla']))
    (root/'baseline.json').write_text(json.dumps(baselines,indent=2)+'\n')
if not training_path.exists():save_checkpoint('init')
if pending_evaluation:
    evaluate();save_checkpoint('eval')

leader=train_envs[0]
while ep<args.stop_after:
    batch=[];sla_rates=[];recorded=[]
    batch_index=ep//4
    # Two distinct windows that will occur in this batch; rotate the subset.
    appearing=list(dict.fromkeys((ep+i)%len(train_envs) for i in range(4)))
    offset=batch_index%len(appearing)
    wanted={appearing[offset],appearing[(offset+1)%len(appearing)]}
    excluded={job for job,last_batch in sampler_ledger.items()
              if batch_index-last_batch<protocol['sampler']['cooldown_batches']}
    for _ in range(min(4,args.stop_after-ep)):
        index=ep%len(train_envs);e=train_envs[index];apply_dual(e,leader)
        recorder=DiverseRepresentativeRecorder(100000+ep,train_windows[index]['stratum'],
            e.max_steps,(batch_index+index)%3,excluded) if index in wanted and index not in [x[0] for x in recorded] else None
        started=time.time();steps,trans,stats=trainer.rollout(e,ag,recorder=recorder)
        if recorder is not None:recorder.case=recorder.selected
        if recorder is not None and recorder.case is not None:
            recorder.reference_return=sum(trainer.GAMMA**i*x['r'] for i,x in enumerate(steps[recorder.case['step']:]))
            recorder.reference_stats=dict(stats)
            recorded.append((index,recorder,dict((n,getattr(e,n)) for n in ("sla_lambda_adm","sla_lambda_restart","sla_lambda_comp"))))
            excluded.add(recorder.case['job_id'])
            sampler_ledger[recorder.case['job_id']]=batch_index
        batch.append(trainer.collect_episode(ag,steps,trans))
        sla=e.sla_stats();n=max(1,sla['n'])
        sla_rates.append((sla['admission_late']/n,sla['restart_late']/n,
                          (sla['completion_late']+sla['dropped'])/n))
        a4=trans['a4'];multi=[x for x in a4 if np.sum(x['mask'][:-1])>=2]
        informative=int(sum(np.ptp(x['cf_utility'][:-1][x['mask'][:-1]])>1e-6 for x in multi))
        row=dict(episode=ep,hour=train_windows[index]['hour'],reward=sum(s['r'] for s in steps),
            violations=stats['rack_viol'],completed=stats['completed'],sla=sla['sla_viol'],
            wait=stats['wait_cost'],preempt=stats['preempted'],queue_relief_decisions=stats.get('queue_relief_decisions',0),
            queue_relief_preemptions=stats.get('queue_relief_preemptions',0),a4_multirack=len(multi),
            a4_rank_informative=informative,wall_s=time.time()-started)
        row['placement_cache']=dict(e.plan_cache_stats)
        row['a1_temporal_observation']=True;row['a1_delay_credit']=True;row['a1_label_flips']=stats.get('a1_delay_credit_label_flips',0)
        row['sampler']=dict(stratum=train_windows[index]['stratum'],eligible=recorder.eligible,
            selected_job=recorder.case['job_id'] if recorder.case is not None else None,
            selected_step=recorder.case['step'] if recorder.case is not None else None) if recorder else None
        episode_rows.append(row);print(json.dumps(row),flush=True);ep+=1
    # Labels are generated after the PPO trajectories, with unchanged actor weights.
    nrng=np.random.get_state();trng=torch.get_rng_state()
    branch_rows=[];labels=[];policy_sha=actor_digest(ag.actors)
    try:
        for index,recorder,dual in recorded[:2]:
            e=branch_envs[index]
            for n,v in dual.items():setattr(e,n,v)
            case=recorder.case
            cache_id=cache_key(case,recorder.tape,policy_sha,recipe_sha,dual,train_windows[index]['hour'])
            cached=load_label(root/'branch_cache',cache_id)
            if cached is not None:
                labels.append(cached['label'])
                branch_rows.extend([dict(row,episode=ep,cache_reused=True) for row in cached['rows']])
                continue
            if ep==4 and not branch_rows:
                original_action=recorder.tape[case['index']][1]
                control,control_stats=replay_branch(trainer,ag,e,recorder.tape,case,original_action,replay_suffix=True)
                if abs(control-recorder.reference_return)>1e-7:raise RuntimeError('unchanged-action replay reward differs')
                for k in ('rack_viol','completed','wait_cost','preempted'):
                    if control_stats[k]!=recorder.reference_stats[k]:raise RuntimeError('unchanged-action replay metric differs')
                print(json.dumps(dict(prefix_control='passed',return_error=abs(control-recorder.reference_return))),flush=True)
            returns=np.full(case['mask'].shape,np.nan,dtype=np.float64)
            # Compare alternatives with the deployment action under the SAME
            # deterministic continuation, not with the stochastic rollout suffix.
            reference=select_grouped_admission(trainer.actor_probabilities(
                ag.actors['a4'],case['obs'],case['mask']),case['mask'])
            if reference==e.num_racks:
                continue  # This is a routing teacher; it has no DEFER outcome.
            candidates=[reference]
            used={tuple(case['plans'][reference])}
            for action in case['candidates']:
                key=tuple(case['plans'][action])
                if key not in used and len(candidates)<3:
                    candidates.append(action);used.add(key)
            outcomes={};case_rows=[]
            for action in candidates:
                started=time.time()
                value,stats=replay_branch(trainer,ag,e,recorder.tape,case,action)
                returns[action]=value
                outcomes[action]=stats['constraint_metrics']
                result=dict(episode=ep,hour=train_windows[index]['hour'],step=case['step'],
                    job_id=case['job_id'],action=action,discounted_return=value,
                    violations=stats['rack_viol'],completed=stats['completed'],wait=stats['wait_cost'],
                    preempt=stats['preempted'],constraint_metrics=stats['constraint_metrics'],
                    reference_action=reference,wall_s=time.time()-started,stratum=case['stratum'],
                    time_band=case['time_band'],policy_sha256=policy_sha,cache_reused=False)
                branch_rows.append(result);print(json.dumps(dict(branch=result)),flush=True)
                case_rows.append(result)
            label=make_branch_label(case,reference,outcomes,returns)
            labels.append(label);store_label(root/'branch_cache',cache_id,label,case_rows)
    finally:
        np.random.set_state(nrng);torch.set_rng_state(trng)
    (root/f'branches_ep{ep:03d}.json').write_text(json.dumps(branch_rows,indent=2)+'\n')
    labels_path=root/f'labels_ep{ep:03d}.pt'
    labels_temp=labels_path.with_suffix('.tmp')
    torch.save(dict(episode=ep,labels=labels,branch_rows=branch_rows),labels_temp)
    labels_temp.replace(labels_path)
    batch[0]['branch_labels']=labels
    before={n:{k:v.detach().clone() for k,v in a.state_dict().items()} for n,a in ag.actors.items()}
    # Joint PPO update retains trajectories for all four actors.
    trainer.ppo_update_batch(ag,batch)
    changed={n:any(not torch.equal(v,ag.actors[n].state_dict()[k]) for k,v in state.items()) for n,state in before.items()}
    print(json.dumps(dict(episode=ep,actors_changed=changed,branch_states=len(labels))),flush=True)
    if not any(changed.values()):raise RuntimeError('PPO batch changed no actor')
    rates=np.mean(sla_rates,axis=0)
    leader.update_sla_multipliers(admission_rate=float(rates[0]),restart_rate=float(rates[1]),completion_rate=float(rates[2]))
    pending_evaluation=ep in protocol['evaluation_episodes']
    # First make training durable; a restart during evaluation resumes that replay.
    save_checkpoint('train')
    if pending_evaluation:
        evaluate();save_checkpoint('eval')
if ep==args.episodes:
    print(f'Fresh A4 run complete: {ep} new episodes',flush=True)
    (root/'complete.json').write_text(json.dumps(dict(episode=ep,best=best,new_training_episodes=ep,
        initialization='fresh',long_training_started=True),indent=2)+'\n')
else:
    print(json.dumps(dict(bounded_check_complete=ep,total_authorized_episodes=args.episodes)),flush=True)
