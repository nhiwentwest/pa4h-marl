"""Absolute validation criteria; baseline comparisons do not select weights."""
import math


def validation_quality(rows, windows, horizon, constraints):
    refs={w['hour']:w for w in windows}
    if len(rows)!=len(refs) or {r['hour'] for r in rows}!=set(refs):
        raise ValueError('development windows missing or duplicated')
    if any(not math.isfinite(float(r[k])) for r in rows
           for k in ('reward','completed','sla','violations','wait','preempt')):
        raise ValueError('non-finite validation metric')
    scores=[];slices=[]
    for row in rows:
        stats=refs[row['hour']]
        completion=row['completed']/max(1,stats['feasible_completion_upper_bound'])
        sla=row['sla']/max(1,stats['jobs'])
        power=row['violations']
        qualified=(completion>=constraints['min_feasible_completion_fraction']
                   and sla<=constraints['max_sla_fraction']
                   and power<=constraints['max_power_violations'])
        slices.append(dict(hour=row['hour'],qualified=qualified,
                           feasible_completion_fraction=completion,sla_fraction=sla,power=power))
        scores.append(row['reward']/horizon)
    return dict(score=sum(scores)/len(scores),qualified=all(x['qualified'] for x in slices),
                metric='mean raw validation reward per simulator step',slices=slices)


def baseline_comparison(rows, baselines):
    refs={r['hour']:r for r in baselines}
    if len(rows)!=len(refs) or {r['hour'] for r in rows}!=set(refs):
        raise ValueError('baseline windows missing or duplicated')
    return [dict(hour=r['hour'],
        reward_delta=r['reward']-refs[r['hour']]['reward'],
        reward_relative_delta=(r['reward']-refs[r['hour']]['reward'])/max(abs(refs[r['hour']]['reward']),1e-12),
        completed_delta=r['completed']-refs[r['hour']]['completed'],
        sla_delta=r['sla']-refs[r['hour']]['sla'],
        power_delta=r['violations']-refs[r['hour']]['violations'],
        wait_delta=r['wait']-refs[r['hour']]['wait'],
        preempt_delta=r['preempt']-refs[r['hour']]['preempt']) for r in rows]
