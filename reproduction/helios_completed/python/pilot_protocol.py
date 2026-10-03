"""Contracts for the bounded, isolated Helios pilot."""
import hashlib
import json


def validate_protocol(protocol):
    train=protocol['train_hours'];dev=protocol['development_hours']
    width=protocol['window_hours'];heldout=protocol['heldout_hours']
    if width<=0 or protocol['horizon']<=0 or not 0<protocol['subsample']<=1:
        raise ValueError('invalid workload configuration')
    if len(set(train))!=len(train) or len(set(dev))!=len(dev) or not train or not dev:
        raise ValueError('empty or duplicate windows')
    intervals=[(h,h+width,'train') for h in train]+[(h,h+width,'development') for h in dev]
    intervals.append((heldout[0],heldout[1],'holdout'))
    for i,(a,b,_) in enumerate(intervals):
        for c,d,_ in intervals[i+1:]:
            if max(a,c)<min(b,d):raise ValueError('workload windows overlap')
    if protocol['pilot_episodes']%protocol['batch_episodes'] or protocol['batch_episodes']!=4:
        raise ValueError('invalid pilot batch boundary')
    if protocol['long_training_automatic'] or protocol['heldout_used']:
        raise ValueError('pilot may not train indefinitely or consume holdout')


def recipe_digest(protocol,credit,seed):
    return hashlib.sha256(json.dumps(dict(protocol=protocol,credit=credit,seed=seed),
                                     sort_keys=True).encode()).hexdigest()


def validate_resume(metadata,recipe_sha,source_sha,episode):
    if metadata['recipe_sha256']!=recipe_sha or metadata['source_sha256']!=source_sha:
        raise ValueError('resume recipe or source differs')
    if metadata['episode']!=episode or episode%4:
        raise ValueError('resume is not a complete matching PPO batch')
