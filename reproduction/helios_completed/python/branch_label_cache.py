"""Reuse only outcomes from an identical state AND frozen continuation policy."""
import hashlib
import json
from pathlib import Path
import torch


def actor_digest(actors):
    digest=hashlib.sha256()
    for name,actor in sorted(actors.items()):
        digest.update(name.encode())
        for key,value in sorted(actor.state_dict().items()):
            digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def cache_key(case, tape, policy_sha, recipe_sha, dual, hour):
    payload=dict(recipe_sha=recipe_sha,policy_sha=policy_sha,hour=hour,dual=dual,
        job_id=case['job_id'],step=case['step'],index=case['index'],
        obs=case['obs'].tolist(),mask=case['mask'].tolist(),
        plans=case['plans'],candidates=case['candidates'],prefix=tape[:case['index']+1])
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def load_label(directory, key):
    path=Path(directory)/f'{key}.pt'
    if not path.exists():return None
    data=torch.load(path,map_location='cpu',weights_only=False)
    if data.get('schema')!='frozen-branch-cache-v1' or data.get('key')!=key:
        raise ValueError('branch cache identity differs')
    return data


def store_label(directory, key, label, rows):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    path=root/f'{key}.pt';temporary=path.with_suffix('.tmp')
    torch.save(dict(schema='frozen-branch-cache-v1',key=key,label=label,rows=rows),temporary)
    temporary.replace(path)
