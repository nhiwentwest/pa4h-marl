"""Return labels for anchor aliases, without changing the anchor policy."""
import numpy as np


def plan_key(hosts):
    """Preserve ordered hosts: ring permutations are not assumed equivalent."""
    key=tuple(int(h) for h in hosts)
    if not key or len(set(key))!=len(key) or any(h<0 for h in key):
        raise ValueError('placement requires distinct nonnegative hosts')
    return key


def expand_alias_returns(returns,mask,plans,tolerance=1e-8):
    """Share measured labels only across exact plans at the same frozen state.

    Caller must bind plans and returns to the same state and teacher fingerprint.
    A missing plan or different ring ordering never receives an inferred return.
    Existing finite labels are preserved; the input array is never modified.
    """
    values=np.asarray(returns,dtype=np.float64)
    legal=np.asarray(mask,dtype=bool)
    if values.ndim!=1 or values.shape!=legal.shape or not len(values) or not legal.any():
        raise ValueError('return labels and legality mask must match')
    measured=np.isfinite(values)
    if np.isinf(values).any() or np.any(measured & ~legal) or measured[-1]:
        raise ValueError('illegal, infinite or DEFER routing label')
    groups={};mapped={}
    for action,hosts in plans.items():
        a=int(action)
        if not 0<=a<len(values)-1 or not legal[a]:
            raise ValueError('plan describes an illegal routing action')
        key=plan_key(hosts);mapped[a]=key;groups.setdefault(key,[]).append(a)
    if any(int(a) not in mapped for a in np.flatnonzero(measured)):
        raise ValueError('measured action has no recorded plan')
    result=values.copy()
    for actions in groups.values():
        labels=[values[a] for a in actions if measured[a]]
        if not labels:continue
        if max(labels)-min(labels)>tolerance:
            raise ValueError('conflicting returns for identical ordered plan')
        for a in actions:
            if not measured[a]:result[a]=labels[0]
    return result
