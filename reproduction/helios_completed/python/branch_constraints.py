"""Qualify branch supervision against the experiment's recorded guardrails.

Returns stay unchanged. UNKNOWN means insufficient measurements, not failure
and not success. Callers must bind each case to one state and frozen teacher.
"""
import numpy as np

UNSAFE, UNKNOWN, SAFE = -1, 0, 1
METRICS = ('completed', 'sla', 'power')


def _checked_metrics(values):
    result = {}
    for name in METRICS:
        if name not in values:
            continue
        value = float(values[name])
        if not np.isfinite(value) or value < 0:
            raise ValueError('invalid branch outcome metric')
        result[name] = value
    return result


def capture_outcome(env):
    """Store the same completion/SLA/power measures used at qualification."""
    if not env.done:
        raise ValueError('branch outcome requires a complete replay')
    return _checked_metrics(dict(completed=env.ep_stats['completed'],
        sla=env.sla_stats()['sla_viol'], power=env.ep_stats['rack_viol']))


def qualify_actions(mask, plans, reference_action, reference_metrics, measured_metrics):
    legal = np.asarray(mask, dtype=bool)
    if legal.ndim != 1 or not legal.any():
        raise ValueError('invalid branch mask')
    nr = len(legal)-1
    normalized = {}
    for action, hosts in plans.items():
        a = int(action)
        key = tuple(int(h) for h in hosts)
        if not 0 <= a < nr or not legal[a] or not key or len(set(key)) != len(key) or min(key) < 0:
            raise ValueError('invalid branch plan')
        normalized[a] = key
    if reference_action not in normalized:
        raise ValueError('reference placement is missing')
    reference = _checked_metrics(reference_metrics)
    reference_plan = normalized[reference_action]
    by_plan = {reference_plan: dict(reference)}
    for action, values in measured_metrics.items():
        a = int(action)
        if a not in normalized:
            raise ValueError('measured placement is missing')
        values = _checked_metrics(values)
        known = by_plan.setdefault(normalized[a], {})
        if any(known[k] != values[k] for k in known.keys() & values.keys()):
            raise ValueError('conflicting metrics for identical ordered plan')
        known.update(values)
    qualified = np.full(legal.shape, UNKNOWN, dtype=np.int8)
    reasons = {}
    for action, key in normalized.items():
        if key == reference_plan:
            qualified[action] = SAFE
            reasons[action] = []
            continue
        measured = by_plan.get(key, {})
        missing = [k for k in METRICS if k not in measured or k not in reference]
        worse = [k for k in METRICS if k in measured and k in reference and
                 (measured[k] < reference[k] if k == 'completed' else measured[k] > reference[k])]
        qualified[action] = UNSAFE if worse else UNKNOWN if missing else SAFE
        reasons[action] = worse if worse else ['missing:'+k for k in missing]
    return qualified, reasons


def qualify_case(case):
    """Read both new complete outcomes and partial legacy bank measurements."""
    reference = case.get('reference_outcome')
    if reference is None:
        stats = case['reference_stats']
        reference = dict(completed=stats['completed'], power=stats['rack_viol'])
    measured = {}
    for row in case['branch_rows']:
        measured[int(row['action'])] = row.get('constraint_metrics',
            {k: row[k] for k in METRICS if k in row})
    return qualify_actions(case['mask'], case.get('all_plans', case['plans']),
                           case['reference_action'], reference, measured)
