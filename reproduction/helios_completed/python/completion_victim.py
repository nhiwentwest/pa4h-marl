"""An isolated A3 decoding candidate; does not change simulator legality.

Prefer a causal victim that does not necessarily lose completion solely due to
the minimum one-step restart delay. Preserve the original choices if every
legal victim has zero completion slack. This is not a guarantee of future
completion or power feasibility; those require full scheduler replay.
"""

def completion_preserving_mask(mask, remaining, step, horizon):
    if not 0 <= step < horizon or len(remaining) > len(mask):
        raise ValueError('invalid state or candidate slots')
    if not any(mask) or any(r < 0 for r in remaining):
        raise ValueError('empty legal actions or negative remaining duration')
    original = [bool(v) for v in mask]
    if any(original[len(remaining):]):
        raise ValueError('legal slot lacks a candidate')
    preferred = original.copy()
    for i, duration in enumerate(remaining):
        earliest_finish = step + duration
        if duration > 0 and earliest_finish <= horizon < earliest_finish + 1:
            preferred[i] = False
    return preferred if any(preferred) else original
