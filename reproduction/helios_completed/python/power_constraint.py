"""Hard next-step power feasibility, separate from the scheduler's soft reward.

This is an explicit action constraint, not a learned improvement in A1. The
physical simulator, telemetry, reward coefficients and historical recipes stay
unchanged. If the existing relief path cannot satisfy the constraint, halt before
advancing rather than silently counting an infeasible replay as a success.
"""
import math
import numpy as np


class PowerConstraintError(RuntimeError):
    pass


def power_action_mask(mask, before, after, gap_eps=1e-6):
    original=np.asarray(mask,dtype=bool)
    if original.shape!=(2,) or not original.any():
        raise ValueError('A1 needs a nonempty WAIT/PREEMPT mask')
    if not all(math.isfinite(v) and 0<=v<=1 for v in (before,after)):
        raise ValueError('invalid projected violation fraction')
    if before<=gap_eps:
        return original.copy()
    if after>=before-gap_eps or not original[1]:
        raise PowerConstraintError('risk remains without a legal causal preemption')
    return np.asarray([False,True],dtype=bool)


def require_power_feasible(env, gap_eps=1e-6):
    risks=[float(env.project_rack_violation_fraction(r)) for r in range(env.num_racks)]
    bad=[r for r,v in enumerate(risks) if not math.isfinite(v) or v<0 or v>gap_eps]
    if bad:
        raise PowerConstraintError(f'next-step power infeasible on racks {bad}; risks={risks}')
