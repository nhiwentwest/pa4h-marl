"""Decode admission mass first, then conditional rack probability.

This is MAP for the grouped admission decision followed by conditional MAP for
the rack. It intentionally differs from joint-action argmax and does not claim
to be a return-maximizing oracle. It honors the original feasibility mask and
uses no future arrivals or changes to actor parameters.
"""
import math

def select_grouped_admission(probabilities, mask):
    if len(probabilities) != len(mask) or len(mask) < 2 or not any(mask):
        raise ValueError('invalid action probabilities or mask')
    legal = [i for i, enabled in enumerate(mask) if enabled]
    if any(not math.isfinite(float(probabilities[i])) or probabilities[i] < 0
           for i in legal):
        raise ValueError('legal action probabilities must be finite and nonnegative')
    if math.fsum(float(probabilities[i]) for i in legal) <= 0:
        raise ValueError('zero probability on legal actions')
    defer = len(mask) - 1
    racks = [i for i in legal if i != defer]
    admit_mass = math.fsum(float(probabilities[i]) for i in racks)
    defer_mass = float(probabilities[defer]) if mask[defer] else 0.
    if not racks or admit_mass < defer_mass:
        return defer
    return max(racks, key=lambda i: float(probabilities[i]))
