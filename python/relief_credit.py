"""Shared, per-candidate credit for gang-wide power relief."""
from dataclasses import dataclass
import math

import numpy as np


RELIEF_FEATURE_NAMES = (
    "global_violation_relief", "energy_benefit",
    "immediate_cost", "sla_proxy_cost",
)
RELIEF_FEATURE_DIM = len(RELIEF_FEATURE_NAMES)


@dataclass(frozen=True)
class ReliefCredit:
    global_violation_relief: float
    energy_benefit: float
    immediate_cost: float
    sla_proxy_cost: float

    def __post_init__(self):
        if not all(math.isfinite(float(v)) for v in self.__dict__.values()):
            raise ValueError("non-finite relief credit")

    def features(self) -> np.ndarray:
        result = np.asarray(tuple(self.__dict__.values()), dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("relief credit overflows float32 observation")
        return result

    def delta_utility(self, violation_weight: float) -> float:
        result = (float(violation_weight) * self.global_violation_relief
                  + self.energy_benefit - self.immediate_cost - self.sla_proxy_cost)
        if not math.isfinite(result):
            raise ValueError("non-finite relief utility")
        return result

    def a1_utilities(self, violation_weight: float) -> np.ndarray:
        return np.asarray([0.0, self.delta_utility(violation_weight)], dtype=np.float32)
