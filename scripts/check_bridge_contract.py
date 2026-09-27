"""Fail early when the Java bridge lacks methods required by GangEnv."""

import os

from py4j.java_gateway import GatewayParameters, JavaGateway


REQUIRED_METHODS = {
    "reset", "getNumHosts", "getNumRacks", "getRackBudgetW",
    "getGlobalStateDim", "getHostFreeMap", "getRackPowerW",
    "getRackPeakW", "getRackViolCountStep", "getRackFutureTrajectory",
    "submitJob", "submitJobPacked", "preemptJob", "reapFinishedJobs", "stepGang",
    "getEnergyKwh", "getGangCrossRackFraction", "getGangNetworkMetrics", "getJobProgress",
}


def main():
    port = int(os.environ.get("BRIDGE_PORT", "25333"))
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))
    try:
        available = {method.getName() for method in gateway.entry_point.getClass().getMethods()}
        missing = sorted(REQUIRED_METHODS - available)
        if missing:
            raise SystemExit("Gang bridge is incomplete; missing: " + ", ".join(missing))
        print(f"Gang bridge contract OK ({len(REQUIRED_METHODS)} methods)")
    finally:
        gateway.close()


if __name__ == "__main__":
    main()
