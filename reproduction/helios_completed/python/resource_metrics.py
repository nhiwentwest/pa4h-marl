"""
resource_metrics.py — wall-clock / CPU / RSS metering for training and eval.

NOTE: reconstructed (2026-08-01) from the call sites in marl_gang_train.py and
eval_gang.py after the original stayed on the Lightning workspace. Interface:

    meter = ResourceMeter.for_bridge_port(port)   # python proc + java bridge
    meter.sample()                                # call periodically
    d = meter.finish()   # {wall_time_s, cpu_time_s, avg_cpu_pct,
                         #  peak_rss_mb, monitored_pids}
"""
import os
import time

try:
    import psutil
except ImportError:                      # metering degrades gracefully
    psutil = None


class ResourceMeter:
    """Aggregates resource usage of the Python controller plus the Java
    CloudSim bridge process bound to a given Py4J port."""

    def __init__(self, pids=None):
        self._start_wall = time.time()
        self._procs = []
        self._peak_rss = 0.0
        self._cpu_samples = []
        self._start_cpu = 0.0
        if psutil is None:
            return
        pids = set(pids or []) | {os.getpid()}
        for pid in pids:
            try:
                p = psutil.Process(pid)
                p.cpu_percent(interval=None)          # prime the counter
                self._procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self._start_cpu = self._cpu_time_total()
        self.sample()

    @classmethod
    def for_bridge_port(cls, port):
        """Meter this process and the JVM listening on ``port`` (if found)."""
        pids = []
        if psutil is not None:
            try:
                for conn in psutil.net_connections(kind="tcp"):
                    if (conn.laddr and conn.laddr.port == int(port)
                            and conn.status == psutil.CONN_LISTEN
                            and conn.pid):
                        pids.append(conn.pid)
            except (psutil.AccessDenied, PermissionError):
                # macOS denies net_connections without root: fall back to
                # matching java processes by command line.
                for p in psutil.process_iter(["name", "cmdline"]):
                    try:
                        if (p.info["name"] or "").startswith("java") and any(
                                "Py4jBridge" in c for c in (p.info["cmdline"] or [])):
                            pids.append(p.pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        return cls(pids)

    def _cpu_time_total(self):
        total = 0.0
        for p in self._procs:
            try:
                t = p.cpu_times()
                total += t.user + t.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    def sample(self):
        """Record an instantaneous CPU%/RSS sample; call once per episode."""
        if psutil is None:
            return
        rss = 0.0
        cpu = 0.0
        for p in self._procs:
            try:
                rss += p.memory_info().rss
                cpu += p.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self._peak_rss = max(self._peak_rss, rss)
        self._cpu_samples.append(cpu)

    def finish(self):
        self.sample()
        wall = time.time() - self._start_wall
        cpu_time = self._cpu_time_total() - self._start_cpu if psutil else 0.0
        samples = [s for s in self._cpu_samples if s > 0.0]
        avg_cpu = (sum(samples) / len(samples)) if samples else 0.0
        return {
            "wall_time_s": wall,
            "cpu_time_s": cpu_time,
            "avg_cpu_pct": avg_cpu,
            "peak_rss_mb": self._peak_rss / (1024.0 * 1024.0),
            "monitored_pids": len(self._procs),
        }
