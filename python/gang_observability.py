"""
gang_observability.py — TensorBoard + Prometheus sink for gang training.

NOTE: reconstructed (2026-08-01) from the call sites in marl_gang_train.py
after the original stayed on the Lightning workspace. Both backends are
optional: without tensorboard/prometheus_client installed (or with
OBS_ENABLED=0) every method degrades to a no-op, so training never fails on a
missing observability dependency.

Interface used by marl_gang_train.py:
    obs = GangObservability(tb_log_dir, prometheus_port=8000)
    obs = GangObservability.disabled()
    obs.log_episode(ep, {metric: value, ...})
    obs.log_agent(ep, agent_name, entropy=..., samples=...)
    obs.log_eval(ep, decoder_name, reward, stats_dict, sla_dict)
    obs.flush(); obs.close()
"""


class GangObservability:
    def __init__(self, log_dir, prometheus_port=None):
        self._writer = None
        self._gauges = {}
        self._prom = False
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir)
        except Exception as exc:                       # tensorboard optional
            print(f"[obs] TensorBoard disabled: {exc}")
        if prometheus_port:
            try:
                from prometheus_client import Gauge, start_http_server
                start_http_server(int(prometheus_port))
                self._Gauge = Gauge
                self._prom = True
                print(f"[obs] Prometheus metrics on :{prometheus_port}")
            except Exception as exc:                   # prometheus optional
                print(f"[obs] Prometheus disabled: {exc}")

    @classmethod
    def disabled(cls):
        o = cls.__new__(cls)
        o._writer = None
        o._gauges = {}
        o._prom = False
        return o

    # ---------------- internals ----------------
    def _scalar(self, tag, value, step):
        if self._writer is not None:
            try:
                self._writer.add_scalar(tag, float(value), int(step))
            except Exception:
                pass
        if self._prom:
            name = "gang_" + tag.replace("/", "_").replace(".", "_")
            g = self._gauges.get(name)
            if g is None:
                try:
                    g = self._Gauge(name, tag)
                    self._gauges[name] = g
                except Exception:
                    return
            try:
                g.set(float(value))
            except Exception:
                pass

    # ---------------- public API ----------------
    def log_episode(self, ep, metrics):
        for key, value in (metrics or {}).items():
            try:
                self._scalar(f"episode/{key}", value, ep)
            except (TypeError, ValueError):
                pass

    def log_agent(self, ep, agent, entropy=None, samples=None, **extra):
        if entropy is not None:
            self._scalar(f"agent/{agent}/entropy", entropy, ep)
        if samples is not None:
            self._scalar(f"agent/{agent}/samples", samples, ep)
        for key, value in extra.items():
            self._scalar(f"agent/{agent}/{key}", value, ep)

    def log_eval(self, ep, decoder, reward, stats=None, sla=None):
        self._scalar(f"eval/{decoder}/reward", reward, ep)
        for src, prefix in ((stats, "stats"), (sla, "sla")):
            for key, value in (src or {}).items():
                try:
                    self._scalar(f"eval/{decoder}/{prefix}_{key}", value, ep)
                except (TypeError, ValueError):
                    pass

    def flush(self):
        if self._writer is not None:
            try:
                self._writer.flush()
            except Exception:
                pass

    def close(self):
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
