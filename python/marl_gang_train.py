"""
marl_gang_train.py — 4-agent CTDE-PPO for the GPU-exclusive rack-power env (hướng A).

Clean-room training loop over GangEnv (NOT the legacy VM-migration loop). The four
agents form a placement/relief hierarchy; they share a cooperative TEAM reward (the
Java rack-budget objective: −energy − 1000·SLAV − 2·rackViol − 0.5·checkpoint) and a
single centralized critic over global rack state (MAPPO / CTDE, Yu et al. 2022).

  A4 placer   : per pending job, pick an anchor rack (or DEFER)          [NR+1]
  A2 detector : flag the rack most at risk of a power-budget violation   [NR+1]
  A1 trigger  : on the flagged rack, preempt-to-relieve vs. wait         [2]
  A3 victim   : which running job on that rack to checkpoint-preempt     [RACK_SIZE]

No Autoformer, no STGNN, no forecasting: risk is read from the deterministic NREL
future-power trajectory the bridge already exposes. Single-env for correctness;
scale later.
"""
import os
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F

from gang_env import (GangEnv, RACK_OBS_DIM, JOB_OBS_DIM, NETWORK_OBS_DIM,
                      RACK_HISTORY_LEN, RACK_SIZE, SIMULATOR_SEMANTICS,
                      INTERVAL_SEC,
                      SLA_TARGET_ADMISSION, SLA_TARGET_RESTART, SLA_TARGET_COMPLETION,
                      NET_COMM_TAX, W_CKPT, W_ENERGY, W_SERVE, W_VIOL, W_WAIT,
                      W_SLA_BASE)
from models import Actor, CentralizedCritic, RackSTGNNActor, select_action
from resource_metrics import ResourceMeter
from gang_observability import GangObservability
from nrel_injection_bridge import STEPS_PER_HOUR
from relief_credit import ReliefCredit, RELIEF_FEATURE_NAMES, RELIEF_FEATURE_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = float(os.environ.get("GAMMA", 0.99))
LAM = float(os.environ.get("GAE_LAMBDA", 0.95))
CLIP = 0.2
LR = float(os.environ.get("LR", 3e-4))
EPOCHS = int(os.environ.get("PPO_EPOCHS", 4))
ENT_COEF = float(os.environ.get("ENT_COEF", 0.01))
ENT_FLOOR_PENALTY = float(os.environ.get("ENT_FLOOR_PENALTY", 0.10))
ENTROPY_TARGET = dict(a1=0.20, a2=0.50, a3=0.20, a4=0.15)
ENT_ADAPT_LR = float(os.environ.get("ENT_ADAPT_LR", 0.10))
ENT_COEF_MAX = float(os.environ.get("ENT_COEF_MAX", 1.0))
RELIEF_SLA_WEIGHT = float(os.environ.get("RELIEF_SLA_WEIGHT", 0.10))
# Team reward is already normalized to ~O(1)/step in GangEnv, so no down-scaling
# (a small scale here would collapse advantages and kill learning).
REWARD_SCALE = float(os.environ.get("REWARD_SCALE", 1.0))
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", 60))
# Episodes pooled into ONE PPO update. >1 is the variance fix for this single-env,
# high-variance gang problem: the gradient reflects the average episode, so the
# policy stops thrashing between good/bad rollouts. 1 == the old per-episode update.
BATCH_EPISODES = int(os.environ.get("BATCH_EPISODES", 4))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 120))
SUBSAMPLE = float(os.environ.get("SUBSAMPLE", 0.05))
SEED = int(os.environ.get("SEED", 0))
USE_STGNN = os.environ.get("USE_STGNN", "0").strip().lower() in ("1", "true", "yes")
USE_HISTORY_MLP = os.environ.get("USE_HISTORY_MLP", "0").strip().lower() in ("1", "true", "yes")
USE_CF_SUPERVISION = os.environ.get("USE_CF_SUPERVISION", "1").strip().lower() in ("1", "true", "yes")

# --- deployable A4 credit fixes ---
A4_ENT_COEF = float(os.environ.get("A4_ENT_COEF", ENT_COEF))
# A4 gets an exact, per-placement counterfactual advantage. Keep a small team
# term so placement still reflects downstream preemption/SLA consequences.
A4_TEAM_ADV_COEF = float(os.environ.get("A4_TEAM_ADV_COEF", 0.15))
RELIEF_TEAM_ADV_COEF = float(os.environ.get("RELIEF_TEAM_ADV_COEF", 0.05))
CF_AUX_COEF = float(os.environ.get("CF_AUX_COEF", 1.0))
A4_FEAS_AUX_COEF = float(os.environ.get("A4_FEAS_AUX_COEF", 0.25))
CF_GAP_EPS = float(os.environ.get("CF_GAP_EPS", 1e-6))
# #1: stop once argmax-eval stalls (0 = off; keep off to capture the full curve).
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", 0))
DEPLOY_MIN_SERVED_RATE = float(os.environ.get("DEPLOY_MIN_SERVED_RATE", 0.60))
RESUME = os.environ.get("RESUME", "0").strip().lower() in ("1", "true", "yes")
WORKLOAD_START_HOUR = os.environ.get("WORKLOAD_START_HOUR", "").strip()
WORKLOAD_WINDOW_HOURS = os.environ.get("WORKLOAD_WINDOW_HOURS", "").strip()
VALIDATION_START_HOUR = os.environ.get("VALIDATION_START_HOUR", "").strip()
VALIDATION_WINDOW_HOURS = os.environ.get("VALIDATION_WINDOW_HOURS", "").strip()
A1_VIOL_WEIGHT = float(os.environ.get("A1_VIOL_WEIGHT", 3.0))
A1_OBS_DIM = RACK_OBS_DIM + 5 + RELIEF_FEATURE_DIM
A3_SLOT_DIM = JOB_OBS_DIM + 2 + RELIEF_FEATURE_DIM
A3_OBS_DIM = RACK_OBS_DIM + RACK_SIZE * A3_SLOT_DIM


def relief_credit_metadata():
    return dict(schema="global-relief-v1", features=list(RELIEF_FEATURE_NAMES),
                a1_obs_dim=A1_OBS_DIM, a3_slot_dim=A3_SLOT_DIM,
                a3_obs_dim=A3_OBS_DIM, violation_weight=A1_VIOL_WEIGHT)


def build_global(env):
    """Critic global state: rack features + queue/time scalars."""
    rf = env.rack_features().reshape(-1)
    scal = np.array([len(env.pending) / 50.0,
                     len(env.running) / max(1, env.num_hosts),
                     env.step_idx / max(1, env.max_steps)], dtype=np.float32)
    return np.concatenate([rf, scal, env.sla_global_features(),
                           env.network_features()]).astype(np.float32)


class Agents:
    """Holds the 4 actors + centralized critic and their PPO optimizers."""

    def __init__(self, num_racks, global_dim):
        NR = num_racks
        self.use_stgnn = USE_STGNN
        self.use_history_mlp = USE_HISTORY_MLP
        if self.use_stgnn and self.use_history_mlp:
            raise ValueError("USE_STGNN and USE_HISTORY_MLP are mutually exclusive")
        graph_history_dim = NR * RACK_HISTORY_LEN * RACK_OBS_DIM
        self.dims = dict(
            # Each actor observes the counterfactual quantity that defines its own
            # target.  Previously those labels were partly hidden, making the
            # policy impossible to identify at inference even with perfect credit.
            a4=((graph_history_dim if (self.use_stgnn or self.use_history_mlp)
                 else NR * RACK_OBS_DIM)
                + JOB_OBS_DIM + NETWORK_OBS_DIM + NR + 1, NR + 1),
            a2=((graph_history_dim if (self.use_stgnn or self.use_history_mlp)
                 else NR * RACK_OBS_DIM) + NR, NR),
            a1=(A1_OBS_DIM, 2),
            a3=(A3_OBS_DIM, RACK_SIZE),
        )
        self.actors = {
            "a1": Actor(*self.dims["a1"]).to(DEVICE),
            "a3": Actor(*self.dims["a3"]).to(DEVICE),
        }
        if self.use_stgnn:
            self.actors["a2"] = RackSTGNNActor(
                NR, RACK_HISTORY_LEN, RACK_OBS_DIM, context_dim=NR,
                include_defer=False).to(DEVICE)
            self.actors["a4"] = RackSTGNNActor(
                NR, RACK_HISTORY_LEN, RACK_OBS_DIM,
                context_dim=JOB_OBS_DIM + NETWORK_OBS_DIM + NR + 1,
                include_defer=True).to(DEVICE)
        else:
            self.actors["a2"] = Actor(*self.dims["a2"]).to(DEVICE)
            self.actors["a4"] = Actor(*self.dims["a4"]).to(DEVICE)
        self.critic = CentralizedCritic(global_dim).to(DEVICE)
        self.opt = {k: torch.optim.Adam(a.parameters(), lr=LR) for k, a in self.actors.items()}
        self.opt["critic"] = torch.optim.Adam(self.critic.parameters(), lr=LR)
        self.ent_coef = {k: (A4_ENT_COEF if k == "a4" else ENT_COEF)
                         for k in self.actors}

    def value(self, g):
        with torch.no_grad():
            return float(self.critic(torch.as_tensor(g, device=DEVICE).unsqueeze(0)).item())

    def save(self, tag="gang"):
        out_dir = os.environ.get("GANG_CHECKPOINT_DIR", ".")
        os.makedirs(out_dir, exist_ok=True)
        for k, a in self.actors.items():
            assert_finite_model(a, k)
            torch.save(a.state_dict(), os.path.join(out_dir, f"{k}_{tag}.pt"))
        assert_finite_model(self.critic, "critic")
        torch.save(self.critic.state_dict(), os.path.join(out_dir, f"critic_{tag}.pt"))


def assert_finite_model(model, name):
    if any(not torch.isfinite(p).all().item() for p in model.state_dict().values()
           if torch.is_tensor(p)):
        raise FloatingPointError(f"non-finite {name} checkpoint tensor")


def _optimizer_to_device(optimizer, device=DEVICE):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_training_state(path, ag, env, episode, best_batch, best_eval,
                        best_deploy_key, best_decoder, stale):
    """Atomically save everything required to continue at the next PPO batch."""
    state = {
        "schema_version": 2,
        "semantic_version": "gang-ctde-sla-v2-a4-hybrid",
        "simulator_semantics": SIMULATOR_SEMANTICS,
        "relief_credit": relief_credit_metadata(),
        "episode": int(episode),
        "actors": {k: v.state_dict() for k, v in ag.actors.items()},
        "critic": ag.critic.state_dict(),
        "optimizers": {k: v.state_dict() for k, v in ag.opt.items()},
        "entropy_coefficients": dict(ag.ent_coef),
        "use_stgnn": bool(ag.use_stgnn),
        "use_history_mlp": bool(ag.use_history_mlp),
        "best_batch": float(best_batch),
        "best_eval": float(best_eval),
        "best_deploy_key": best_deploy_key,
        "best_decoder": best_decoder,
        "stale": int(stale),
        "sla": {
            "lambda_admission": float(env.sla_lambda_adm),
            "lambda_restart": float(getattr(env, "sla_lambda_restart", 1.0)),
            "lambda_completion": float(env.sla_lambda_comp),
            "rate_ema": np.pad(np.asarray(env._sla_rate_ema, dtype=np.float64),
                               (0, max(0, 3 - len(env._sla_rate_ema))))[:3],
            "dual_updates": int(env._sla_dual_updates),
            "ema_initialized": bool(env._sla_ema_initialized),
        },
        "rng": {
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    for name, actor in ag.actors.items():
        assert_finite_model(actor, name)
    assert_finite_model(ag.critic, "critic")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_training_state(path, ag, env):
    """Restore a trusted local training-state checkpoint."""
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    if state.get("schema_version") != 2:
        raise ValueError("v1 semantic checkpoints are baseline-only; fresh v2 training state required")
    if state.get("simulator_semantics") != SIMULATOR_SEMANTICS:
        raise ValueError("checkpoint simulator semantics differ; use old actor weights for diagnostic replay only")
    if state.get("relief_credit") != relief_credit_metadata():
        raise ValueError("checkpoint relief_credit descriptor differs; fresh v5 training required")
    if bool(state.get("use_stgnn")) != bool(ag.use_stgnn):
        raise ValueError("checkpoint architecture does not match USE_STGNN")
    if bool(state.get("use_history_mlp", False)) != bool(ag.use_history_mlp):
        raise ValueError("checkpoint architecture does not match USE_HISTORY_MLP")
    if set(state["actors"]) != set(ag.actors):
        raise ValueError("checkpoint actor set does not match current trainer")
    for name, actor in ag.actors.items():
        actor.load_state_dict(state["actors"][name])
    ag.critic.load_state_dict(state["critic"])
    for name, optimizer in ag.opt.items():
        optimizer.load_state_dict(state["optimizers"][name])
        _optimizer_to_device(optimizer)
    ag.ent_coef = {k: float(v) for k, v in state["entropy_coefficients"].items()}
    sla = state["sla"]
    env.sla_lambda_adm = float(sla["lambda_admission"])
    env.sla_lambda_restart = float(sla["lambda_restart"])
    env.sla_lambda_comp = float(sla["lambda_completion"])
    env._sla_rate_ema = np.asarray(sla["rate_ema"], dtype=np.float64)
    env._sla_dual_updates = int(sla["dual_updates"])
    env._sla_ema_initialized = bool(sla["ema_initialized"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"].cpu())
    if torch.cuda.is_available() and state["rng"].get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return state


def weighted_sla_penalty(risk_cost, priority_total, multiplier):
    """Local SLA credit in the same base-plus-dual form as the team reward."""
    return ((W_SLA_BASE + max(0.0, float(multiplier))) * 3.0
            * max(0.0, float(risk_cost)) / max(1.0, float(priority_total)))


def gae(rewards, values, dones, last_v=0.0):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae_run = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        nv = last_v if t == T - 1 else values[t + 1]
        delta = rewards[t] + GAMMA * nv * nonterm - values[t]
        gae_run = delta + GAMMA * LAM * nonterm * gae_run
        adv[t] = gae_run
    ret = adv + np.array(values, dtype=np.float32)
    return adv, ret


def counterfactual_advantage(utilities, probs, action):
    """A(action) = U(action) - E_pi[U(action)] over legal A4 actions.

    ``utilities`` comes from an exact NREL replay of each placement option, not
    a hand-tuned reward coefficient. Invalid actions are ignored even if a
    caller accidentally gives them non-zero probability.
    """
    u = np.asarray(utilities, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    legal = np.isfinite(u)
    if action < 0 or action >= len(u) or not legal[action]:
        raise ValueError("chosen A4 action must be legal")
    p = np.where(legal, np.maximum(p, 0.0), 0.0)
    z = p.sum()
    if z <= 0.0:
        p = legal.astype(np.float64) / legal.sum()
    else:
        p /= z
    return float(u[action] - np.dot(p[legal], u[legal]))


def counterfactual_observation(utilities, mask):
    """Finite, scale-stable copy of a full action-utility vector for actor input."""
    u = np.asarray(utilities, dtype=np.float32)
    legal = np.asarray(mask, dtype=bool) & np.isfinite(u)
    out = np.zeros_like(u, dtype=np.float32)
    if not legal.any():
        return out
    vals = u[legal]
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo > CF_GAP_EPS:
        out[legal] = 2.0 * (vals - lo) / (hi - lo) - 1.0
    return out


def counterfactual_policy_loss(logits, utilities, masks, gap_eps=CF_GAP_EPS):
    """Full-information policy-improvement loss.

    PPO only sees the sampled action, so a collapsed policy may never sample the
    better counterfactual again.  This loss uses all legal utilities already
    computed by the simulator.  Ties and forced rows intentionally contribute no
    gradient; they are not genuine actor decisions.
    """
    legal = masks.bool() & torch.isfinite(utilities)
    neg_inf = torch.full_like(utilities, -torch.inf)
    pos_inf = torch.full_like(utilities, torch.inf)
    best_u = torch.where(legal, utilities, neg_inf).max(dim=1).values
    worst_u = torch.where(legal, utilities, pos_inf).min(dim=1).values
    informative = (legal.sum(dim=1) > 1) & ((best_u - worst_u) > gap_eps)
    if not bool(informative.any()):
        return logits.sum() * 0.0, 0
    # Share target mass between numerically tied best actions instead of choosing
    # an arbitrary rack/victim.
    target_mask = legal & (utilities >= (best_u.unsqueeze(1) - gap_eps))
    target = target_mask.float() / target_mask.sum(dim=1, keepdim=True).clamp_min(1)
    logp = F.log_softmax(logits, dim=-1)
    # Avoid 0 * -inf on illegal actions (which poisoned v1 updates with NaN).
    row_loss = -torch.where(target_mask, target * logp, torch.zeros_like(logp)).sum(dim=1)
    return row_loss[informative].mean(), int(informative.sum().item())


def _zscore(values):
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean()) / (values.std() + 1e-8)


def masked_categorical_entropy(logits, masks=None):
    """Categorical entropy without multiplying masked ``-inf`` logits by zero."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    valid = torch.isfinite(log_probs) if masks is None else masks.bool()
    safe_log_probs = torch.where(valid, log_probs, torch.zeros_like(log_probs))
    return -(probs * safe_log_probs).sum(dim=-1)


def combine_local_team_advantages(local, team, team_coef):
    """Normalize local and delayed team credit separately before combining them."""
    return _zscore(local) + float(team_coef) * _zscore(team)


class DeterministicMarginalDecoder:
    """Online dependent rounding for a sequence of categorical decisions.

    Pointwise argmax destroys a learned mixed placement policy by sending every
    near-symmetric job to the same rack.  This decoder is deterministic, but it
    rounds the cumulative policy mass rather than each decision independently.
    Therefore a uniform five-rack policy becomes 0,1,2,3,4,... instead of
    0,0,0,0,0,... .  No RNG or seed is involved.
    """

    def __init__(self, action_dim):
        self.action_dim = int(action_dim)
        self.expected = np.zeros(self.action_dim, dtype=np.float64)
        self.realized = np.zeros(self.action_dim, dtype=np.float64)

    def select(self, probabilities, mask, weight=1.0):
        p = np.asarray(probabilities, dtype=np.float64)
        legal = np.asarray(mask, dtype=bool)
        if p.shape != (self.action_dim,) or legal.shape != (self.action_dim,):
            raise ValueError("probabilities/mask must match decoder action_dim")
        p = np.where(legal & np.isfinite(p), np.maximum(p, 0.0), 0.0)
        total = float(p.sum())
        if total <= 0.0:
            p = legal.astype(np.float64)
            total = float(p.sum())
        if total <= 0.0:
            raise ValueError("decoder requires at least one legal action")
        p /= total
        w = max(1e-6, float(weight))
        self.expected += w * p
        deficit = np.where(legal, self.expected - self.realized, -np.inf)
        action = int(np.argmax(deficit))
        self.realized[action] += w
        return action


def actor_probabilities(actor, obs, mask):
    with torch.no_grad():
        ob = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mk = torch.as_tensor(mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
        return torch.softmax(actor(ob, mk), dim=-1)[0].cpu().numpy()


def make_deterministic_act(ag, a4_mode="sequence"):
    """Build one episode-scoped deterministic actor callback.

    ``sequence`` is the deploy decoder; ``pointwise`` is retained only as an
    audit baseline to expose local-argmax mode collapse.
    """
    if a4_mode not in ("sequence", "pointwise"):
        raise ValueError("a4_mode must be 'sequence' or 'pointwise'")
    decoder = DeterministicMarginalDecoder(ag.dims["a4"][1])

    def act(name, obs, mask, step, decode_weight=1.0, **_kwargs):
        mask = mask.astype(bool) if mask is not None else np.ones(
            ag.dims[name][1], dtype=bool)
        if name == "a4" and a4_mode == "sequence":
            probs = actor_probabilities(ag.actors[name], obs.astype(np.float32), mask)
            return decoder.select(probs, mask, weight=decode_weight)
        action, _ = select_action(ag.actors[name], obs.astype(np.float32), mask,
                                  mode="eval")
        return int(action)

    return act


def chosen_projected_peaks(projected, action):
    """Return the rack projection for a placement, or an explicit NA vector for DEFER."""
    p = np.asarray(projected, dtype=np.float32)
    if 0 <= action < len(p):
        return p[action].copy()
    return np.full(p.shape[1], np.nan, dtype=np.float32)


def relief_action_utilities(before_violation, after_violation, preempt_cost,
                            sla_weight=RELIEF_SLA_WEIGHT, num_racks=1,
                            extra_preempt_cost=0.0, energy_benefit=0.0,
                            sla_penalty=None):
    """Legacy rack-local diagnostic; v5 production uses ReliefCredit.

    ``extra_preempt_cost`` is the exact immediate service + checkpoint + queue
    cost. ``energy_benefit`` is the normalized energy saving.  The optional SLA
    penalty is derived from the current multiplier and job priority; the fallback
    keeps this helper useful in bridge-free unit tests.
    """
    before = np.clip(float(before_violation), 0.0, 1.0)
    after = np.clip(float(after_violation), 0.0, 1.0)
    cost = max(0.0, float(preempt_cost))
    if sla_penalty is None:
        sla_penalty = sla_weight * cost / (1.0 + cost)
    wait_u = -A1_VIOL_WEIGHT * before / max(1, int(num_racks))
    preempt_u = (-A1_VIOL_WEIGHT * after / max(1, int(num_racks))
                 + float(energy_benefit) - max(0.0, float(extra_preempt_cost))
                 - max(0.0, float(sla_penalty)))
    return np.array([wait_u, preempt_u], dtype=np.float32)


def victim_tradeoff_utilities(reliefs, costs, sla_weight=RELIEF_SLA_WEIGHT):
    """Raw bounded A3 score: actual violation reduction minus SLA harm."""
    r = np.asarray(reliefs, dtype=np.float32)
    c = np.asarray(costs, dtype=np.float32)
    if len(r) == 0:
        return r
    return r - sla_weight * (c / (1.0 + np.maximum(c, 0.0)))


def adaptive_entropy_coef(current, observed, target):
    return float(np.clip(float(current) + ENT_ADAPT_LR * (float(target) - float(observed)),
                         ENT_COEF, ENT_COEF_MAX))


def _bump_role(env, key):
    env.ep_stats[key] = env.ep_stats.get(key, 0) + 1


def build_relief_candidate_credit(env, job, before_by_rack, preempt_cost):
    """Project one whole-gang removal without changing placement or simulator state."""
    before = np.asarray(before_by_rack, dtype=np.float64)
    if before.shape != (env.num_racks,) or not np.isfinite(before).all():
        raise ValueError("before_by_rack must contain finite risk for every rack")
    after = before.copy()
    for rack in sorted({env.rack_of(host) for host in job.placed_hosts}):
        after[rack] = env.project_rack_violation_fraction(rack, remove_job=job)
    if not np.isfinite(after).all():
        raise FloatingPointError("non-finite projected rack risk")
    immediate = ((W_SERVE + W_CKPT) * job.num_nodes / max(1, env.num_hosts)
                 + W_WAIT * float(job.priority) / max(1.0, env.num_hosts * 3.0))
    energy = W_ENERGY * env.project_job_energy_saved_normalized(job)
    sla = weighted_sla_penalty(preempt_cost, env._sla_priority_total,
                               env.sla_lambda_comp)
    credit = ReliefCredit(float(np.sum(before - after) / env.num_racks),
                          float(energy), float(immediate), float(sla))
    credit.features()
    return after, credit


def gang_step(env, ag, t, act):
    """The 4-agent decision orchestration for one env step, shared by training
    (rollout) and evaluation (eval_gang) so they can never diverge. `act(name,
    obs, mask, step)` returns the chosen action; the caller decides whether it
    samples+records (train) or takes argmax (eval). Actuates on env directly."""
    NR = env.num_racks

    # ---- A4: place pending jobs (priority order) ----
    placements = 0
    for job in list(env.pending):
        if placements >= env.num_hosts:
            break
        rf_now = env.rack_features()
        rf_flat = (env.rack_history_features(rf_now).reshape(-1)
                   if ag is not None and (getattr(ag, "use_stgnn", False)
                                          or getattr(ag, "use_history_mlp", False))
                   else rf_now.reshape(-1))
        jf = env.job_features(job)
        details = env.a4_counterfactual_details(job)
        utility, mask, projected = details["utility"], details["mask"], details["projected"]
        obs = np.concatenate([rf_flat, jf, env.network_features(),
                              counterfactual_observation(utility, mask)])
        a = act("a4", obs, mask, t,
                credit_context=dict(utility=utility, projected_peaks=projected,
                                    aux_feasible=mask[:NR].astype(np.float32),
                                    aux_forced=np.float32(details["forced_defer"]),
                                    horizon_risk=details["horizon_risk"]),
                record=bool(np.sum(mask) > 1), decode_weight=job.num_nodes)
        if a < NR and env.place_job(job, a):
            placements += 1

    # A2/A3/A1 are event-driven. Recompute after each preemption: one job can
    # affect several racks, and a single scheduler step may need several reliefs.
    # A rack where A1 waits (or A3 cannot help) is skipped until another job is
    # preempted; that changes the state and allows reconsideration.
    blocked_racks = set()
    counted_risk_step = False
    while env.running and len(blocked_racks) < NR:
        outcome = _relieve_one_rack(env, ag, t, act, blocked_racks,
                                   count_risk_step=not counted_risk_step)
        if outcome is None:
            break
        counted_risk_step = True
        preempted, flagged = outcome
        if preempted:
            blocked_racks.clear()
        else:
            blocked_racks.add(flagged)


def _relieve_one_rack(env, ag, t, act, blocked_racks, count_risk_step):
    """Take one A2→A3→A1 decision; return (preempted, rack), or None if safe."""
    NR = env.num_racks
    rf = env.rack_features()
    a2_utility = np.array([env.project_rack_violation_fraction(r)
                           for r in range(NR)], dtype=np.float32)
    a2_mask = a2_utility > CF_GAP_EPS
    for rack in blocked_racks:
        a2_mask[rack] = False
    if not bool(a2_mask.any()):
        return None
    if count_risk_step:
        _bump_role(env, "risk_steps")
    a2_base = (env.rack_history_features(rf).reshape(-1)
               if ag is not None and (getattr(ag, "use_stgnn", False)
                                      or getattr(ag, "use_history_mlp", False))
               else rf.reshape(-1))
    a2_obs = np.concatenate([a2_base, a2_utility])
    a2_choice = bool(a2_mask.sum() > 1)
    a2 = int(act("a2", a2_obs, a2_mask, t,
                 credit_context=dict(utility=a2_utility), record=a2_choice))
    _bump_role(env, "a2_ranked")
    _bump_role(env, "a2_choices" if a2_choice else "a2_forced")
    if a2 == int(np.argmax(np.where(a2_mask, a2_utility, -np.inf))):
        _bump_role(env, "a2_correct")
        if a2_choice:
            _bump_role(env, "a2_choice_correct")
    flagged = a2

    # ---- A3 ranks only victims with a positive, causal violation reduction ----
    before_violation = float(a2_utility[flagged])
    cands = env.jobs_on_rack(flagged)[:RACK_SIZE]
    if not cands:
        return False, flagged
    costs = np.array([env.preemption_sla_cost(jb) for jb in cands], dtype=np.float32)
    projected = [build_relief_candidate_credit(env, jb, a2_utility, cost)
                 for jb, cost in zip(cands, costs)]
    after = np.array([v[0][flagged] for v in projected], dtype=np.float32)
    credits = [v[1] for v in projected]
    relief = before_violation - after
    positive = relief > CF_GAP_EPS
    if not bool(positive.any()):
        return False, flagged

    a3_mask = np.zeros(RACK_SIZE, dtype=bool)
    a3_mask[:len(cands)] = positive
    a3_utility = np.full(RACK_SIZE, -np.inf, dtype=np.float32)
    for i, credit in enumerate(credits):
        if positive[i]:
            a3_utility[i] = credit.delta_utility(A1_VIOL_WEIGHT)

    slot_dim = A3_SLOT_DIM
    a3_slots = np.zeros(RACK_SIZE * slot_dim, dtype=np.float32)
    for i, jb in enumerate(cands):
        start = i * slot_dim
        a3_slots[start:start + JOB_OBS_DIM] = env.job_features(jb)
        a3_slots[start + JOB_OBS_DIM] = relief[i]
        a3_slots[start + JOB_OBS_DIM + 1] = costs[i] / (1.0 + costs[i])
        a3_slots[start + JOB_OBS_DIM + 2:start + slot_dim] = credits[i].features()
    a3_obs = np.concatenate([rf[flagged], a3_slots])
    a3_choice = bool(a3_mask.sum() > 1)
    a3 = int(act("a3", a3_obs, a3_mask, t,
                 credit_context=dict(utility=a3_utility), record=a3_choice))
    _bump_role(env, "a3_candidate")
    _bump_role(env, "a3_choices" if a3_choice else "a3_forced")
    if a3 == int(np.argmax(np.where(a3_mask, a3_utility, -np.inf))):
        _bump_role(env, "a3_correct")
        if a3_choice:
            _bump_role(env, "a3_choice_correct")

    # ---- A1 consumes A3's candidate and decides whether net relief is worth it ----
    selected_cost = float(costs[a3])
    selected_after = float(after[a3])
    selected_relief = float(relief[a3])
    norm_preempt_cost = selected_cost / (1.0 + selected_cost)
    selected_credit = credits[a3]
    a1_obs = np.concatenate([rf[flagged], [len(env.pending) / 50.0,
                                           before_violation, selected_after,
                                           selected_relief, norm_preempt_cost],
                             selected_credit.features()])
    a1_utility = selected_credit.a1_utilities(A1_VIOL_WEIGHT)
    a1_mask = np.ones(2, dtype=bool)
    a1 = int(act("a1", a1_obs, a1_mask, t,
                 credit_context=dict(utility=a1_utility), record=True))
    oracle_a1 = int(np.argmax(a1_utility))
    _bump_role(env, "a1_decisions")
    _bump_role(env, "relief_opportunities")
    _bump_role(env, "a1_oracle_preempt" if oracle_a1 == 1 else "a1_oracle_wait")
    if a1 == oracle_a1:
        _bump_role(env, "a1_correct")
    if a1 == 1:
        _bump_role(env, "a1_preempt")
        _bump_role(env, "preempt_on_risk")
        env.preempt_job(cands[a3])
        _bump_role(env, "a3_executed")
    return a1 == 1, flagged


def rollout(env, ag):
    """One training episode. Returns per-agent transitions + per-step critic data."""
    env.reset()
    steps = []          # per step: global_state, reward, done, value
    trans = {k: [] for k in ag.actors}
    a4_aux = []

    def act(name, obs, mask, step, credit_context=None, record=True, **_kwargs):
        a, lp = select_action(ag.actors[name], obs.astype(np.float32),
                              mask.astype(bool) if mask is not None else None, mode="train")
        if name == "a4" and credit_context is not None and "aux_feasible" in credit_context:
            a4_aux.append(dict(obs=obs.astype(np.float32),
                               feasible=credit_context["aux_feasible"],
                               forced=credit_context["aux_forced"],
                               horizon_risk=credit_context["horizon_risk"]))
        if not record:
            return a
        item = dict(obs=obs.astype(np.float32), action=int(a),
                                logp=float(lp),
                                mask=(mask.astype(bool) if mask is not None else None),
                                step=step)
        if credit_context is not None:
            with torch.no_grad():
                ob_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                mk_t = torch.as_tensor(mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
                probs = torch.softmax(ag.actors[name](ob_t, mk_t), dim=-1)[0].cpu().numpy()
            item["local_adv"] = counterfactual_advantage(
                credit_context["utility"], probs, int(a))
            item["cf_utility"] = np.asarray(
                credit_context["utility"], dtype=np.float32).copy()
            if "projected_peaks" in credit_context:
                item["projected_peaks"] = chosen_projected_peaks(
                    credit_context["projected_peaks"], int(a))
        trans[name].append(item)
        return a

    for t in range(env.max_steps):
        g = build_global(env)
        v = ag.value(g)
        gang_step(env, ag, t, act)
        m = env.advance()
        r = m["reward"] * REWARD_SCALE
        done = 1.0 if env.done else 0.0
        steps.append(dict(g=g, r=r, done=done, v=v))
        if env.done:
            break

    for name, items in trans.items():
        env.ep_stats[f"{name}_samples"] = len(items)
        env.ep_stats[f"{name}_unique_actions"] = len({x["action"] for x in items})
        if name == "a4":
            env.ep_stats["a4_defers"] = sum(1 for x in items if x["action"] == env.num_racks)

    trans["_a4_aux"] = a4_aux
    return steps, trans, env.ep_stats


def collect_episode(ag, steps, trans):
    """Compute GAE for one episode and package critic + per-agent samples (raw
    advantages; normalized later across the whole batch). Returns a plain-numpy
    bundle so several episodes can be pooled into one low-variance PPO update."""
    rewards = [s["r"] for s in steps]
    values = [s["v"] for s in steps]
    dones = [s["done"] for s in steps]
    last_v = 0.0 if dones[-1] else ag.value(steps[-1]["g"])
    adv, ret = gae(rewards, values, dones, last_v)
    agents = {}
    a4_aux = trans.get("_a4_aux", [])
    for name, tr in trans.items():
        if name.startswith("_"):
            continue
        if not tr:
            continue
        has_mask = tr[0]["mask"] is not None
        team_adv = np.array([adv[x["step"]] for x in tr], dtype=np.float32)
        agents[name] = dict(
            obs=np.stack([x["obs"] for x in tr]),
            acts=np.array([x["action"] for x in tr]),
            old_lp=np.array([x["logp"] for x in tr], dtype=np.float32),
            team_adv=team_adv,
            local_adv=(np.array([x["local_adv"] for x in tr], dtype=np.float32)
                       if "local_adv" in tr[0] else None),
            cf_utility=(np.stack([x["cf_utility"] for x in tr])
                        if "cf_utility" in tr[0] else None),
            masks=(np.stack([x["mask"] for x in tr]) if has_mask else None),
        )
    aux = None if not a4_aux else dict(
        obs=np.stack([x["obs"] for x in a4_aux]),
        feasible=np.stack([x["feasible"] for x in a4_aux]),
        forced=np.asarray([x["forced"] for x in a4_aux], dtype=np.float32),
        horizon_risk=np.stack([x["horizon_risk"] for x in a4_aux]))
    return dict(G=np.stack([s["g"] for s in steps]),
                RET=ret.astype(np.float32), agents=agents, a4_aux=aux)


def ppo_update_batch(ag, batch):
    """One PPO update pooled over a BATCH of episodes — the variance fix. Advantages
    are normalized across the whole batch (not per-episode), so the gradient reflects
    the average episode rather than a single high-variance rollout."""
    G = torch.as_tensor(np.concatenate([b["G"] for b in batch]), device=DEVICE)
    RET = torch.as_tensor(np.concatenate([b["RET"] for b in batch]),
                          device=DEVICE).unsqueeze(1)
    for _ in range(EPOCHS):
        closs = F.mse_loss(ag.critic(G), RET)
        if not torch.isfinite(closs).item():
            raise FloatingPointError("non-finite critic loss")
        ag.opt["critic"].zero_grad(); closs.backward(); ag.opt["critic"].step()

    logs = {}
    names = set().union(*[set(b["agents"]) for b in batch])
    for name in names:
        parts = [b["agents"][name] for b in batch if name in b["agents"]]
        if not parts:
            continue
        obs = torch.as_tensor(np.concatenate([p["obs"] for p in parts]), device=DEVICE)
        acts = torch.as_tensor(np.concatenate([p["acts"] for p in parts]), device=DEVICE)
        old_lp = torch.as_tensor(np.concatenate([p["old_lp"] for p in parts]), device=DEVICE)
        team_adv = np.concatenate([p["team_adv"] for p in parts])
        local_parts = [p["local_adv"] for p in parts if p["local_adv"] is not None]
        if USE_CF_SUPERVISION and local_parts and sum(len(x) for x in local_parts) == len(team_adv):
            local_adv = np.concatenate(local_parts)
            team_coef = A4_TEAM_ADV_COEF if name == "a4" else RELIEF_TEAM_ADV_COEF
            adv = combine_local_team_advantages(local_adv, team_adv, team_coef)
        else:
            local_adv = None
            adv = _zscore(team_adv)
        a_adv = torch.as_tensor(adv, device=DEVICE)
        if not torch.isfinite(a_adv).all().item():
            raise FloatingPointError(f"non-finite {name} advantage")
        masks = (torch.as_tensor(np.concatenate([p["masks"] for p in parts]), device=DEVICE)
                 if parts[0]["masks"] is not None else None)
        actor = ag.actors[name]
        ecoef = ag.ent_coef[name]
        cf_parts = [p["cf_utility"] for p in parts if p["cf_utility"] is not None]
        cf_utility = (torch.as_tensor(np.concatenate(cf_parts), device=DEVICE)
                      if cf_parts and sum(len(x) for x in cf_parts) == len(acts)
                      else None)
        ent = 0.0
        cf_rows = 0
        for _ in range(EPOCHS):
            logits = actor(obs, masks)
            dist = torch.distributions.Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(acts) - old_lp)
            ent_rows = masked_categorical_entropy(logits, masks)
            choice_rows = ((masks.sum(dim=1) > 1) if masks is not None
                           else torch.ones_like(ent_rows, dtype=torch.bool))
            ent = (ent_rows[choice_rows].mean() if bool(choice_rows.any())
                   else torch.zeros((), device=DEVICE))
            entropy_floor = torch.relu(torch.as_tensor(
                ENTROPY_TARGET.get(name, 0.0), device=DEVICE) - ent)
            loss = -torch.min(ratio * a_adv,
                              torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a_adv).mean() \
                - ecoef * ent + ENT_FLOOR_PENALTY * entropy_floor.pow(2)
            if USE_CF_SUPERVISION and cf_utility is not None and masks is not None:
                cf_loss, cf_rows = counterfactual_policy_loss(
                    logits, cf_utility, masks)
                loss = loss + CF_AUX_COEF * cf_loss
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"non-finite {name} actor loss")
            ag.opt[name].zero_grad(); loss.backward(); ag.opt[name].step()
        local_mean = (float(local_adv.mean()) if local_adv is not None else float("nan"))
        observed_ent = float(ent.detach()) if torch.is_tensor(ent) else float(ent)
        if masks is None or bool((masks.sum(dim=1) > 1).any()):
            ag.ent_coef[name] = adaptive_entropy_coef(
                ecoef, observed_ent, ENTROPY_TARGET.get(name, 0.0))
        cf_acc, cf_regret = float("nan"), float("nan")
        if cf_utility is not None and masks is not None:
            with torch.no_grad():
                final_logits = actor(obs, masks)
                legal = masks.bool() & torch.isfinite(cf_utility)
                neg_inf = torch.full_like(cf_utility, -torch.inf)
                pos_inf = torch.full_like(cf_utility, torch.inf)
                best_u, best_a = torch.where(legal, cf_utility, neg_inf).max(dim=1)
                worst_u = torch.where(legal, cf_utility, pos_inf).min(dim=1).values
                informative = ((legal.sum(dim=1) > 1)
                               & ((best_u - worst_u) > CF_GAP_EPS))
                if bool(informative.any()):
                    pred = final_logits.argmax(dim=1)
                    chosen_u = cf_utility.gather(1, pred.unsqueeze(1)).squeeze(1)
                    cf_acc = float((pred[informative] == best_a[informative]).float().mean())
                    cf_regret = float((best_u[informative] - chosen_u[informative]).mean())
        logs[name] = (len(acts), observed_ent, local_mean, ag.ent_coef[name],
                      cf_rows, cf_acc, cf_regret)
    aux_parts = [b["a4_aux"] for b in batch if b.get("a4_aux") is not None]
    if USE_CF_SUPERVISION and aux_parts and hasattr(ag.actors["a4"], "auxiliary"):
        aux_obs = torch.as_tensor(np.concatenate([p["obs"] for p in aux_parts]), device=DEVICE)
        aux_feasible = torch.as_tensor(np.concatenate([p["feasible"] for p in aux_parts]), device=DEVICE)
        aux_forced = torch.as_tensor(np.concatenate([p["forced"] for p in aux_parts]), device=DEVICE)
        for _ in range(EPOCHS):
            feas_logits, forced_logits = ag.actors["a4"].auxiliary(aux_obs)
            aux_loss = (F.binary_cross_entropy_with_logits(feas_logits, aux_feasible)
                        + F.binary_cross_entropy_with_logits(forced_logits, aux_forced))
            if not torch.isfinite(aux_loss).item():
                raise FloatingPointError("non-finite A4 auxiliary loss")
            ag.opt["a4"].zero_grad()
            (A4_FEAS_AUX_COEF * aux_loss).backward()
            ag.opt["a4"].step()
        logs["a4_aux"] = (len(aux_obs), float("nan"), float(aux_loss.detach()),
                          ag.ent_coef["a4"], 0, float("nan"), float("nan"))
    return logs



def evaluate_deterministic(env, ag, a4_mode="sequence"):
    """Deterministic deployment replay.

    The default is sequence-level marginal rounding. Pointwise argmax remains an
    explicit audit mode so its collapse can never be silently hidden.
    """
    env.reset()
    act_eval = make_deterministic_act(ag, a4_mode=a4_mode)
    steps_eval = []
    for t in range(env.max_steps):
        gang_step(env, ag, t, act_eval)
        m = env.advance()
        steps_eval.append(m["reward"] * REWARD_SCALE)
        if env.done:
            break
            
    stats = env.ep_stats
    sla = env.sla_stats()
    team_r = sum(steps_eval) / REWARD_SCALE
    return team_r, stats, sla


def evaluate_argmax(env, ag):
    """Backward-compatible pointwise-argmax audit, not the deploy decoder."""
    return evaluate_deterministic(env, ag, a4_mode="pointwise")


def save_sla_multipliers(env):
    out_dir = os.environ.get("GANG_CHECKPOINT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sla_multipliers.json"), "w") as f:
        json.dump(dict(admission=env.sla_lambda_adm,
                       restart=env.sla_lambda_restart,
                       completion=env.sla_lambda_comp,
                       target_admission=SLA_TARGET_ADMISSION,
                       target_restart=SLA_TARGET_RESTART,
                       target_completion=SLA_TARGET_COMPLETION), f, indent=2)


def save_deployment_decoder(name):
    out_dir = os.environ.get("GANG_CHECKPOINT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "deployment_decoder.json"), "w") as f:
        json.dump(dict(name=name, scope="A4 placement sequence", random=False,
                       pointwise_argmax_retained=True), f, indent=2)


def save_metric_manifest(overwrite=True):
    """Freeze metric names/definitions so later runs cannot silently drop fields."""
    out_dir = os.environ.get("GANG_CHECKPOINT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "metric_manifest.json")
    if not overwrite and os.path.exists(manifest_path):
        return
    manifest = dict(
        schema_version="gang_metrics_v1",
        semantic_schema_version="gang_metrics_v2",
        simulator_semantics=SIMULATOR_SEMANTICS,
        relief_credit=relief_credit_metadata(),
        numerics_checked=True,
        choice_accuracy="genuine legal choices only; NaN when denominator is zero",
        architecture=dict(
            a2_a4=("rack_stgnn_shared_scorer" if USE_STGNN else
                   "history_mlp" if USE_HISTORY_MLP else "mlp"),
            a1_a3="mlp",
            history_steps=(RACK_HISTORY_LEN if (USE_STGNN or USE_HISTORY_MLP) else 1),
            counterfactual_supervision=USE_CF_SUPERVISION,
        ),
        workload=dict(source=os.environ.get("POD_HOURLY_JOBS"),
                      train_start_hour=WORKLOAD_START_HOUR,
                      validation_start_hour=VALIDATION_START_HOUR,
                      window_hours=WORKLOAD_WINDOW_HOURS,
                      subsample=SUBSAMPLE, seed=SEED),
        sla=dict(
            sla_viol="unique jobs that are dropped or completed late",
            admission_late="jobs first placed after SLA_MAX_WAIT_STEPS",
            restart_late="jobs whose cumulative post-preemption requeue exceeds SLA_MAX_RESTART_WAIT_STEPS",
            completion_late="completed jobs past completion deadline",
            completion_constraint="completion_late + dropped, matching dual update",
            dropped="jobs unfinished at fixed episode horizon",
        ),
        network=dict(
            cross_rack_frac="node-weighted active-gang fraction located off its rack",
            net_comm_tax=NET_COMM_TAX,
            energy_effect="GPU power multiplied by 1 + net_comm_tax * cross_rack_frac",
            simulated_packet_telemetry=True,
            source="ring all-reduce flows routed over shared Edge-Aggregate-Edge fat-tree links",
            fields=["network_cross_bytes", "network_local_bytes",
                    "network_delay_sec", "network_max_link_util",
                    "network_extra_steps"],
            warning="simulator telemetry, not hardware-measured network counters",
        ),
        power=dict(
            rack_viol="0.2-second fine samples above rack budget, summed over racks",
            energy_kwh="cumulative Java NREL replay energy including network power tax",
        ),
        resource=dict(fields=["wall_time_s", "cpu_time_s", "avg_cpu_pct", "peak_rss_mb"]),
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def deployment_key(stats, sla, reward):
    """Constraint-first checkpoint key, then power violations, then reward.

    Once SLA is inside its allowed budget, a tiny further SLA improvement must not
    overwrite a low-power checkpoint with the historical 46k-violation collapse.
    """
    n = max(1, int(sla["n"]))
    served_rate = float(sla["served"]) / n
    adm_rate = float(sla["admission_late"]) / n
    restart_rate = float(sla.get("restart_late", 0)) / n
    comp_rate = float(sla["completion_late"] + sla["dropped"]) / n
    feasible = (adm_rate <= SLA_TARGET_ADMISSION and
                restart_rate <= SLA_TARGET_RESTART and
                comp_rate <= SLA_TARGET_COMPLETION)
    goodput_ok = served_rate >= DEPLOY_MIN_SERVED_RATE
    sla_rate = float(sla["sla_viol"]) / n
    sla_excess = max(0.0, adm_rate - SLA_TARGET_ADMISSION,
                     restart_rate - SLA_TARGET_RESTART,
                     comp_rate - SLA_TARGET_COMPLETION)
    net_total = max(1.0, float(stats.get("network_total_bytes", 0.0)))
    net_cross_frac = float(stats.get("network_cross_bytes", 0.0)) / net_total
    net_steps = max(1.0, float(stats.get("cross_rack_n", 0.0)))
    net_delay_n = (float(stats.get("network_delay_sec", 0.0))
                   / (net_steps * max(1.0, float(INTERVAL_SEC))))
    net_cost = net_delay_n + float(stats.get("network_max_link_util", 0.0))
    return (int(feasible), int(goodput_ok), -sla_excess,
            -int(stats["rack_viol"]), -net_cost, -net_cross_frac,
            -sla_rate, served_rate, float(reward))


def network_console_summary(stats):
    total = max(1.0, float(stats.get("network_total_bytes", 0.0)))
    return dict(
        total_gb=total / 1e9,
        cross_gb=float(stats.get("network_cross_bytes", 0.0)) / 1e9,
        cross_pct=100.0 * float(stats.get("network_cross_bytes", 0.0)) / total,
        delay_sec=float(stats.get("network_delay_sec", 0.0)),
        max_link_pct=100.0 * float(stats.get("network_max_link_util", 0.0)),
        xrack=(float(stats.get("cross_rack_sum", 0.0))
               / max(1.0, float(stats.get("cross_rack_n", 0.0)))),
    )

def train():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    port = int(os.environ.get("BRIDGE_PORT", 25333))
    validation_env = None
    if VALIDATION_START_HOUR and not WORKLOAD_START_HOUR:
        raise ValueError("VALIDATION_START_HOUR requires WORKLOAD_START_HOUR")
    if WORKLOAD_START_HOUR:
        start_hour = int(WORKLOAD_START_HOUR)
        window_hours = int(WORKLOAD_WINDOW_HOURS)
        if window_hours <= 0:
            raise ValueError("WORKLOAD_WINDOW_HOURS must be positive")
        start_step = start_hour * STEPS_PER_HOUR
        end_step = (start_hour + window_hours) * STEPS_PER_HOUR
        env = GangEnv(
            port=port, subsample=SUBSAMPLE, max_steps=MAX_STEPS,
            min_arrival_step=start_step, arrival_end_step=end_step)
        for job in env._all_jobs:
            job.arrival_step -= start_step
        print(f"[workload] source={os.environ.get('POD_HOURLY_JOBS', 'default')} "
              f"hours=[{start_hour},{start_hour + window_hours}) "
              f"steps=[0,{window_hours * STEPS_PER_HOUR})")
        if VALIDATION_START_HOUR:
            val_start_hour = int(VALIDATION_START_HOUR)
            val_window_hours = int(VALIDATION_WINDOW_HOURS or window_hours)
            if val_window_hours <= 0:
                raise ValueError("VALIDATION_WINDOW_HOURS must be positive")
            if max(start_hour, val_start_hour) < min(
                    start_hour + window_hours, val_start_hour + val_window_hours):
                raise ValueError("training and validation windows overlap")
            val_start_step = val_start_hour * STEPS_PER_HOUR
            validation_env = GangEnv(
                port=port, subsample=SUBSAMPLE, max_steps=MAX_STEPS,
                min_arrival_step=val_start_step,
                arrival_end_step=(val_start_hour + val_window_hours) * STEPS_PER_HOUR)
            if not validation_env._all_jobs:
                raise ValueError("validation window contains no matched GPU jobs")
            for job in validation_env._all_jobs:
                job.arrival_step -= val_start_step
            print(f"[validation] hours=[{val_start_hour},{val_start_hour + val_window_hours}) "
                  f"jobs={len(validation_env._all_jobs)}")
    else:
        env = GangEnv(port=port, subsample=SUBSAMPLE,
                      max_arrival_step=48, max_steps=MAX_STEPS)
    gdim = len(build_global(env))
    ag = Agents(env.num_racks, gdim)
    out_dir = os.environ.get("GANG_CHECKPOINT_DIR", ".")
    training_state_path = os.environ.get(
        "TRAIN_STATE_PATH", os.path.join(out_dir, "training_state.pt"))
    best_batch = -1e18
    best_eval = -1e18
    best_deploy_key = None
    best_decoder = None
    stale = 0
    ep = 0
    if RESUME:
        resume_path = os.environ.get("RESUME_PATH", training_state_path)
        state = load_training_state(resume_path, ag, env)
        ep = int(state["episode"])
        best_batch = float(state["best_batch"])
        best_eval = float(state["best_eval"])
        best_deploy_key = state["best_deploy_key"]
        best_decoder = state["best_decoder"]
        stale = int(state["stale"])
        print(f"[resume] loaded {resume_path}; next_episode={ep} "
              f"best_eval={best_eval:.2f} decoder={best_decoder}")
    if os.environ.get("OBS_ENABLED", "1").strip().lower() in ("1", "true", "yes"):
        obs = GangObservability(
            os.environ.get("TB_LOG_DIR", "runs/gang"),
            prometheus_port=int(os.environ.get("PROMETHEUS_PORT", "8000")))
    else:
        obs = GangObservability.disabled()
    save_metric_manifest()
    total_meter = ResourceMeter.for_bridge_port(port)
    print(f"[gang-train] racks={env.num_racks} gdim={gdim} "
          f"architecture={'STGNN' if ag.use_stgnn else 'MLP'} dims={ag.dims} "
          f"jobs={len(env._all_jobs)} device={DEVICE}")

    csv_path = os.environ.get("GANG_CSV", "gang_train_summary.csv")
    if RESUME and os.path.exists(csv_path):
        with open(csv_path, newline="") as src:
            rows = list(csv.reader(src))
        header, data = rows[0], rows[1:]
        data = [row for row in data if row and int(row[0]) < ep]
        with open(csv_path, "w", newline="") as dst:
            writer = csv.writer(dst)
            writer.writerow(header)
            writer.writerows(data)
    csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    out = open(csv_path, "a" if RESUME else "w", newline="")
    w = csv.writer(out)
    if not csv_exists:
        w.writerow(["ep", "team_reward", "rack_viol", "admitted", "preempted",
                "completed", "wait_cost", "energy_kwh", "cross_rack_frac",
                "network_cross_gb", "network_local_gb", "network_total_gb",
                "network_cross_byte_pct", "network_delay_sec",
                "network_max_link_util_pct", "network_extra_steps",
                "sla_viol", "served", "sla_admission_late",
                "sla_restart_late",
                "sla_completion_late", "sla_completion_viol", "sla_dropped", "sla_late",
                "sla_viol_pct", "sla_admission_pct", "sla_restart_pct", "sla_completion_pct",
                "wall_time_s", "cpu_time_s",
                "avg_cpu_pct", "peak_rss_mb", "a2_ranked", "a3_candidate",
                "a1_decisions", "a1_preempt", "a3_executed",
                "relief_opportunities", "risk_steps", "a2_choices", "a2_forced",
                "a2_correct", "a2_choice_correct", "a3_choices", "a3_forced", "a3_correct", "a3_choice_correct",
                "a1_oracle_preempt", "a1_oracle_wait", "a1_correct",
                  "a1_unique", "a2_unique", "a3_unique", "a4_unique", "a1_samples", "a2_samples", "a3_samples", "a4_samples", "a1_preempt_rate", "a2_choice_accuracy", "a3_choice_accuracy", "a4_defer_rate", "a4_placement_rate"])
    print(f"[gang-train] batch={BATCH_EPISODES} ep/update over {NUM_EPISODES} episodes "
          f"starting_at={ep}")
    while ep < NUM_EPISODES:
        batch, batch_r, batch_sla, batch_roles, last = [], [], [], [], None
        for _ in range(BATCH_EPISODES):
            if ep >= NUM_EPISODES:
                break
            ep_meter = ResourceMeter.for_bridge_port(port)
            steps, trans, stats = rollout(env, ag)
            ep_resource = ep_meter.finish()
            total_meter.sample()
            team_r = sum(s["r"] for s in steps) / REWARD_SCALE
            batch.append(collect_episode(ag, steps, trans))
            batch_r.append(team_r)
            xrack = stats["cross_rack_sum"] / max(1, stats["cross_rack_n"])
            sla = env.sla_stats()   # bounded job-level SLA: dropped + late (disjoint)
            n_sla = max(1, sla["n"])
            batch_sla.append((sla["admission_late"] / n_sla,
                              sla["restart_late"] / n_sla,
                              (sla["completion_late"] + sla["dropped"]) / n_sla))
            batch_roles.append(stats)
            w.writerow([ep, f"{team_r:.2f}", stats["rack_viol"], stats["admitted"],
                        stats["preempted"], stats["completed"], f"{stats['wait_cost']:.0f}",
                        f"{stats['energy']:.1f}", f"{xrack:.4f}",
                        f"{stats['network_cross_bytes'] / 1e9:.6f}",
                        f"{stats['network_local_bytes'] / 1e9:.6f}",
                        f"{stats['network_total_bytes'] / 1e9:.6f}",
                        f"{100.0 * stats['network_cross_bytes'] / max(1.0, stats['network_total_bytes']):.3f}",
                        f"{stats['network_delay_sec']:.6f}",
                        f"{100.0 * stats['network_max_link_util']:.3f}",
                        f"{stats['network_extra_steps']:.0f}",
                        sla["sla_viol"], sla["served"], sla["admission_late"],
                        sla["restart_late"],
                        sla["completion_late"], sla["completion_late"] + sla["dropped"],
                        sla["dropped"], sla["late"],
                        f"{100.0 * sla['sla_viol'] / n_sla:.2f}",
                        f"{100.0 * sla['admission_late'] / n_sla:.2f}",
                        f"{100.0 * sla['restart_late'] / n_sla:.2f}",
                        f"{100.0 * (sla['completion_late'] + sla['dropped']) / n_sla:.2f}",
                        f"{ep_resource['wall_time_s']:.3f}",
                        f"{ep_resource['cpu_time_s']:.3f}",
                        f"{ep_resource['avg_cpu_pct']:.1f}",
                        f"{ep_resource['peak_rss_mb']:.1f}",
                        stats["a2_ranked"], stats["a3_candidate"],
                        stats["a1_decisions"], stats["a1_preempt"],
                        stats["a3_executed"], stats["relief_opportunities"],
                        stats.get("risk_steps", 0), stats.get("a2_choices", 0),
                        stats.get("a2_forced", 0), stats.get("a2_correct", 0),
                        stats.get("a2_choice_correct", 0),
                        stats.get("a3_choices", 0), stats.get("a3_forced", 0),
                        stats.get("a3_correct", 0), stats.get("a3_choice_correct", 0),
                        stats.get("a1_oracle_preempt", 0),
                        stats.get("a1_oracle_wait", 0), stats.get("a1_correct", 0),
                        stats.get("a1_unique_actions", 0), stats.get("a2_unique_actions", 0),
                        stats.get("a3_unique_actions", 0), stats.get("a4_unique_actions", 0),
                        stats.get("a1_samples", 0), stats.get("a2_samples", 0),
                        stats.get("a3_samples", 0), stats.get("a4_samples", 0),
                        stats.get("a1_preempt", 0) / max(1, stats.get("a1_decisions", 1)),
                        (stats.get("a2_choice_correct", 0) / stats["a2_choices"]
                         if stats.get("a2_choices", 0) else float("nan")),
                        (stats.get("a3_choice_correct", 0) / stats["a3_choices"]
                         if stats.get("a3_choices", 0) else float("nan")),
                        stats.get("a4_defers", 0) / max(1, stats.get("a4_samples", 1)),
                        (stats.get("a4_samples", 0) - stats.get("a4_defers", 0)) / max(1, stats.get("a4_samples", 1))])
            obs.log_episode(ep, {
                "team_reward": team_r,
                "rack_viol": stats["rack_viol"],
                "energy_kwh": stats["energy"],
                "sla_viol_pct": 100.0 * sla["sla_viol"] / n_sla,
                "sla_admission_pct": 100.0 * sla["admission_late"] / n_sla,
                "sla_restart_pct": 100.0 * sla["restart_late"] / n_sla,
                "sla_completion_pct": 100.0 * (sla["completion_late"] + sla["dropped"]) / n_sla,
                "forced_defer_states": stats.get("forced_defer_states", 0),
                "feasible_anchors_mean": stats.get("feasible_anchor_sum", 0.0) /
                                         max(1, stats.get("a4_decision_states", 0)),
                "network_cross_gb": stats["network_cross_bytes"] / 1e9,
                "network_delay_sec": stats["network_delay_sec"],
                "network_max_link_util_pct": 100.0 * stats["network_max_link_util"],
                "avg_cpu_pct": ep_resource["avg_cpu_pct"],
                "peak_rss_mb": ep_resource["peak_rss_mb"],
            })
            obs.flush()
            print(f"ep {ep:3d} | Rteam={team_r:8.2f} viol={stats['rack_viol']:7d} "
                  f"adm={stats['admitted']:3d} pre={stats['preempted']:3d} "
                  f"comp={stats['completed']:3d} sla={sla['sla_viol']:3d} "
                  f"wait={stats['wait_cost']:5.0f} "
                  f"E={stats['energy']:6.1f} xrack={xrack:.3f} "
                  f"netX={stats['network_cross_bytes']/1e9:.2f}GB "
                  f"netD={stats['network_delay_sec']:.1f}s "
                  f"time={ep_resource['wall_time_s']:.1f}s "
                  f"cpu={ep_resource['avg_cpu_pct']:.0f}% "
                  f"ram={ep_resource['peak_rss_mb']:.0f}MB "
                  f"roles=risk:{stats.get('risk_steps', 0)} "
                  f"a2:{stats['a2_ranked']} a3:{stats['a3_candidate']}/"
                  f"{stats['a3_executed']} a1:{stats['a1_decisions']}/"
                  f"{stats['a1_preempt']}")
            last = stats
            ep += 1
        out.flush()
        logs = ppo_update_batch(ag, batch)
        total_meter.sample()
        env.update_sla_multipliers(
            admission_rate=float(np.mean([x[0] for x in batch_sla])),
            restart_rate=float(np.mean([x[1] for x in batch_sla])),
            completion_rate=float(np.mean([x[2] for x in batch_sla])))
        save_sla_multipliers(env)
        role_totals = {k: sum(int(s.get(k, 0)) for s in batch_roles) for k in
                       ("a2_ranked", "a3_candidate", "a1_decisions", "a1_preempt",
                        "a3_executed", "relief_opportunities", "preempt_on_risk",
                        "risk_steps", "a2_choices", "a2_forced", "a2_correct", "a2_choice_correct",
                        "a3_choices", "a3_forced", "a3_correct", "a3_choice_correct",
                        "a1_oracle_preempt", "a1_oracle_wait", "a1_correct")}
        batch_mean = sum(batch_r) / max(1, len(batch_r))
        best_batch = max(best_batch, batch_mean)

        # Compare both deterministic decoders on the same validation episode.
        # Selection is constraint-first; neither decoder is assumed superior.
        selection_env = validation_env or env
        for attr in ("sla_lambda_adm", "sla_lambda_restart", "sla_lambda_comp"):
            setattr(selection_env, attr, getattr(env, attr))
        eval_r, eval_stats, eval_sla = evaluate_deterministic(
            selection_env, ag, a4_mode="sequence")
        point_r, point_stats, point_sla = evaluate_deterministic(
            selection_env, ag, a4_mode="pointwise")
        seq_net = network_console_summary(eval_stats)
        point_net = network_console_summary(point_stats)
        n_eval = max(1, eval_sla["n"])
        adm_rate = eval_sla["admission_late"] / n_eval
        comp_rate = (eval_sla["completion_late"] + eval_sla["dropped"]) / n_eval
        restart_rate = eval_sla["restart_late"] / n_eval
        feasible = (adm_rate <= SLA_TARGET_ADMISSION
                    and restart_rate <= SLA_TARGET_RESTART
                    and comp_rate <= SLA_TARGET_COMPLETION)
        eval_decisions = int(eval_stats.get("a1_decisions", 0))
        selective_coverage = (eval_stats.get("a1_oracle_preempt", 0) > 0
                              and eval_stats.get("a1_oracle_wait", 0) > 0)
        a1_accuracy = eval_stats.get("a1_correct", 0) / max(1, eval_decisions)
        role_ready = bool(eval_decisions >= 3 and selective_coverage
                          and a1_accuracy >= 0.60)
        candidates = [
            (deployment_key(eval_stats, eval_sla, eval_r), "sequence", eval_r),
            (deployment_key(point_stats, point_sla, point_r), "pointwise", point_r),
        ]
        deploy_key, decoder_name, selected_r = max(candidates, key=lambda x: x[0])
        if best_deploy_key is None or deploy_key > best_deploy_key:
            best_deploy_key = deploy_key
            best_eval = selected_r
            best_decoder = decoder_name
            ag.save(tag="gang_best")
            save_deployment_decoder(decoder_name)
            stale = 0
        else:
            stale += 1
        save_training_state(
            training_state_path, ag, env, episode=ep,
            best_batch=best_batch, best_eval=best_eval,
            best_deploy_key=best_deploy_key, best_decoder=best_decoder,
            stale=stale)
        cnt = {k: v[0] for k, v in logs.items()}
        ent = {k: f"{v[1]:.2f}" for k, v in logs.items()}
        entcoef = {k: f"{v[3]:.3f}" for k, v in logs.items()}
        cfdiag = {k: dict(rows=v[4], acc=(None if np.isnan(v[5]) else round(v[5], 3)),
                          regret=(None if np.isnan(v[6]) else round(v[6], 5)))
                  for k, v in logs.items()}
        for agent, values in logs.items():
            obs.log_agent(ep - 1, agent, entropy=values[1], samples=values[0])
        obs.log_eval(ep - 1, "sequence", eval_r, eval_stats, eval_sla)
        obs.log_eval(ep - 1, "pointwise", point_r, point_stats, point_sla)
        obs.flush()
        a4_local = logs.get("a4", (0, 0.0, float("nan")))[2]
        print(f"  -> update: batch_mean={batch_mean:7.2f} | eval_seq={eval_r:7.2f} | "
              f"eval_point={point_r:7.2f} | "
              f"best_eval={best_eval:7.2f}({best_decoder}) | stale={stale}")
        print(f"     samples={cnt}")
        print(f"     entropy={ent}")
        print(f"     entropy_coef={entcoef}")
        print(f"     counterfactual={cfdiag}")
        print(f"     a4_counterfactual_adv_mean={a4_local:.4f}")
        print(f"     sla_dual: admission={env.sla_lambda_adm:.3f} "
              f"restart={env.sla_lambda_restart:.3f} completion={env.sla_lambda_comp:.3f}")
        print(f"     roles={role_totals}")
        print(f"     eval_components: E={eval_stats['energy']:.1f} viol={eval_stats['rack_viol']} "
              f"sla={eval_sla['sla_viol']} wait={eval_stats['wait_cost']:.0f} "
              f"pre={eval_stats['a1_preempt']} a3={eval_stats['a3_candidate']}/"
              f"{eval_stats['a3_executed']} opp={eval_decisions} "
              f"oracle={eval_stats.get('a1_oracle_preempt', 0)}/"
              f"{eval_stats.get('a1_oracle_wait', 0)} "
              f"a1acc={a1_accuracy:.2f} promotion_ready={role_ready} "
              f"xrack={seq_net['xrack']:.3f} net={seq_net['total_gb']:.2f}GB "
              f"netX={seq_net['cross_gb']:.2f}GB/{seq_net['cross_pct']:.1f}% "
              f"netD={seq_net['delay_sec']:.1f}s link={seq_net['max_link_pct']:.2f}%")
        print(f"     pointwise_audit: viol={point_stats['rack_viol']} "
              f"pre={point_stats['a1_preempt']} admitted={point_stats['admitted']} "
              f"served={point_sla['served']} sla={point_sla['sla_viol']} "
              f"xrack={point_net['xrack']:.3f} net={point_net['total_gb']:.2f}GB "
              f"netX={point_net['cross_gb']:.2f}GB/{point_net['cross_pct']:.1f}% "
              f"netD={point_net['delay_sec']:.1f}s "
              f"link={point_net['max_link_pct']:.2f}%")
        if EARLY_STOP_PATIENCE > 0 and stale >= EARLY_STOP_PATIENCE:
            print(f"[early-stop] deterministic deploy eval stalled {stale} updates; "
                  f"best_eval={best_eval:.2f}")
            break

    ag.save()
    out.close()
    obs.close()
    total_resource = total_meter.finish()
    out_dir = os.environ.get("GANG_CHECKPOINT_DIR", ".")
    with open(os.path.join(out_dir, "training_resources.json"), "w") as f:
        json.dump(total_resource, f, indent=2)
    print(f"[resources] wall={total_resource['wall_time_s']:.2f}s "
          f"cpu={total_resource['avg_cpu_pct']:.1f}% "
          f"peak_ram={total_resource['peak_rss_mb']:.1f}MB")
    print(f"[gang-train] done; final -> *_gang.pt ; "
          f"best(eval={best_eval:.2f}, decoder={best_decoder}) -> *_gang_best.pt")


if __name__ == "__main__":
    train()
