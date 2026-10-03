"""A4 training-only action-value baseline fitted to realized team returns.

The policy target is sampled discounted return minus the policy-marginal
baseline, not a pressure ranking and not an oracle Q estimate. Baselines are
frozen before fitting each batch and gated by prediction on a withheld episode.
The critic is never consulted during decentralized deployment.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def discounted_returns(rewards, dones, gamma):
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = float(rewards[t]) + gamma * running * (1.0 - float(dones[t]))
        out[t] = running
    return out


def marginal_baseline(q, probabilities, masks):
    legal = masks.bool()
    if not bool(legal.any(dim=-1).all()):
        raise ValueError('baseline requires a legal action in every row')
    safe_q = torch.where(legal, q, torch.zeros_like(q))
    if not bool(torch.isfinite(safe_q).all()):
        raise ValueError('non-finite legal action value')
    p = torch.where(legal, probabilities.clamp_min(0), torch.zeros_like(probabilities))
    total = p.sum(dim=-1, keepdim=True)
    if not bool(torch.isfinite(p).all()) or not bool((total > 0).all()):
        raise ValueError('invalid behavior probabilities')
    return (safe_q * p / total).sum(dim=-1)


def return_advantage(returns, baseline):
    residual = np.asarray(returns, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)
    if not np.isfinite(residual).all():
        raise ValueError('non-finite return advantage')
    return (residual - residual.mean()) / (residual.std() + 1e-8)


class ActionValue(nn.Module):
    def __init__(self, global_dim, observation_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(global_dim + observation_dim, 128),
                                 nn.Tanh(), nn.Linear(128, 128), nn.Tanh(),
                                 nn.Linear(128, action_dim))
    def forward(self, global_state, observation):
        return self.net(torch.cat((global_state, observation), dim=-1))


class ReturnCritic:
    def __init__(self, global_dim, observation_dim, action_dim, lr=3e-4, device='cpu'):
        self.model = ActionValue(global_dim, observation_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.updates = 0
        self.quality_streak = 0
        self.ready = False

    def observe_quality(self, q_mse, value_mse):
        # Two consecutive pre-fit held-out checks; regressions immediately fall
        # back to the existing global GAE baseline. No validation/holdout data.
        useful = np.isfinite(q_mse) and q_mse < .95 * value_mse
        self.quality_streak = self.quality_streak + 1 if useful else 0
        self.ready = self.quality_streak >= 2

    def fit(self, g, obs, actions, returns, epochs=4):
        for _ in range(epochs):
            predicted = self.model(g, obs).gather(1, actions[:, None]).squeeze(1)
            loss = F.mse_loss(predicted, returns)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('non-finite A4 return critic loss')
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
            self.optimizer.step()
        self.updates += 1
        return float(loss.detach())

    def state_dict(self):
        return dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                    updates=self.updates, quality_streak=self.quality_streak, ready=self.ready)

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.updates = int(state['updates'])
        self.quality_streak = int(state['quality_streak'])
        self.ready = bool(state['ready'])


def publish_selected_bundle(root, episode, reward, decoder, key):
    """Publish weights and their own SLA/decoder sidecars through one pointer."""
    import json
    import os
    from pathlib import Path
    import shutil
    import tempfile
    root = Path(root)
    bundle = Path(tempfile.mkdtemp(prefix=f'selected_ep{episode:03d}_', dir=root))
    pointer = root / (bundle.name + '.link')
    try:
        for role in ('a1', 'a2', 'a3', 'a4', 'critic'):
            shutil.copy2(root / f'{role}_gang_best.pt', bundle / f'{role}_gang.pt')
        for sidecar in ('sla_multipliers.json', 'deployment_decoder.json', 'metric_manifest.json'):
            shutil.copy2(root / sidecar, bundle / sidecar)
        (bundle / 'selection.json').write_text(json.dumps(dict(episode=episode, reward=reward,
            decoder=decoder, protocol='train-1000-1010-50pct', key=key), indent=2))
        pointer.symlink_to(bundle.name, target_is_directory=True)
        os.replace(pointer, root / 'selected')
    except BaseException:
        pointer.unlink(missing_ok=True)
        shutil.rmtree(bundle)
        raise
