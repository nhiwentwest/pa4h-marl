"""
models.py — network definitions for the PA4H-MARL gang scheduler.

NOTE: the original models.py lived on the Lightning training workspace and was
not synced back before that studio expired. This file was reconstructed
(2026-08-01) from (a) the call sites in marl_gang_train.py / eval_gang.py,
(b) the ST-GNN building blocks of the earlier V8 codebase, and (c) the
architecture description in the paper (rack-level ST-GNN: dilated TCN branch +
edge-aware GATv2 branch, fused and scored by a rack-shared head). It is
interface-compatible with the training/eval scripts; bit-exact parity with the
checkpoints trained on the original file is not guaranteed.

Actors:
  Actor                — MLP policy used by A1 (WAIT/PREEMPT) and A3 (victim).
  RackSTGNNActor       — ST-GNN policy used by A2 (risk) and A4 (placement,
                         with DEFER head and feasibility/forced-DEFER
                         auxiliary heads).
  CentralizedCritic    — CTDE value network over the global state.

Every action selection goes through select_action(actor, obs, mask, mode):
  mode="train": stochastic sample, returns (action, log_prob)   -- PPO
  mode="eval" : deterministic argmax, returns (action, log_prob) -- deployment
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """MLP actor — sees only its local observation (decentralized execution)."""

    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, action_dim)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, x, mask=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.net(x)
        logits = self.head(h)
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    def get_action(self, obs, mask=None, deterministic=False):
        with torch.no_grad():
            logits = self.forward(obs, mask).squeeze(0)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = probs.argmax()
            else:
                action = torch.distributions.Categorical(probs).sample()
            log_prob = torch.log(probs[action] + 1e-8)
        return action.item(), log_prob.item()


class DilatedTCNBlock(nn.Module):
    """Small residual TCN block used by the ST-GNN temporal branch."""

    def __init__(self, channels, dilation, kernel_size=3):
        super().__init__()
        padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                              dilation=dilation, padding=padding)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        residual = x
        y = self.conv(x)
        y = y[..., : x.shape[-1]]        # drop causal-padding overshoot
        y = F.gelu(y)
        y = y + residual
        return self.norm(y.transpose(1, 2)).transpose(1, 2)


class EdgeAwareGATv2Layer(nn.Module):
    """Dependency-free GATv2-style message passing with edge attributes."""

    def __init__(self, hidden, edge_dim=3, heads=4):
        super().__init__()
        self.hidden = int(hidden)
        self.edge_dim = int(edge_dim)
        self.heads = max(1, int(heads))
        self.head_dim = self.hidden // self.heads
        if self.head_dim * self.heads != self.hidden:
            raise ValueError("hidden must be divisible by heads")
        self.src = nn.Linear(self.hidden, self.hidden, bias=False)
        self.dst = nn.Linear(self.hidden, self.hidden, bias=False)
        self.edge = nn.Linear(self.edge_dim, self.hidden, bias=False)
        self.attn = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.out = nn.Linear(self.hidden, self.hidden)
        self.norm = nn.LayerNorm(self.hidden)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, h, edge_index, edge_attr):
        # h: [B, R, H]; edge_index: [2, E]; edge_attr: [E, edge_dim]
        src_idx = edge_index[0].long()
        dst_idx = edge_index[1].long()
        q = self.dst(h[:, dst_idx]).unflatten(-1, (self.heads, self.head_dim))
        k = self.src(h[:, src_idx]).unflatten(-1, (self.heads, self.head_dim))
        e = self.edge(edge_attr).unflatten(-1, (self.heads, self.head_dim))
        pair = F.gelu(q + k + e.unsqueeze(0))
        scores = (pair * self.attn.unsqueeze(0).unsqueeze(0)).sum(-1)
        scores = scores / (self.head_dim ** 0.5)            # [B, E, heads]
        messages = k + e.unsqueeze(0)                       # [B, E, heads, hd]
        num_nodes = h.shape[1]
        out = h.new_zeros(h.shape[0], num_nodes, self.heads, self.head_dim)
        for node in range(num_nodes):
            sel = (dst_idx == node)
            if not bool(sel.any()):
                continue
            w = F.softmax(scores[:, sel], dim=1).unsqueeze(-1)
            out[:, node] = (w * messages[:, sel]).sum(dim=1)
        out = self.out(out.reshape(h.shape[0], num_nodes, self.hidden))
        return self.norm(h + F.gelu(out))


class RackSTGNNActor(nn.Module):
    """Rack-level spatio-temporal graph actor for A2/A4.

    Flat observation layout (matches gang_env/marl_gang_train):
      [ num_racks * history_len * feat_dim  ||  context_dim ]
    Output: one logit per rack (+ a DEFER logit when include_defer=True).
    """

    def __init__(self, num_racks, history_len, feat_dim, context_dim=0,
                 include_defer=False, hidden=64, heads=4, gat_layers=2):
        super().__init__()
        self.num_racks = int(num_racks)
        self.history_len = int(history_len)
        self.feat_dim = int(feat_dim)
        self.context_dim = int(context_dim)
        self.include_defer = bool(include_defer)
        self.hidden = int(hidden)
        self.graph_dim = self.num_racks * self.history_len * self.feat_dim
        self.obs_dim = self.graph_dim + self.context_dim
        self.action_dim = self.num_racks + (1 if self.include_defer else 0)

        self.input_proj = nn.Linear(self.feat_dim, self.hidden)
        self.tcn = nn.Sequential(
            DilatedTCNBlock(self.hidden, dilation=1),
            DilatedTCNBlock(self.hidden, dilation=2),
            DilatedTCNBlock(self.hidden, dilation=4),
        )
        self.spatial_proj = nn.Linear(self.feat_dim, self.hidden)
        self.gat_layers = nn.ModuleList(
            EdgeAwareGATv2Layer(self.hidden, edge_dim=3, heads=heads)
            for _ in range(int(gat_layers)))
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden * 2, self.hidden), nn.GELU(),
            nn.LayerNorm(self.hidden))
        self.global_mlp = nn.Sequential(
            nn.Linear(self.hidden, self.hidden), nn.GELU(),
            nn.LayerNorm(self.hidden))
        score_in = self.hidden * 2 + self.context_dim
        self.score_head = nn.Linear(score_in, 1)
        if self.include_defer:
            self.defer_head = nn.Sequential(
                nn.Linear(self.hidden + self.context_dim, self.hidden),
                nn.GELU(),
                nn.Linear(self.hidden, 1))
        # Auxiliary representation heads (A4): per-rack feasibility and
        # forced-DEFER prediction, trained with BCE after the PPO update.
        self.aux_feas_head = nn.Linear(score_in, 1)
        self.aux_forced_head = nn.Sequential(
            nn.Linear(self.hidden + self.context_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1))

        # Fully-connected directed rack graph through the shared aggregate
        # tier; every inter-rack edge carries the same normalized attribute.
        src, dst = [], []
        for q in range(self.num_racks):
            for r in range(self.num_racks):
                if q != r:
                    src.append(q)
                    dst.append(r)
        self.register_buffer("edge_index",
                             torch.tensor([src, dst], dtype=torch.long))
        self.register_buffer("edge_attr",
                             torch.ones(len(src), 3, dtype=torch.float32))

    def _encode(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        graph = x[:, :self.graph_dim].view(
            batch, self.num_racks, self.history_len, self.feat_dim)
        ctx = x[:, self.graph_dim:]
        t = self.input_proj(graph)                                # [B,R,L,H]
        t = t.reshape(batch * self.num_racks, self.history_len, self.hidden)
        t = self.tcn(t.transpose(1, 2))[:, :, -1]
        t = t.reshape(batch, self.num_racks, self.hidden)
        s = self.spatial_proj(graph[:, :, -1, :])
        for layer in self.gat_layers:
            s = layer(s, self.edge_index, self.edge_attr)
        z = self.fusion(torch.cat([s, t], dim=-1))                # [B,R,H]
        g = self.global_mlp(z.mean(dim=1))                        # [B,H]
        return z, g, ctx

    def forward(self, x, mask=None):
        z, g, ctx = self._encode(x)
        gc = torch.cat([g, ctx], dim=-1)
        per_rack = torch.cat(
            [z, gc.unsqueeze(1).expand(-1, self.num_racks, -1)], dim=-1)
        logits = self.score_head(per_rack).squeeze(-1)            # [B,R]
        if self.include_defer:
            defer = self.defer_head(gc)                           # [B,1]
            logits = torch.cat([logits, defer], dim=-1)
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    def auxiliary(self, x):
        """Return (per-rack feasibility logits [B,R], forced-DEFER logits [B])."""
        z, g, ctx = self._encode(x)
        gc = torch.cat([g, ctx], dim=-1)
        per_rack = torch.cat(
            [z, gc.unsqueeze(1).expand(-1, self.num_racks, -1)], dim=-1)
        feas = self.aux_feas_head(per_rack).squeeze(-1)
        forced = self.aux_forced_head(gc).squeeze(-1)
        return feas, forced

    def get_action(self, obs, mask=None, deterministic=False):
        with torch.no_grad():
            logits = self.forward(obs, mask).squeeze(0)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = probs.argmax()
            else:
                action = torch.distributions.Categorical(probs).sample()
            log_prob = torch.log(probs[action] + 1e-8)
        return action.item(), log_prob.item()


class CentralizedCritic(nn.Module):
    """Centralized critic sees the GLOBAL state during training (CTDE)."""

    def __init__(self, global_obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def select_action(actor, obs, mask=None, mode="train"):
    """Unified action selection for every layer.

    mode="train" -> stochastic sample (real log_prob for PPO);
    mode="eval"  -> deterministic argmax (deployment / evaluation).
    Returns (action: int, log_prob: float).
    """
    if mode not in ("train", "eval"):
        raise ValueError(f"select_action mode must be 'train' or 'eval', got {mode!r}")
    if isinstance(obs, np.ndarray):
        obs = torch.FloatTensor(obs).unsqueeze(0)
    if mask is not None and isinstance(mask, np.ndarray):
        mask = torch.BoolTensor(mask).unsqueeze(0)
    device = next(actor.parameters()).device
    obs = obs.to(device)
    if mask is not None:
        mask = mask.to(device)
    return actor.get_action(obs, mask, deterministic=(mode == "eval"))
