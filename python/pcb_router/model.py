"""DualStreamRouter: dual-stream actor-critic for gridless PCB routing.

Logical stream : dense message-passing GNN over the netlist (nodes = pads,
                 edges = required nets). Pure torch -- no torch_geometric --
                 so it runs on a stock Colab instance with zero installs.
Physical stream: PointNet-style encoder over the egocentric point cloud.
Fusion         : concat -> MLP trunk -> masked hybrid actor + scalar critic.

Action tuple (contract shared with masker.py / config.py):
    action_type : Categorical over {EXTEND, PLACE_VIA, COMMIT_NET}
    angle_bin   : Categorical over N_ANGLE_BINS directions (masked; indices
                  are in the target-aligned canonical frame -- bin 0 points
                  at the target, see masker.py)
    dist_frac   : Categorical over N_DIST_BINS distance steps, conditioned on
                  the sampled angle_bin (autoregressive -- step length may
                  depend on direction, e.g. "full step at the target,
                  shorter when skirting an obstacle")
    layer       : Categorical over MAX_LAYERS via targets (masked)
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .config import (A_EXTEND, A_VIA, HEAD_FEAT_DIM, MAX_LAYERS,
                     N_ACTION_TYPES, N_ANGLE_BINS, NODE_FEAT_DIM,
                     POINT_FEAT_DIM, N_DIST_BINS)

MASK_FILL = -1e9


class RouterAction(NamedTuple):
    action_type: torch.Tensor   # (B,) long
    angle_bin: torch.Tensor     # (B,) long
    dist_frac: torch.Tensor     # (B,) long (bin index)
    layer: torch.Tensor         # (B,) long


def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
    """A fully-masked categorical (e.g. no via anywhere) would produce NaNs;
    fall back to uniform -- the env ignores unused sub-actions anyway."""
    empty = mask.sum(-1, keepdim=True) == 0
    return torch.where(empty, torch.ones_like(mask), mask)


class DenseGNN(nn.Module):
    """Message passing over a padded dense adjacency (N_MAX_PINS is small)."""

    def __init__(self, in_dim: int, hidden: int, rounds: int = 3):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.msg = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(rounds))
        self.upd = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(rounds))
        self.norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(rounds))

    def forward(self, x, adj, node_mask):
        # x (B,N,F), adj (B,N,N), node_mask (B,N)
        eye = torch.eye(adj.size(1), device=adj.device).unsqueeze(0)
        a = (adj + eye) * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        a = a / a.sum(-1, keepdim=True).clamp(min=1.0)          # row-normalized
        h = self.proj(x)
        for msg, upd, norm in zip(self.msg, self.upd, self.norm):
            m = torch.bmm(a, msg(h))
            h = norm(h + torch.relu(upd(torch.cat([h, m], -1))))
        return h * node_mask.unsqueeze(-1)                      # (B,N,H)


class BoardPointNet(nn.Module):
    """Shared MLP + symmetric max-pool over the egocentric point cloud."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, points, point_mask):
        feats = self.mlp(points)                                # (B,P,H)
        feats = feats.masked_fill(point_mask.unsqueeze(-1) == 0, -1e9)
        pooled = feats.max(dim=1).values
        # Terminal states can have zero points; zero the pooled feature there.
        has_pts = (point_mask.sum(-1, keepdim=True) > 0).float()
        return pooled * has_pts


class ActorRouter(nn.Module):
    def __init__(self, hidden: int = 256, trunk_hidden: int = 512):
        super().__init__()
        self.gnn = DenseGNN(NODE_FEAT_DIM, hidden)
        self.pointnet = BoardPointNet(POINT_FEAT_DIM, hidden)
        self.head_proj = nn.Linear(HEAD_FEAT_DIM, hidden)

        # Fusion: [graph_global | current_net_embed | board_global | head]
        self.trunk = nn.Sequential(
            nn.Linear(4 * hidden, trunk_hidden), nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(),
        )

        self.type_head = nn.Linear(trunk_hidden, N_ACTION_TYPES)
        self.angle_head = nn.Linear(trunk_hidden, N_ANGLE_BINS)
        # Distance is autoregressive on the chosen angle: dist logits see an
        # embedding of the angle bin actually taken, so the factored policy
        # can express angle-dependent step lengths.
        self.angle_emb = nn.Embedding(N_ANGLE_BINS, 32)
        self.dist_head = nn.Linear(trunk_hidden + 32, N_DIST_BINS)
        self.layer_head = nn.Linear(trunk_hidden, MAX_LAYERS)

    def _fuse(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_h = self.gnn(obs["node_feats"], obs["adj"], obs["node_mask"])
        nm = obs["node_mask"].unsqueeze(-1)
        graph_h = (node_h * nm).sum(1) / nm.sum(1).clamp(min=1.0)
        cm = obs["cur_net_mask"].unsqueeze(-1)
        net_h = (node_h * cm).sum(1) / cm.sum(1).clamp(min=1.0)
        board_h = self.pointnet(obs["points"], obs["point_mask"])
        head_h = torch.relu(self.head_proj(obs["head_state"]))
        return self.trunk(torch.cat([graph_h, net_h, board_h, head_h], -1))

    def _dists(self, z: torch.Tensor, masks: Dict[str, torch.Tensor]):
        t_mask = _safe_mask(masks["type"])
        a_mask = _safe_mask(masks["angle"])
        l_mask = _safe_mask(masks["layer"])
        return {
            "type": Categorical(logits=self.type_head(z) + (1 - t_mask) * MASK_FILL),
            "angle": Categorical(logits=self.angle_head(z) + (1 - a_mask) * MASK_FILL),
            "layer": Categorical(logits=self.layer_head(z) + (1 - l_mask) * MASK_FILL),
        }

    def _dist_dist(self, z: torch.Tensor, angle_bin: torch.Tensor) -> Categorical:
        """Distance distribution conditioned on the angle actually taken."""
        return Categorical(logits=self.dist_head(
            torch.cat([z, self.angle_emb(angle_bin)], -1)))

    def _joint_logp(self, d, dd: Categorical, a: RouterAction) -> torch.Tensor:
        is_extend = (a.action_type == A_EXTEND).float()
        is_via = (a.action_type == A_VIA).float()
        return (d["type"].log_prob(a.action_type)
                + is_extend * (d["angle"].log_prob(a.angle_bin)
                               + dd.log_prob(a.dist_frac))
                + is_via * d["layer"].log_prob(a.layer))

    def act(self, obs, masks, deterministic: bool = False) -> Tuple[RouterAction, torch.Tensor]:
        z = self._fuse(obs)
        d = self._dists(z, masks)
        if deterministic:
            a_type = d["type"].probs.argmax(-1)
            angle = d["angle"].probs.argmax(-1)
            layer = d["layer"].probs.argmax(-1)
        else:
            a_type = d["type"].sample()
            angle = d["angle"].sample()
            layer = d["layer"].sample()
        dd = self._dist_dist(z, angle)
        dist = dd.probs.argmax(-1) if deterministic else dd.sample()
        a = RouterAction(action_type=a_type, angle_bin=angle,
                         dist_frac=dist, layer=layer)
        return a, self._joint_logp(d, dd, a)

    def evaluate_actions(self, obs, masks, action: RouterAction):
        z = self._fuse(obs)
        d = self._dists(z, masks)
        # Condition on the stored angle -- the same one act() sampled, so
        # act/evaluate log-probs agree exactly.
        dd = self._dist_dist(z, action.angle_bin)
        p = d["type"].probs
        # dd.entropy() is H(dist | taken angle): a per-sample estimate of the
        # conditional entropy, the standard choice for autoregressive heads.
        entropy = (d["type"].entropy()
                   + p[..., A_EXTEND] * (d["angle"].entropy() + dd.entropy())
                   + p[..., A_VIA] * d["layer"].entropy())
        return self._joint_logp(d, dd, action), entropy


class CriticRouter(nn.Module):
    def __init__(self, hidden: int = 256, trunk_hidden: int = 512):
        super().__init__()
        self.gnn = DenseGNN(NODE_FEAT_DIM, hidden)
        self.pointnet = BoardPointNet(POINT_FEAT_DIM, hidden)
        self.head_proj = nn.Linear(HEAD_FEAT_DIM, hidden)

        # Fusion: [graph_global | current_net_embed | board_global | head]
        self.trunk = nn.Sequential(
            nn.Linear(4 * hidden, trunk_hidden), nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(),
        )

        self.critic = nn.Sequential(
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(),
            nn.Linear(trunk_hidden, 1),
        )

    def _fuse(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_h = self.gnn(obs["node_feats"], obs["adj"], obs["node_mask"])
        nm = obs["node_mask"].unsqueeze(-1)
        graph_h = (node_h * nm).sum(1) / nm.sum(1).clamp(min=1.0)
        cm = obs["cur_net_mask"].unsqueeze(-1)
        net_h = (node_h * cm).sum(1) / cm.sum(1).clamp(min=1.0)
        board_h = self.pointnet(obs["points"], obs["point_mask"])
        head_h = torch.relu(self.head_proj(obs["head_state"]))
        return self.trunk(torch.cat([graph_h, net_h, board_h, head_h], -1))

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = self._fuse(obs)
        return self.critic(z).squeeze(-1)


class DualStreamRouter(nn.Module):
    def __init__(self, hidden: int = 256, trunk_hidden: int = 512):
        super().__init__()
        self.actor = ActorRouter(hidden, trunk_hidden)
        self.critic = CriticRouter(hidden, trunk_hidden)

    def act(self, obs, masks,
            deterministic: bool = False) -> Tuple[RouterAction, torch.Tensor, torch.Tensor]:
        """Sample an action (or take the mode, for evaluation/demos).
        Returns (action, joint_log_prob, value)."""
        a, logp = self.actor.act(obs, masks, deterministic)
        val = self.critic(obs)
        return a, logp, val

    def evaluate_actions(self, obs, masks, action: RouterAction):
        """Log-prob / entropy / value for PPO's surrogate loss."""
        logp, entropy = self.actor.evaluate_actions(obs, masks, action)
        val = self.critic(obs)
        return logp, entropy, val
