"""Proximal Policy Optimization with GAE for the routing agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .model import DualStreamRouter, RouterAction

OBS_KEYS = ("node_feats", "adj", "node_mask", "cur_net_mask",
            "points", "point_mask", "head_state")
MASK_KEYS = ("type", "angle", "layer")


def to_torch(obs: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}


@dataclass
class PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 256
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5


class RolloutBuffer:
    def __init__(self, T: int, obs_spec: Dict[str, np.ndarray], device):
        self.T, self.device, self.i = T, device, 0
        self.obs = {k: torch.zeros((T, *v.shape), dtype=torch.float32)
                    for k, v in obs_spec.items()}
        self.masks = {"type": torch.zeros((T, 3)),
                      "angle": torch.zeros((T, 64)),
                      "layer": torch.zeros((T, 12))}
        self.a_type = torch.zeros(T, dtype=torch.long)
        self.a_angle = torch.zeros(T, dtype=torch.long)
        self.a_dist = torch.zeros(T)
        self.a_layer = torch.zeros(T, dtype=torch.long)
        self.logp = torch.zeros(T)
        self.value = torch.zeros(T)
        self.reward = torch.zeros(T)
        self.done = torch.zeros(T)

    def add(self, obs, masks, action: RouterAction, logp, value, reward, done):
        i = self.i
        for k in OBS_KEYS:
            self.obs[k][i] = torch.from_numpy(obs[k])
        for k in MASK_KEYS:
            self.masks[k][i] = torch.from_numpy(masks[k])
        self.a_type[i] = action.action_type
        self.a_angle[i] = action.angle_bin
        self.a_dist[i] = action.dist_frac
        self.a_layer[i] = action.layer
        self.logp[i], self.value[i] = logp, value
        self.reward[i], self.done[i] = reward, float(done)
        self.i += 1

    def compute_gae(self, last_value: float, cfg: PPOConfig):
        adv = torch.zeros(self.T)
        gae = 0.0
        for t in reversed(range(self.T)):
            nonterminal = 1.0 - self.done[t]
            next_v = last_value if t == self.T - 1 else self.value[t + 1]
            delta = self.reward[t] + cfg.gamma * next_v * nonterminal - self.value[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            adv[t] = gae
        self.adv = adv
        self.ret = adv + self.value


class PPO:
    def __init__(self, model: DualStreamRouter, cfg: PPOConfig = None,
                 device: str = "cpu"):
        self.model = model.to(device)
        self.cfg = cfg or PPOConfig()
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)

    def update(self, buf: RolloutBuffer) -> Dict[str, float]:
        cfg, dev = self.cfg, self.device
        T = buf.T
        adv = (buf.adv - buf.adv.mean()) / (buf.adv.std() + 1e-8)
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        n_updates = 0

        for _ in range(cfg.epochs):
            for idx in torch.randperm(T).split(cfg.minibatch):
                obs = {k: buf.obs[k][idx].to(dev) for k in OBS_KEYS}
                masks = {k: buf.masks[k][idx].to(dev) for k in MASK_KEYS}
                action = RouterAction(buf.a_type[idx].to(dev),
                                      buf.a_angle[idx].to(dev),
                                      buf.a_dist[idx].to(dev),
                                      buf.a_layer[idx].to(dev))
                logp, entropy, value = self.model.evaluate_actions(obs, masks, action)

                ratio = torch.exp(logp - buf.logp[idx].to(dev))
                a = adv[idx].to(dev)
                s1 = ratio * a
                s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = 0.5 * (value - buf.ret[idx].to(dev)).pow(2).mean()
                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy.mean()

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               cfg.max_grad_norm)
                self.opt.step()

                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                n_updates += 1

        return {k: v / max(n_updates, 1) for k, v in stats.items()}


def save_checkpoint(path, model: DualStreamRouter, ppo: "PPO", stage: int,
                    steps_done: int, completions, history: Optional[list] = None):
    """Full training state, not just weights: resuming from this must not
    reset Adam momentum, forget the curriculum stage, or lose step count."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": ppo.opt.state_dict(),
        "stage": stage,
        "steps_done": steps_done,
        "completions": list(completions),
        "history": history or [],
    }, tmp)
    Path(tmp).replace(path)   # atomic on the same filesystem: no half-written files


def load_checkpoint(path, model: DualStreamRouter, ppo: Optional["PPO"] = None,
                    device: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device)
    if "model" not in ckpt:      # back-compat: older checkpoints were a bare state_dict
        ckpt = {"model": ckpt, "stage": 0, "steps_done": 0, "completions": [], "history": []}
    model.load_state_dict(ckpt["model"])
    if ppo is not None and ckpt.get("optimizer"):
        ppo.opt.load_state_dict(ckpt["optimizer"])
    return ckpt


def collect_rollout(env, model: DualStreamRouter, T: int, device: str,
                    obs=None, masks=None):
    """Roll the policy for T steps (auto-resetting). Returns the filled
    buffer, episode stats, and the carried-over (obs, masks)."""
    if obs is None:
        obs, masks = env.reset()
    buf = RolloutBuffer(T, obs, device)
    ep_returns, ep_completions, ep_drc, ep_ret = [], [], [], 0.0
    commit_legal_steps = commit_taken_steps = 0

    for _ in range(T):
        t_obs = to_torch(obs, device)
        t_masks = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                   for k, v in masks.items()}
        action, logp, value = model.act(t_obs, t_masks)
        a = (int(action.action_type), int(action.angle_bin),
             float(action.dist_frac), int(action.layer))
        if masks["type"][2]:                      # A_COMMIT legal this step
            commit_legal_steps += 1
            if a[0] == 2:
                commit_taken_steps += 1
        next_obs, next_masks, reward, done, info = env.step(a)
        squeezed = RouterAction(*(x.squeeze(0) for x in action))
        buf.add(obs, masks, squeezed, logp.item(), value.item(), reward, done)
        ep_ret += reward
        if done:
            ep_returns.append(ep_ret)
            ep_completions.append(info["nets_done"] / info["nets_total"])
            ep_drc.append(info["drc"])
            ep_ret = 0.0
            next_obs, next_masks = env.reset()
        obs, masks = next_obs, next_masks

    with torch.no_grad():
        _, _, last_v = model.act(to_torch(obs, device),
                                 {k: torch.from_numpy(v).unsqueeze(0).to(device)
                                  for k, v in masks.items()})
    buf.compute_gae(float(last_v.item()), PPOConfig())
    stats = {
        "returns": ep_returns, "completions": ep_completions, "drc": ep_drc,
        # When COMMIT is legal (head within snap distance of target), how
        # often does the policy actually take it vs. keep extending past it?
        "commit_rate": (commit_taken_steps / commit_legal_steps
                        if commit_legal_steps else float("nan")),
        "commit_legal_steps": commit_legal_steps,
    }
    return buf, stats, (obs, masks)
