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
    epochs: int = 2
    minibatch: int = 512
    lr: float = 5e-5          # lowered from 1e-4 for gentler PPO updates
    vf_coef: float = 0.5
    vf_clip: float = 10.0     # value-clip range in RETURN units (~ one net
                              # completion C); NOT the 0.2 policy-ratio clip --
                              # see the note by the value-clipping code below
    ent_coef: float = 0.02    # raised from 0.01 to encourage exploration and prevent collapse
    ent_coef_min: float = 0.003  # minimum entropy coefficient (annealed to this)
    ent_anneal_steps: int = 500000  # steps over which to anneal entropy
    max_grad_norm: float = 0.5


class RolloutBuffer:
    def __init__(self, T: int, obs_spec: Dict[str, np.ndarray], device):
        self.T, self.device, self.i = T, device, 0
        self.obs = {k: torch.zeros((T, *v.shape), dtype=torch.float32)
                    for k, v in obs_spec.items()}
        self.masks = {"type": torch.zeros((T, 3)),
                      "angle": torch.zeros((T, 128)),
                      "layer": torch.zeros((T, 12))}
        self.a_type = torch.zeros(T, dtype=torch.long)
        self.a_angle = torch.zeros(T, dtype=torch.long)
        self.a_dist = torch.zeros(T, dtype=torch.long)
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

    def get_ent_coef(self, steps_done: int) -> float:
        """Anneal entropy coefficient over time to sharpen policy decisions."""
        cfg = self.cfg
        progress = min(steps_done / cfg.ent_anneal_steps, 1.0)
        return cfg.ent_coef * (1.0 - progress) + cfg.ent_coef_min * progress

    def update(self, buf: RolloutBuffer, steps_done: int = 0) -> Dict[str, float]:
        cfg, dev = self.cfg, self.device
        T = buf.T
        ent_coef = self.get_ent_coef(steps_done)

        # Print epoch-level debugging statistics
        type_fracs = [float((buf.a_type == i).float().mean()) for i in range(3)]
        print(f"  [DEBUG UPDATE] Buffer size T={T} | "
              f"Action type fracs: EXTEND={type_fracs[0]:.2%}, VIA={type_fracs[1]:.2%}, COMMIT={type_fracs[2]:.2%} | "
              f"Mean Reward: {float(buf.reward.mean()):.4f} | "
              f"Advantage Mean: {float(buf.adv.mean()):.4f}, Std: {float(buf.adv.std()):.4f} | "
              f"Value Mean: {float(buf.value.mean()):.4f}, Std: {float(buf.value.std()):.4f} | "
              f"ent_coef: {ent_coef:.4f}", flush=True)

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

                # PPO2-style value clipping, in RETURN units. The first version
                # of this reused the policy's 0.2 ratio clip -- but 0.2 is a
                # *ratio* bound, and applied to values (which span tens of
                # units here: C=+10 per net, terminal B/F) it capped the critic
                # to +-0.2 of movement per update. The critic could never learn
                # that being near a target is worth ~C, which starved the
                # policy of its only dense credit signal. vf_clip=10 (one net
                # completion) still stops a single freak batch from yanking
                # the critic across the whole return range, but lets it track
                # the task's actual value scale.
                old_value = buf.value[idx].to(dev)
                ret = buf.ret[idx].to(dev)
                value_clipped = old_value + (value - old_value).clamp(-cfg.vf_clip, cfg.vf_clip)
                v_loss = 0.5 * torch.max((value - ret).pow(2),
                                         (value_clipped - ret).pow(2)).mean()
                loss = pi_loss + cfg.vf_coef * v_loss - ent_coef * entropy.mean()

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
                    steps_done: int, completions, history: Optional[list] = None,
                    consecutive_hits: int = 0, detour_factors=None):
    """Full training state, not just weights: resuming from this must not
    reset Adam momentum, forget the curriculum stage, or lose step count.
    consecutive_hits is the curriculum advance-gate streak (see train.py) --
    persisted so resuming a run doesn't silently forget an in-progress streak.
    detour_factors is the parallel rolling window for the efficiency half of
    that gate (completion rate alone doesn't catch a stage graduating a
    policy that "completes wastefully" -- see train.py)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": ppo.opt.state_dict(),
        "stage": stage,
        "steps_done": steps_done,
        "completions": list(completions),
        "history": history or [],
        "consecutive_hits": consecutive_hits,
        "detour_factors": list(detour_factors) if detour_factors is not None else [],
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
        # Adam's state_dict includes each param group's lr as of the save --
        # loading it silently reverts any PPOConfig.lr change made since then
        # (e.g. lowering it to fix an unstable run). Keep the momentum/variance
        # state but re-apply whatever lr the caller's PPO/PPOConfig asked for.
        for group in ppo.opt.param_groups:
            group["lr"] = ppo.cfg.lr
    return ckpt


def collect_rollout(env, model: DualStreamRouter, T: int, device: str,
                    obs=None, masks=None, cfg: Optional[PPOConfig] = None):
    """Roll the policy for T steps (auto-resetting). Returns the filled
    buffer, episode stats, and the carried-over (obs, masks)."""
    if obs is None:
        obs, masks = env.reset()
    buf = RolloutBuffer(T, obs, device)
    ep_returns, ep_completions, ep_drc, ep_nets_total, ep_detour, ep_ret = \
        [], [], [], [], [], 0.0
    commit_legal_steps = commit_taken_steps = 0

    with torch.no_grad():
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
            # Debug printing for the first few steps of the epoch (e.g. first 20 steps)
            if _ < 20 and env.head is not None:
                types = ["EXTEND", "VIA", "COMMIT"]
                print(f"  [DEBUG STEP {_}] Head: ({env.head.x:.2f}, {env.head.y:.2f}, L{env.head.layer}) "
                      f"Target: ({env.head.target_x:.2f}, {env.head.target_y:.2f}) "
                      f"Budget: {env.budget} | "
                      f"Action: {types[a[0]]} (angle bin {a[1]}, dist_frac {a[2]:.3f}, layer {a[3]}) | "
                      f"Reward: {reward:+.4f} | Masks: type={masks['type']}, "
                      f"angle_sum={masks['angle'].sum()}, layer_sum={masks['layer'].sum()}", flush=True)
            squeezed = RouterAction(*(x.squeeze(0) for x in action))
            buf.add(obs, masks, squeezed, logp.item(), value.item(), reward, done)
            ep_ret += reward
            if done:
                ep_returns.append(ep_ret)
                ep_completions.append(info["nets_done"] / info["nets_total"])
                ep_drc.append(info["drc"])
                ep_nets_total.append(info["nets_total"])
                ep_detour.append(info["detour_factor"])
                ep_ret = 0.0
                next_obs, next_masks = env.reset()
            obs, masks = next_obs, next_masks

        _, _, last_v = model.act(to_torch(obs, device),
                                 {k: torch.from_numpy(v).unsqueeze(0).to(device)
                                  for k, v in masks.items()})
    if cfg is None:
        cfg = PPOConfig()
    buf.compute_gae(float(last_v.item()), cfg)
    stats = {
        "returns": ep_returns,
        "completions": ep_completions,
        "drc": ep_drc,
        "nets_total": ep_nets_total,
        "detour_factor": ep_detour,
        # When COMMIT is legal (head within snap distance of target), how
        # often does the policy actually take it vs. keep extending past it?
        "commit_rate": (commit_taken_steps / commit_legal_steps
                        if commit_legal_steps else float("nan")),
        "commit_legal_steps": commit_legal_steps,
    }
    return buf, stats, (obs, masks)


class VecRolloutBuffer:
    """Same role as RolloutBuffer, but fed by VecRoutingEnv: N env-steps
    land in the buffer every outer loop iteration instead of 1, so
    model.act sees a real batch (the actual fix for GPU under-utilization
    -- see collect_rollout_vec).

    Stored internally as (steps_per_env, n_envs, ...) so GAE can be
    computed correctly per-env along the time axis (advantages must not
    leak across different environments' trajectories). After finalize()
    everything is flattened to (T, ...) with T = steps_per_env * n_envs
    under the *same* attribute names RolloutBuffer uses, so PPO.update
    works on either buffer type unmodified.
    """

    def __init__(self, steps_per_env: int, n_envs: int,
                obs_spec: Dict[str, np.ndarray], device):
        self.steps_per_env, self.n_envs, self.device = steps_per_env, n_envs, device
        S, N = steps_per_env, n_envs
        self._obs = {k: torch.zeros((S, N, *v.shape), dtype=torch.float32)
                    for k, v in obs_spec.items()}
        self._masks = {"type": torch.zeros((S, N, 3)),
                      "angle": torch.zeros((S, N, 128)),
                      "layer": torch.zeros((S, N, 12))}
        self._a_type = torch.zeros((S, N), dtype=torch.long)
        self._a_angle = torch.zeros((S, N), dtype=torch.long)
        self._a_dist = torch.zeros((S, N), dtype=torch.long)
        self._a_layer = torch.zeros((S, N), dtype=torch.long)
        self._logp = torch.zeros((S, N))
        self._value = torch.zeros((S, N))
        self._reward = torch.zeros((S, N))
        self._done = torch.zeros((S, N))
        self.t = 0

    def add(self, obs, masks, action: RouterAction, logp, value, reward, done):
        """obs/masks: dict of (N, ...) arrays; action fields/logp/value: (N,)
        tensors; reward/done: (N,) arrays -- one batched env-step."""
        t = self.t
        for k in OBS_KEYS:
            self._obs[k][t] = torch.from_numpy(obs[k])
        for k in MASK_KEYS:
            self._masks[k][t] = torch.from_numpy(masks[k])
        self._a_type[t] = action.action_type
        self._a_angle[t] = action.angle_bin
        self._a_dist[t] = action.dist_frac
        self._a_layer[t] = action.layer
        self._logp[t] = logp
        self._value[t] = value
        self._reward[t] = torch.from_numpy(reward).float()
        self._done[t] = torch.from_numpy(done.astype(np.float32))
        self.t += 1

    def finalize(self, last_value: torch.Tensor, cfg: PPOConfig):
        S, N = self.steps_per_env, self.n_envs
        adv = torch.zeros((S, N))
        gae = torch.zeros(N)
        for t in reversed(range(S)):
            nonterminal = 1.0 - self._done[t]
            next_v = last_value if t == S - 1 else self._value[t + 1]
            delta = self._reward[t] + cfg.gamma * next_v * nonterminal - self._value[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            adv[t] = gae
        ret = adv + self._value

        self.T = S * N
        self.obs = {k: v.reshape(self.T, *v.shape[2:]) for k, v in self._obs.items()}
        self.masks = {k: v.reshape(self.T, *v.shape[2:]) for k, v in self._masks.items()}
        self.a_type = self._a_type.reshape(self.T)
        self.a_angle = self._a_angle.reshape(self.T)
        self.a_dist = self._a_dist.reshape(self.T)
        self.a_layer = self._a_layer.reshape(self.T)
        self.logp = self._logp.reshape(self.T)
        self.value = self._value.reshape(self.T)
        self.reward = self._reward.reshape(self.T)
        self.done = self._done.reshape(self.T)
        self.adv = adv.reshape(self.T)
        self.ret = ret.reshape(self.T)


def collect_rollout_vec(vec_env, model: DualStreamRouter, steps_per_env: int,
                        device: str, obs=None, masks=None, cfg: Optional[PPOConfig] = None):
    """Vectorized counterpart to collect_rollout: steps N envs together each
    iteration so model.act processes a batch of N instead of 1. Same return
    shape as collect_rollout (buffer, stats, carried (obs, masks))."""
    n = vec_env.n
    if obs is None:
        obs, masks = vec_env.reset()
    buf = VecRolloutBuffer(steps_per_env, n, {k: v[0] for k, v in obs.items()}, device)
    ep_returns, ep_completions, ep_drc, ep_nets_total, ep_detour = [], [], [], [], []
    ep_ret = np.zeros(n, dtype=np.float32)
    commit_legal_steps = commit_taken_steps = 0

    with torch.no_grad():
        for _ in range(steps_per_env):
            t_obs = {k: torch.from_numpy(v).to(device) for k, v in obs.items()}
            t_masks = {k: torch.from_numpy(v).to(device) for k, v in masks.items()}
            action, logp, value = model.act(t_obs, t_masks)

            legal_commit = masks["type"][:, 2].astype(bool)
            commit_legal_steps += int(legal_commit.sum())
            taken_type = action.action_type.cpu().numpy()
            commit_taken_steps += int(((taken_type == 2) & legal_commit).sum())

            actions = [(int(action.action_type[i]), int(action.angle_bin[i]),
                       float(action.dist_frac[i]), int(action.layer[i])) for i in range(n)]
            next_obs, next_masks, rewards, dones, infos = vec_env.step(actions)
            # Debug printing for the first environment (env 0) for the first 20 steps
            if _ < 20 and vec_env.envs[0].head is not None:
                types = ["EXTEND", "VIA", "COMMIT"]
                env0 = vec_env.envs[0]
                print(f"  [DEBUG ENV0 STEP {_}] Head: ({env0.head.x:.2f}, {env0.head.y:.2f}, L{env0.head.layer}) "
                      f"Target: ({env0.head.target_x:.2f}, {env0.head.target_y:.2f}) "
                      f"Budget: {env0.budget} | "
                      f"Action: {types[actions[0][0]]} (angle bin {actions[0][1]}, dist_frac {actions[0][2]:.3f}, layer {actions[0][3]}) | "
                      f"Reward: {rewards[0]:+.4f} | Masks: type={masks['type'][0]}, "
                      f"angle_sum={masks['angle'][0].sum()}, layer_sum={masks['layer'][0].sum()}", flush=True)

            buf.add(obs, masks, action, logp, value, rewards, dones)
            ep_ret += rewards
            for i in range(n):
                if dones[i]:
                    ep_returns.append(float(ep_ret[i]))
                    ep_completions.append(infos[i]["nets_done"] / infos[i]["nets_total"])
                    ep_drc.append(infos[i]["drc"])
                    ep_nets_total.append(infos[i]["nets_total"])
                    ep_detour.append(infos[i]["detour_factor"])
                    ep_ret[i] = 0.0
            obs, masks = next_obs, next_masks

        t_obs = {k: torch.from_numpy(v).to(device) for k, v in obs.items()}
        t_masks = {k: torch.from_numpy(v).to(device) for k, v in masks.items()}
        _, _, last_v = model.act(t_obs, t_masks)
    if cfg is None:
        cfg = PPOConfig()
    buf.finalize(last_v.cpu(), cfg)
    stats = {
        "returns": ep_returns,
        "completions": ep_completions,
        "drc": ep_drc,
        "nets_total": ep_nets_total,
        "detour_factor": ep_detour,
        "commit_rate": (commit_taken_steps / commit_legal_steps
                        if commit_legal_steps else float("nan")),
        "commit_legal_steps": commit_legal_steps,
    }
    return buf, stats, (obs, masks)
