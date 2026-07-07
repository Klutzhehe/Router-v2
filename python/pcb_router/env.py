"""RoutingEnv: the headless CAD engine as an RL environment.

Gym-style API over the gridless board. One net is routed at a time (net order
= ascending HPWL). Observations are fixed-size padded tensors (see config.py)
so PPO batching is trivial. Rewards implement docs/reward-function.md exactly;
the 2.5D field-solver terms enter through the PhysicsEvaluator hook.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from . import geometry as geo
from .board import Board, KIND_PAD
from .config import (A_COMMIT, A_EXTEND, A_VIA, EnvConfig, HEAD_FEAT_DIM,
                     MAX_LAYERS, N_ANGLE_BINS, N_MAX_PINS, NODE_FEAT_DIM,
                     P_MAX, POINT_FEAT_DIM)
from .masker import ActionMask, ActionMasker, RoutingHead

_DIRS = geo.unit_dirs(N_ANGLE_BINS)


class PhysicsEvaluator:
    """API hook for a 2.5D electromagnetic field solver.

    Called once at episode end with the fully routed board. The default
    implementation returns zeros; a real solver (or learned surrogate) plugs
    in here without touching the environment.
    """

    def evaluate(self, board: Board, completed: list) -> Dict[str, float]:
        return {"impedance": 0.0, "skew": 0.0, "crosstalk": 0.0}


class RoutingEnv:
    def __init__(self, board_factory: Callable[[np.random.Generator], Board],
                 cfg: Optional[EnvConfig] = None, seed: int = 0,
                 physics: Optional[PhysicsEvaluator] = None):
        self.factory = board_factory
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.physics = physics or PhysicsEvaluator()

        self.board: Board = None
        self.masker: ActionMasker = None
        self.head: Optional[RoutingHead] = None
        self.mask: Optional[ActionMask] = None

    # ------------------------------------------------------------ lifecycle
    def reset(self):
        self.board = self.factory(self.rng)
        self.masker = ActionMasker(self.board)
        # Route easy (short-HPWL) nets first.
        self.order = sorted(range(len(self.board.nets)),
                            key=lambda i: self.board.nets[i].hpwl)
        self.cur = 0
        self.completed: list[int] = []
        self.drc_count = 0
        self.ep_steps = 0
        self.done = False
        self._start_current_net()
        self._skip_dead_nets()
        return self._obs(), self._mask_arrays()

    def _start_current_net(self):
        net = self.board.nets[self.order[self.cur]]
        pa, pb = self.board.pads[net.pins[0]], self.board.pads[net.pins[1]]
        self.head = RoutingHead(
            x=pa.x, y=pa.y, layer=pa.layer_lo, net_id=self.order[self.cur],
            half_width=self.board.rules.trace_width / 2.0,
            target_x=pb.x, target_y=pb.y, target_pad=net.pins[1])
        self.budget = self.cfg.max_steps_per_net
        self.mask = self.masker.compute_mask(self.head, self.cfg.lookahead)

    def _advance_net(self, completed: bool):
        if completed:
            self.completed.append(self.order[self.cur])
        self.cur += 1
        if self.cur >= len(self.order):
            self.done = True
            self.head = None
            self.mask = None
        else:
            self._start_current_net()

    def _skip_dead_nets(self):
        """Fail forward past nets with no legal action at all."""
        while not self.done and self.mask.type_mask.sum() == 0:
            self._advance_net(completed=False)

    # ----------------------------------------------------------------- MDP
    def _phi(self) -> float:
        """Potential: -(distance to target)/HPWL of the current net."""
        if self.done or self.head is None:
            return 0.0
        net = self.board.nets[self.order[self.cur]]
        d = np.hypot(self.head.x - self.head.target_x,
                     self.head.y - self.head.target_y)
        return -d / net.hpwl

    def step(self, action: Tuple[int, int, float, int]):
        assert not self.done, "call reset() first"
        a_type, angle_bin, dist_frac, layer = action
        rw, rules = self.cfg.reward, self.board.rules
        net = self.board.nets[self.order[self.cur]]
        hpwl = net.hpwl
        cur_before = self.cur
        phi_before = self._phi()
        r = 0.0

        if a_type == A_EXTEND and self.mask.type_mask[A_EXTEND] \
                and self.mask.angle_mask[angle_bin]:
            dmax = self.mask.max_distance[angle_bin]
            dist = rules.min_segment_length + float(np.clip(dist_frac, 0, 1)) \
                * (dmax - rules.min_segment_length)
            nx = self.head.x + dist * _DIRS[angle_bin, 0]
            ny = self.head.y + dist * _DIRS[angle_bin, 1]
            self.board.add_trace(self.head.x, self.head.y, nx, ny,
                                 self.head.half_width, self.head.layer,
                                 self.head.net_id)
            self.head.x, self.head.y = nx, ny
            r -= rw.lam1 * dist / hpwl

        elif a_type == A_VIA and self.mask.type_mask[A_VIA] \
                and self.mask.layer_mask[layer]:
            lo, hi = min(self.head.layer, layer), max(self.head.layer, layer)
            self.board.add_via(self.head.x, self.head.y, lo, hi, self.head.net_id)
            self.head.layer = int(layer)
            r -= rw.lam2

        elif a_type == A_COMMIT and self.mask.type_mask[A_COMMIT]:
            d = np.hypot(self.head.x - self.head.target_x,
                         self.head.y - self.head.target_y)
            if d > 1e-9:
                self.board.add_trace(self.head.x, self.head.y,
                                     self.head.target_x, self.head.target_y,
                                     self.head.half_width, self.head.layer,
                                     self.head.net_id)
                r -= rw.lam1 * d / hpwl
            r += rw.C
            self._advance_net(completed=True)

        else:
            # Masking should make this unreachable; if it fires, it is a
            # geometry-kernel bug (see docs/reward-function.md).
            self.drc_count += 1
            r -= rw.D

        self.ep_steps += 1

        # Per-net budget and stuck handling.
        if not self.done and self.head is not None:
            if a_type != A_COMMIT:
                self.budget -= 1
                self.mask = self.masker.compute_mask(self.head, self.cfg.lookahead)
                if self.budget <= 0:
                    self._advance_net(completed=False)
            self._skip_dead_nets()

        # Terminal reward.
        if self.done:
            n, nc = len(self.board.nets), len(self.completed)
            r += rw.B * nc / n
            if nc < n:
                r -= rw.F
            phys = self.physics.evaluate(self.board, self.completed)
            r -= rw.lam3 * phys["impedance"] + rw.lam4 * phys["skew"] \
                + rw.lam5 * phys["crosstalk"]

        # Potential-based shaping -- but ONLY within a net, never across a
        # net boundary. Phi is defined relative to whichever net is "current"
        # (see _phi), so per-net trajectories each have their own potential
        # function and the boundary transition has no well-defined PBRS term:
        #   * Evaluating phi_after on the NEXT net leaks its unrelated
        #     geometry into this step's reward (clawed back ~2.3 of every
        #     COMMIT's C when a net remained).
        #   * The textbook fix, phi_after = 0 at the boundary, refunds
        #     +beta*d_end/HPWL at every budget timeout -- a reward for being
        #     FAR from the target when the net is abandoned (measured +3.3
        #     for one wall-hugging net; up to ~+28 on stage 0 for short
        #     nets). Invariant in exact theory, but with GAE(lambda) and an
        #     imperfect critic the concentrated spike gets credited to the
        #     wander-away actions before it, teaching wall-hugging.
        # So: skip the term entirely at boundaries. The un-refunded residual
        # is equivalent to a terminal penalty of beta*d_end/HPWL on each
        # unfinished net -- failure graded by how close the head got, which
        # is the gradient a flat F cannot provide. Near-target commits have
        # ~zero residual, so completions still pay their full C.
        if not (self.done or self.cur != cur_before):
            r += rw.beta * (rw.gamma * self._phi() - phi_before)

        info = {"nets_done": len(self.completed),
                "nets_total": len(self.board.nets),
                "drc": self.drc_count, "steps": self.ep_steps}
        return self._obs(), self._mask_arrays(), float(r), self.done, info

    # ---------------------------------------------------------- observation
    def _mask_arrays(self) -> Dict[str, np.ndarray]:
        if self.mask is None:   # terminal state: nothing is legal
            return {"type": np.zeros(3, np.float32),
                    "angle": np.zeros(N_ANGLE_BINS, np.float32),
                    "layer": np.zeros(MAX_LAYERS, np.float32)}
        return {"type": self.mask.type_mask.astype(np.float32),
                "angle": self.mask.angle_mask.astype(np.float32),
                "layer": self.mask.layer_mask.astype(np.float32)}

    def _obs(self) -> Dict[str, np.ndarray]:
        b = self.board
        cur_net = self.order[self.cur] if not self.done else -1

        # --- Logical stream: netlist graph -----------------------------------
        node_feats = np.zeros((N_MAX_PINS, NODE_FEAT_DIM), np.float32)
        node_mask = np.zeros(N_MAX_PINS, np.float32)
        cur_net_mask = np.zeros(N_MAX_PINS, np.float32)
        adj = np.zeros((N_MAX_PINS, N_MAX_PINS), np.float32)
        denom_l = max(b.num_layers - 1, 1)
        for i, p in enumerate(b.pads[:N_MAX_PINS]):
            net = b.nets[p.net_id]
            node_feats[i] = [p.x / b.width, p.y / b.height, p.layer_lo / denom_l,
                            1.0 if p.net_id == cur_net else 0.0,
                            1.0 if p.net_id in self.completed else 0.0,
                            1.0 if (self.head and i == self.head.target_pad) else 0.0,
                            net.z_required / 100.0, net.signal_type / 2.0]
            node_mask[i] = 1.0
            if p.net_id == cur_net:
                cur_net_mask[i] = 1.0
        for net in b.nets:
            a_, b_ = net.pins
            if a_ < N_MAX_PINS and b_ < N_MAX_PINS:
                adj[a_, b_] = adj[b_, a_] = 1.0

        # --- Physical stream: egocentric point cloud (fully vectorized) ------
        points = np.zeros((P_MAX, POINT_FEAT_DIM), np.float32)
        point_mask = np.zeros(P_MAX, np.float32)
        if self.head is not None:
            hx, hy, hl = self.head.x, self.head.y, self.head.layer
            head_xy = np.array([hx, hy])
            W = self.cfg.obs_window
            arr = b.arrays()
            cols = []   # each: (dist, dxy(2), llo, lhi, kind, same, is_tgt, pri)

            if arr.disc_c.shape[0]:
                d = np.linalg.norm(arr.disc_c - head_xy, axis=1)
                s = d <= W + arr.disc_r
                cols.append((d[s], arr.disc_c[s] - head_xy,
                             arr.disc_llo[s], arr.disc_lhi[s], arr.disc_kind[s],
                             (arr.disc_net[s] == cur_net).astype(float),
                             np.zeros(s.sum()), np.ones(s.sum())))

            if arr.cap_a.shape[0]:
                # J samples per trace, evenly spread over each segment's own
                # length (short segments use fewer of the J slots).
                J = 8
                seg = arr.cap_b - arr.cap_a
                seg_len = np.linalg.norm(seg, axis=1)
                n_s = np.clip((seg_len / self.cfg.cap_sample_spacing).astype(int), 1, J - 1) + 1
                j = np.arange(J)[None, :]
                u = np.minimum(j / np.maximum(n_s[:, None] - 1, 1), 1.0)
                pts = arr.cap_a[:, None, :] + u[..., None] * seg[:, None, :]   # (M,J,2)
                dd = np.linalg.norm(pts - head_xy, axis=2)
                s = (j < n_s[:, None]) & (dd <= W)
                m_idx = np.nonzero(s)[0]
                cols.append((dd[s], pts[s] - head_xy,
                             arr.cap_layer[m_idx], arr.cap_layer[m_idx],
                             np.zeros(len(m_idx)),
                             (arr.cap_net[m_idx] == cur_net).astype(float),
                             np.zeros(len(m_idx)), np.ones(len(m_idx))))

            # Target pad is always visible, even outside the window (pri 0).
            tp = b.pads[self.head.target_pad]
            cols.append((np.zeros(1), np.array([[tp.x - hx, tp.y - hy]]),
                         np.array([tp.layer_lo]), np.array([tp.layer_hi]),
                         np.array([KIND_PAD]), np.ones(1), np.ones(1),
                         np.zeros(1)))

            dist = np.concatenate([c[0] for c in cols])
            dxy = np.concatenate([c[1] for c in cols])
            llo = np.concatenate([c[2] for c in cols])
            lhi = np.concatenate([c[3] for c in cols])
            kind = np.concatenate([c[4] for c in cols]).astype(int)
            same = np.concatenate([c[5] for c in cols])
            tgt = np.concatenate([c[6] for c in cols])
            pri = np.concatenate([c[7] for c in cols])

            order = np.lexsort((dist, pri))[:P_MAX]
            k = len(order)
            points[:k, 0:2] = dxy[order] / W
            points[:k, 2] = (llo[order] - hl) / MAX_LAYERS
            points[:k, 3] = (lhi[order] - hl) / MAX_LAYERS
            points[np.arange(k), 4 + kind[order]] = 1.0
            points[:k, 8] = same[order]
            points[:k, 9] = tgt[order]
            point_mask[:k] = 1.0

        # --- Routing-head state ----------------------------------------------
        head_state = np.zeros(HEAD_FEAT_DIM, np.float32)
        if self.head is not None:
            net = b.nets[cur_net]
            dx, dy = self.head.target_x - self.head.x, self.head.target_y - self.head.y
            d = np.hypot(dx, dy)
            head_state[0:2] = [self.head.x / b.width, self.head.y / b.height]
            head_state[2] = np.clip(dx / net.hpwl, -2, 2) / 2.0
            head_state[3] = np.clip(dy / net.hpwl, -2, 2) / 2.0
            head_state[4] = min(d / net.hpwl, 2.0) / 2.0
            head_state[5 + self.head.layer] = 1.0
            head_state[17] = len(self.completed) / max(len(b.nets), 1)
            head_state[18] = self.budget / self.cfg.max_steps_per_net

        return {"node_feats": node_feats, "adj": adj, "node_mask": node_mask,
                "cur_net_mask": cur_net_mask, "points": points,
                "point_mask": point_mask, "head_state": head_state}


class VecRoutingEnv:
    """N independent RoutingEnv instances stepped together so model.act sees
    a real batch instead of size-1 -- the fix for a GPU sitting nearly idle
    while training is bottlenecked on single-sample Python/CPU overhead.

    Each sub-env auto-resets individually the instant it finishes, same as
    collect_rollout does for a single RoutingEnv, so the batch always holds
    N valid in-progress transitions.
    """

    def __init__(self, board_factory: Callable[[np.random.Generator], Board],
                n_envs: int, cfg: Optional[EnvConfig] = None, seed: int = 0,
                physics: Optional[PhysicsEvaluator] = None):
        self.envs = [RoutingEnv(board_factory, cfg=cfg, seed=seed + i, physics=physics)
                    for i in range(n_envs)]
        self.n = n_envs

    def reset(self):
        obs, masks = zip(*(e.reset() for e in self.envs))
        return self._stack(obs), self._stack(masks)

    def step(self, actions):
        obs, masks, rewards, dones, infos = [], [], [], [], []
        for env, a in zip(self.envs, actions):
            o, m, r, d, info = env.step(a)
            if d:
                o, m = env.reset()
            obs.append(o); masks.append(m)
            rewards.append(r); dones.append(d); infos.append(info)
        return (self._stack(obs), self._stack(masks),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=bool), infos)

    @staticmethod
    def _stack(dicts):
        return {k: np.stack([d[k] for d in dicts]) for k in dicts[0]}
