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
from .board import Board, KIND_PAD, KIND_KEEPOUT
from .config import (A_COMMIT, A_EXTEND, A_VIA, EnvConfig, HEAD_FEAT_DIM,
                     LAYER_ROLE_POWER, MAX_LAYERS, N_ANGLE_BINS, N_MAX_PINS,
                     NODE_FEAT_DIM, P_MAX, POINT_FEAT_DIM, DIST_FRACTIONS)
from .masker import ActionMask, ActionMasker, RoutingHead

_DIRS = geo.unit_dirs(N_ANGLE_BINS)





class PhysicsEvaluator:
    """API hook for a 2.5D electromagnetic field solver.

    Called once at episode end with the fully routed board. impedance/
    crosstalk stay stubbed at zero pending a real solver; skew is real --
    differential pairs (Net.pair_id) get their routed length measured
    directly from board.traces (the "length mismatch is nearly free to
    compute" note in PROJECT.md's roadmap).
    """

    def evaluate(self, board: Board, completed: list) -> Dict[str, float]:
        completed_set = set(completed)
        pair_lengths: Dict[int, Dict[int, float]] = {}
        for (ax, ay, bx, by, _hw, _layer, net_id) in board.traces:
            if net_id not in completed_set:
                continue
            pair_id = board.nets[net_id].pair_id
            if pair_id is None:
                continue
            lengths = pair_lengths.setdefault(pair_id, {})
            lengths[net_id] = lengths.get(net_id, 0.0) + float(np.hypot(bx - ax, by - ay))

        skews = []
        for lengths in pair_lengths.values():
            if len(lengths) == 2:
                v1, v2 = lengths.values()
                skews.append(abs(v1 - v2))
        skew = float(np.mean(skews)) if skews else 0.0
        return {"impedance": 0.0, "skew": skew, "crosstalk": 0.0}


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
        self.last_completion = None

    # ------------------------------------------------------------ lifecycle
    def reset(self):
        self.board = self.factory(self.rng)
        self.masker = ActionMasker(self.board)
        # Route easy (short-HPWL) nets first.
        order = sorted(range(len(self.board.nets)),
                       key=lambda i: self.board.nets[i].hpwl)
        self.order = self._splice_diff_pairs(order)
        self.cur = 0
        self.completed: list[int] = []
        self.last_completion = None
        self.drc_count = 0
        self.ep_steps = 0
        self.done = False
        self._start_current_net()
        self._skip_dead_nets()
        return self._obs(), self._mask_arrays()

    def _splice_diff_pairs(self, order: list) -> list:
        """Re-splice HPWL-sorted net order so each differential pair's two
        nets route back-to-back: the second net then sees the first net's
        copper as an obstacle, which is what naturally pulls it into a
        near-parallel path under the normal clearance rules."""
        nets = self.board.nets
        partner_of: Dict[int, int] = {}
        by_pair: Dict[int, list] = {}
        for idx in order:
            pid = nets[idx].pair_id
            if pid is not None:
                by_pair.setdefault(pid, []).append(idx)
        for idxs in by_pair.values():
            if len(idxs) == 2:
                partner_of[idxs[0]] = idxs[1]
                partner_of[idxs[1]] = idxs[0]

        result, placed = [], set()
        for idx in order:
            if idx in placed:
                continue
            result.append(idx)
            placed.add(idx)
            partner = partner_of.get(idx)
            if partner is not None and partner not in placed:
                result.append(partner)
                placed.add(partner)
        return result

    def _start_current_net(self):
        net = self.board.nets[self.order[self.cur]]
        pa, pb = self.board.pads[net.pins[0]], self.board.pads[net.pins[1]]
        init_angle = float(np.arctan2(pb.y - pa.y, pb.x - pa.x) % (2.0 * np.pi))
        self.head = RoutingHead(
            x=pa.x, y=pa.y, layer=pa.layer_lo, net_id=self.order[self.cur],
            half_width=net.trace_width / 2.0,
            target_x=pb.x, target_y=pb.y, target_pad=net.pins[1],
            prev_heading_angle=init_angle)
        self.budget = self.cfg.max_steps_per_net
        self.mask = self.masker.compute_mask(self.head, self.cfg.lookahead)

    def _advance_net(self, completed: bool):
        if completed:
            self.completed.append(self.order[self.cur])
        self.cur += 1
        if self.cur >= len(self.order):
            self.done = True
            self.last_completion = len(self.completed) / len(self.order)
            self.head = None
            self.mask = None
        else:
            self._start_current_net()

    def _skip_dead_nets(self):
        """Fail forward past nets with no legal action at all."""
        while not self.done and self.mask.type_mask.sum() == 0:
            self._advance_net(completed=False)

    def _simplify_trace(self, net_id: int) -> None:
        """Greedy polyline simplification: try removing intermediate trace points."""
        net_traces = [t for t in self.board.traces if t[6] == net_id]
        if not net_traces:
            return

        # Group net_traces into contiguous runs on the same layer.
        runs = []
        current_run = []
        for t in net_traces:
            if not current_run:
                current_run.append(t)
            else:
                last_t = current_run[-1]
                dist_connect = np.hypot(t[0] - last_t[2], t[1] - last_t[3])
                if t[5] == last_t[5] and dist_connect < 1e-6:
                    current_run.append(t)
                else:
                    runs.append(current_run)
                    current_run = [t]
        if current_run:
            runs.append(current_run)

        # Simplify each run using lookahead check
        new_traces = [t for t in self.board.traces if t[6] != net_id]
        for run in runs:
            hw = run[0][4]
            layer = run[0][5]
            V = [np.array([run[0][0], run[0][1]])]
            for t in run:
                V.append(np.array([t[2], t[3]]))

            simplified_V = []
            i = 0
            n_vertices = len(V)
            while i < n_vertices:
                simplified_V.append(V[i])
                if i == n_vertices - 1:
                    break
                shortcut_j = i + 1
                for j in range(n_vertices - 1, i + 1, -1):
                    if self.masker.segment_legal(V[i], V[j], layer, net_id, hw):
                        shortcut_j = j
                        break
                i = shortcut_j

            # Pull tight intermediate vertices to make them hug obstacles closely
            simplified_V = self._pull_tight(simplified_V, layer, net_id, hw)

            # Chamfer corners > 45 degrees
            simplified_V = self._chamfer_corners(simplified_V, layer, net_id, hw)

            for k in range(len(simplified_V) - 1):
                ax, ay = simplified_V[k]
                bx, by = simplified_V[k+1]
                new_traces.append((float(ax), float(ay), float(bx), float(by), hw, layer, net_id))

        self.board.traces = new_traces
        self.board._version += 1

    def _pull_tight(self, V: list[np.ndarray], layer: int, net_id: int, hw: float) -> list[np.ndarray]:
        if len(V) < 3:
            return V
        
        # We do 2 passes of relaxation
        for _ in range(2):
            for i in range(1, len(V) - 1):
                p_old = V[i]
                p_target = (V[i-1] + V[i+1]) / 2.0
                
                low = 0.0
                high = 1.0
                for _bin in range(8):
                    mid_t = (low + high) / 2.0
                    p = (1.0 - mid_t) * p_old + mid_t * p_target
                    if self.masker.segment_legal(V[i-1], p, layer, net_id, hw) \
                            and self.masker.segment_legal(p, V[i+1], layer, net_id, hw):
                        low = mid_t
                    else:
                        high = mid_t
                
                V[i] = (1.0 - low) * p_old + low * p_target
        return V

    def _chamfer_corners(self, V: list[np.ndarray], layer: int, net_id: int, hw: float) -> list[np.ndarray]:
        if len(V) < 3:
            return V
        
        i = 1
        while i < len(V) - 1:
            v0 = V[i-1]
            v1 = V[i]
            v2 = V[i+1]
            
            d1 = v1 - v0
            d2 = v2 - v1
            
            l1 = np.linalg.norm(d1)
            l2 = np.linalg.norm(d2)
            if l1 < 1e-6 or l2 < 1e-6:
                i += 1
                continue
                
            cos_theta = np.dot(d1, d2) / (l1 * l2)
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            
            # Turn angle > 45 degrees (pi / 4)
            if theta > np.pi / 4.0 + 1e-4:
                # We try to chamfer
                d = min(0.3, l1 / 2.1, l2 / 2.1)
                if d > 1e-4:
                    p1 = v1 - (d1 / l1) * d
                    p2 = v1 + (d2 / l2) * d
                    if self.masker.segment_legal(p1, p2, layer, net_id, hw):
                        V[i] = p1
                        V.insert(i + 1, p2)
                        # Don't increment i, so we check the turn at p1 next
                        continue
            i += 1
        return V

    def _mean_detour_factor(self) -> float:
        """Average (routed length / HPWL) over completed nets this episode --
        a direct efficiency metric distinct from completion rate. A pure
        straight-line route can score *below* 1.0 (Euclidean <= Manhattan =
        HPWL); values well above 1.0 mean wasted copper, not obstacle
        avoidance. NaN if nothing completed (nothing to measure)."""
        if not self.completed:
            return float("nan")
        completed_set = set(self.completed)
        lengths: Dict[int, float] = {}
        for (ax, ay, bx, by, _hw, _layer, net_id) in self.board.traces:
            if net_id in completed_set:
                lengths[net_id] = lengths.get(net_id, 0.0) + float(np.hypot(bx - ax, by - ay))
        factors = [lengths.get(nid, 0.0) / self.board.nets[nid].hpwl for nid in self.completed]
        return float(np.mean(factors))

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
            dist_idx = int(np.clip(dist_frac, 0, len(DIST_FRACTIONS) - 1))
            dist_val = DIST_FRACTIONS[dist_idx]
            dist = rules.min_segment_length + dist_val * (dmax - rules.min_segment_length)
            # angle_bin indexes the target-aligned canonical frame; recover world heading
            world_bin = (angle_bin + self.mask.frame_offset) % N_ANGLE_BINS
            heading_angle = 2.0 * np.pi * world_bin / N_ANGLE_BINS
            nx = self.head.x + dist * _DIRS[world_bin, 0]
            ny = self.head.y + dist * _DIRS[world_bin, 1]
            self.board.add_trace(self.head.x, self.head.y, nx, ny,
                                 self.head.half_width, self.head.layer,
                                 self.head.net_id)
            # Turn penalty: quadratic in the normalized angle, not linear --
            # a 90 deg corner costs 1/4 of a full reversal, not 1/2, so sharp
            # turns are singled out while gentle curve corrections (small
            # turn_frac, squared even smaller) stay nearly free. A linear
            # penalty here priced a 10 deg wobble and a 170 deg hairpin at
            # the same per-degree rate, which barely disciplined jagged
            # zigzag or sharp-cornered loops (see the wasteful-loop render).
            turn_delta = abs(heading_angle - self.head.prev_heading_angle)
            turn_delta = min(turn_delta, 2.0 * np.pi - turn_delta)  # shorter arc
            turn_frac = turn_delta / np.pi  # in [0, 1]
            r -= rw.lam_turn * turn_frac ** 2
            self.head.x, self.head.y = nx, ny
            self.head.prev_heading_angle = heading_angle
            self.head.just_placed_via = False
            r -= rw.lam1 * dist / hpwl
            # Stack-up penalty: legal, but a non-power net dwelling on a
            # dedicated power/ground plane isn't "supposed to be here".
            if net.signal_type != 1 and self.board.layer_roles[self.head.layer] == LAYER_ROLE_POWER:
                r -= rw.lam_stackup * dist / hpwl

        elif a_type == A_VIA and self.mask.type_mask[A_VIA] \
                and self.mask.layer_mask[layer]:
            lo, hi = min(self.head.layer, layer), max(self.head.layer, layer)
            self.board.add_via(self.head.x, self.head.y, lo, hi, self.head.net_id)
            self.head.layer = int(layer)
            self.head.just_placed_via = True
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
                if net.signal_type != 1 and self.board.layer_roles[self.head.layer] == LAYER_ROLE_POWER:
                    r -= rw.lam_stackup * d / hpwl
            r += rw.C
            self._simplify_trace(self.head.net_id)
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
        #
        # The difference is deliberately UNDISCOUNTED (no gamma on phi_after).
        # The textbook beta*(gamma*phi' - phi) form pays beta*(1-gamma)*|phi|
        # per step for standing still -- with phi <= 0 that is a positive
        # annuity proportional to distance-from-target (measured +0.0167/step
        # at d/HPWL=1.1; up to ~+0.10/step in a far corner on a short net,
        # i.e. a completion's worth of free reward per 96-step budget). The
        # gamma term only buys exact policy-invariance under full PBRS, which
        # the boundary rule above already forgoes; without the refund the
        # annuity is pure free income and taught far-wall loitering.
        if not (self.done or self.cur != cur_before):
            r += rw.beta * (self._phi() - phi_before)

        info = {"nets_done": len(self.completed),
                "nets_total": len(self.board.nets),
                "drc": self.drc_count, "steps": self.ep_steps,
                "detour_factor": self._mean_detour_factor() if self.done else float("nan")}
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
                            net.z_required / 100.0, net.signal_type / 2.0,
                            net.trace_width / 0.5, float(b.layer_roles[p.layer_lo])]
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

            # --- Add board boundaries as keep-out points to the point cloud ---
            # Project head position to the four walls and sample points along them
            # if they fall within the observation window W.
            # Left wall (x = 0)
            if hx < W:
                dy_max = np.sqrt(W**2 - hx**2)
                y_min, y_max = max(0.0, hy - dy_max), min(b.height, hy + dy_max)
                ys = np.arange(y_min, y_max + 0.1, 1.0)
                if len(ys) > 0:
                    cols.append((np.sqrt(hx**2 + (ys - hy)**2), np.stack([-np.full_like(ys, hx), ys - hy], axis=-1),
                                 np.zeros(len(ys)), np.full(len(ys), b.num_layers - 1),
                                 np.full(len(ys), KIND_KEEPOUT), np.zeros(len(ys)), np.zeros(len(ys)), np.ones(len(ys))))
            # Right wall (x = width)
            rx = b.width - hx
            if rx < W:
                dy_max = np.sqrt(W**2 - rx**2)
                y_min, y_max = max(0.0, hy - dy_max), min(b.height, hy + dy_max)
                ys = np.arange(y_min, y_max + 0.1, 1.0)
                if len(ys) > 0:
                    cols.append((np.sqrt(rx**2 + (ys - hy)**2), np.stack([np.full_like(ys, rx), ys - hy], axis=-1),
                                 np.zeros(len(ys)), np.full(len(ys), b.num_layers - 1),
                                 np.full(len(ys), KIND_KEEPOUT), np.zeros(len(ys)), np.zeros(len(ys)), np.ones(len(ys))))
            # Top wall (y = 0)
            if hy < W:
                dx_max = np.sqrt(W**2 - hy**2)
                x_min, x_max = max(0.0, hx - dx_max), min(b.width, hx + dx_max)
                xs = np.arange(x_min, x_max + 0.1, 1.0)
                if len(xs) > 0:
                    cols.append((np.sqrt(hy**2 + (xs - hx)**2), np.stack([xs - hx, -np.full_like(xs, hy)], axis=-1),
                                 np.zeros(len(xs)), np.full(len(xs), b.num_layers - 1),
                                 np.full(len(xs), KIND_KEEPOUT), np.zeros(len(xs)), np.zeros(len(xs)), np.ones(len(xs))))
            # Bottom wall (y = height)
            by = b.height - hy
            if by < W:
                dx_max = np.sqrt(W**2 - by**2)
                x_min, x_max = max(0.0, hx - dx_max), min(b.width, hx + dx_max)
                xs = np.arange(x_min, x_max + 0.1, 1.0)
                if len(xs) > 0:
                    cols.append((np.sqrt(by**2 + (xs - hx)**2), np.stack([xs - hx, np.full_like(xs, by)], axis=-1),
                                 np.zeros(len(xs)), np.full(len(xs), b.num_layers - 1),
                                 np.full(len(xs), KIND_KEEPOUT), np.zeros(len(xs)), np.zeros(len(xs)), np.ones(len(xs))))

            # Target pad is always visible, even outside the window (pri 0).
            tp = b.pads[self.head.target_pad]
            cols.append((np.zeros(1), np.array([[tp.x - hx, tp.y - hy]]),
                         np.array([tp.layer_lo]), np.array([tp.layer_hi]),
                         np.array([KIND_PAD]), np.ones(1), np.ones(1),
                         np.zeros(1)))

            dist = np.concatenate([c[0] for c in cols])
            dxy = np.concatenate([c[1] for c in cols])
            # Rotate offsets into the target-aligned canonical frame so the
            # cloud agrees with the rolled angle mask (canonical bin 0 = at
            # the target). Must use the SAME quantized offset the masker
            # rolled by, not the exact atan2 angle, or the observation and
            # the action space would disagree by up to half a bin.
            th = self.mask.frame_offset * (2.0 * np.pi / N_ANGLE_BINS)
            c_, s_ = np.cos(th), np.sin(th)
            dxy = dxy @ np.array([[c_, -s_], [s_, c_]])
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
            # Same canonical-frame rotation as the point cloud: dxc ~ d (the
            # remaining distance), dyc ~ the sub-bin residual -- the target
            # direction itself is baked into the frame, not a feature the
            # network has to decode.
            th = self.mask.frame_offset * (2.0 * np.pi / N_ANGLE_BINS)
            c_, s_ = np.cos(th), np.sin(th)
            dxc, dyc = dx * c_ + dy * s_, -dx * s_ + dy * c_
            head_state[0:2] = [self.head.x / b.width, self.head.y / b.height]
            head_state[2] = np.clip(dxc / net.hpwl, -2, 2) / 2.0
            head_state[3] = np.clip(dyc / net.hpwl, -2, 2) / 2.0
            head_state[4] = min(d / net.hpwl, 2.0) / 2.0
            head_state[5 + self.head.layer] = 1.0
            head_state[17] = len(self.completed) / max(len(b.nets), 1)
            head_state[18] = self.budget / self.cfg.max_steps_per_net
            
            canonical_prev_heading = self.head.prev_heading_angle - th
            head_state[19] = np.cos(canonical_prev_heading)
            head_state[20] = np.sin(canonical_prev_heading)
            
            head_state[21] = 1.0 if (net.signal_type != 1
                                    and b.layer_roles[self.head.layer] == LAYER_ROLE_POWER) else 0.0

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
