"""Dynamic action masking: the Python port of cpp/include/pcb/action_masker.hpp.

For the routing head the masker computes, per step:
  type_mask (3,)       -- EXTEND / PLACE_VIA / COMMIT_NET legality
  angle_mask (64,)     -- directions with >= min_segment_length of legal travel
  max_distance (64,)   -- farthest legal extension per direction (mm)
  layer_mask (12,)     -- legal via target layers

Every EXTEND the agent samples is legal by construction: the policy emits a
*fraction* that the environment scales into [min_segment_length,
max_distance[bin]]. DRC violations at runtime therefore indicate a geometry
bug, not agent misbehaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from .board import Board
from .config import MAX_LAYERS, N_ACTION_TYPES, N_ANGLE_BINS, A_EXTEND, A_VIA, A_COMMIT

_DIRS = geo.unit_dirs(N_ANGLE_BINS)
_SAFETY = 1e-4   # mm shaved off every max distance for float robustness


@dataclass
class RoutingHead:
    x: float
    y: float
    layer: int
    net_id: int
    half_width: float
    target_x: float
    target_y: float
    target_pad: int
    just_placed_via: bool = False
    prev_heading_angle: float = 0.0  # radians; used to compute turn penalty


@dataclass
class ActionMask:
    type_mask: np.ndarray      # (3,)  uint8
    angle_mask: np.ndarray     # (64,) uint8 -- canonical frame (bin 0 = at target)
    max_distance: np.ndarray   # (64,) float -- canonical frame
    layer_mask: np.ndarray     # (12,) uint8
    frame_offset: int          # world bin closest to the target direction;
                               # world_bin = (canonical_bin + frame_offset) % N_ANGLE_BINS


class ActionMasker:
    def __init__(self, board: Board):
        self.board = board
        self.rules = board.rules

    # ------------------------------------------------------------- helpers
    def _foreign_on_layer(self, arr, layer: int, net_id: int):
        """Selection masks for obstacles that can collide with copper of
        `net_id` on `layer` (same-net copper never collides)."""
        d_sel = (arr.disc_net != net_id) & (arr.disc_llo <= layer) & (arr.disc_lhi >= layer)
        c_sel = (arr.cap_net != net_id) & (arr.cap_layer == layer)
        return d_sel, c_sel

    def max_legal_distances(self, origin: np.ndarray, dirs: np.ndarray,
                            layer: int, net_id: int, half_width: float,
                            lookahead: float) -> np.ndarray:
        """Swept-disc cast of the head along `dirs`; (K,) legal distances."""
        arr = self.board.arrays()
        inflate = half_width + self.rules.trace_clearance
        d_sel, c_sel = self._foreign_on_layer(arr, layer, net_id)

        # Broad phase: AABB window around the origin (the "one windowed query"
        # from the C++ design -- here a vectorized filter).
        reach = lookahead + inflate
        if d_sel.any():
            close = np.abs(arr.disc_c - origin).max(axis=1) <= reach + arr.disc_r
            d_sel = d_sel & close
        if c_sel.any():
            lo = np.minimum(arr.cap_a, arr.cap_b) - (arr.cap_r + reach)[:, None]
            hi = np.maximum(arr.cap_a, arr.cap_b) + (arr.cap_r + reach)[:, None]
            close = np.all((origin >= lo) & (origin <= hi), axis=1)
            c_sel = c_sel & close

        t = np.full(dirs.shape[0], lookahead)
        if d_sel.any():
            td = geo.ray_disc_first_hit(origin, dirs, arr.disc_c[d_sel],
                                        arr.disc_r[d_sel] + inflate)
            t = np.minimum(t, td.min(axis=1))
        if c_sel.any():
            tc = geo.ray_capsule_first_hit(origin, dirs, arr.cap_a[c_sel],
                                           arr.cap_b[c_sel],
                                           arr.cap_r[c_sel] + inflate)
            t = np.minimum(t, tc.min(axis=1))

        lo, hi = self.board.outline_inset(half_width + self.rules.board_clearance)
        t = np.minimum(t, geo.rect_inset_exit(origin, dirs, lo, hi))
        return np.maximum(t - _SAFETY, 0.0)

    def segment_legal(self, origin: np.ndarray, dest: np.ndarray,
                      layer: int, net_id: int, half_width: float) -> bool:
        delta = dest - origin
        dist = float(np.linalg.norm(delta))
        if dist < 1e-9:
            return True
        d = (delta / dist)[None, :]
        t = self.max_legal_distances(origin, d, layer, net_id, half_width,
                                     lookahead=dist + 1.0)
        return bool(t[0] >= dist - 1e-6)

    def via_fits(self, pos: np.ndarray, layer_lo: int, layer_hi: int,
                 net_id: int) -> bool:
        """Via barrel+pad must clear foreign copper on EVERY layer it spans."""
        arr = self.board.arrays()
        r_via = self.rules.via_pad_radius
        clr = self.rules.via_clearance

        d_sel = (arr.disc_net != net_id) & ~((arr.disc_lhi < layer_lo) | (arr.disc_llo > layer_hi))
        if d_sel.any():
            dist = np.linalg.norm(arr.disc_c[d_sel] - pos, axis=1)
            if np.any(dist <= arr.disc_r[d_sel] + r_via + clr):
                return False

        c_sel = (arr.cap_net != net_id) & (arr.cap_layer >= layer_lo) & (arr.cap_layer <= layer_hi)
        if c_sel.any():
            dist = geo.point_seg_dist(pos[None, :], arr.cap_a[c_sel], arr.cap_b[c_sel])
            if np.any(dist <= arr.cap_r[c_sel] + r_via + clr):
                return False

        lo, hi = self.board.outline_inset(r_via + self.rules.board_clearance)
        return bool(np.all(pos >= lo) and np.all(pos <= hi))

    # ---------------------------------------------------------------- mask
    def compute_mask(self, head: RoutingHead, lookahead: float) -> ActionMask:
        origin = np.array([head.x, head.y])

        # Target-aligned canonical frame: find the world bin closest to the
        # target direction, then roll the per-direction arrays so canonical
        # bin 0 is that direction. world_bin = (canonical_bin + frame_offset)
        # % N_ANGLE_BINS (see config.py / env.py -- this must be the ONLY
        # place frame_offset is derived from world geometry).
        dx, dy = head.target_x - head.x, head.target_y - head.y
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            frame_offset = 0
        else:
            theta = np.arctan2(dy, dx) % (2.0 * np.pi)
            frame_offset = int(round(theta / (2.0 * np.pi / N_ANGLE_BINS))) % N_ANGLE_BINS

        # EXTEND: per-direction legal travel (world frame, then rolled).
        max_dist_world = self.max_legal_distances(origin, _DIRS, head.layer,
                                                   head.net_id, head.half_width,
                                                   lookahead)
        angle_mask_world = (max_dist_world >= self.rules.min_segment_length).astype(np.uint8)
        max_dist = np.roll(max_dist_world, -frame_offset)
        angle_mask = np.roll(angle_mask_world, -frame_offset)

        # PLACE_VIA: which target layers can this position reach?
        layer_mask = np.zeros(MAX_LAYERS, dtype=np.uint8)
        for tgt in range(self.board.num_layers):
            if tgt == head.layer:
                continue
            lo, hi = min(head.layer, tgt), max(head.layer, tgt)
            if self.via_fits(origin, lo, hi, head.net_id):
                layer_mask[tgt] = 1

        # COMMIT_NET: close enough, right layer, and the closing segment fits.
        target = np.array([head.target_x, head.target_y])
        tp = self.board.pads[head.target_pad]
        d_target = float(np.linalg.norm(target - origin))
        commit_ok = (
            tp.layer_lo <= head.layer <= tp.layer_hi
            and d_target <= self.rules.commit_snap
            and self.segment_legal(origin, target, head.layer, head.net_id,
                                   head.half_width)
        )

        type_mask = np.zeros(N_ACTION_TYPES, dtype=np.uint8)
        type_mask[A_EXTEND] = 1 if angle_mask.any() else 0
        type_mask[A_VIA] = 1 if (layer_mask.any() and not head.just_placed_via) else 0
        type_mask[A_COMMIT] = 1 if commit_ok else 0

        return ActionMask(type_mask=type_mask, angle_mask=angle_mask,
                          max_distance=max_dist, layer_mask=layer_mask,
                          frame_offset=frame_offset)
