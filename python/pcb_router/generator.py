"""Programmatic curriculum board generator.

Boards are assembled from a small footprint library (footprints.py) --
passives, headers, and ICs -- placed like real components with realistic
pin pitch and courtyard obstacles, not floating random points. Stage 0 is
trivially routable (few components, 2 layers, no stack-up); later stages add
components, layers, a stack-up with dedicated power/ground planes, and
differential pairs. Every net is still exactly 2 pads (components change
*where* pads sit and add extra obstacle discs; they don't add multi-pin
nets), so N_MAX_PINS stays the only cap to respect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .board import Board
from .config import DesignRules, LAYER_ROLE_POWER, LAYER_ROLE_SIGNAL, N_MAX_PINS
from .footprints import FOOTPRINTS, place_footprint


@dataclass
class StageSpec:
    layers: int
    size: float                      # square board edge (mm)
    components: Dict[str, int]       # footprint name -> instance count
    n_diff_pairs: int = 0
    extra_keepouts: int = 0          # a few random obstacle discs, for variety
    bottom_mount_prob: float = 0.0   # chance a component mounts on the far
                                     # outer layer instead of the top -- the
                                     # only thing that forces via usage now
                                     # that pads aren't scattered at random
    # COMMIT legal within this of the target (mm). Wider on early stages so an
    # unskilled policy stumbles into completions often enough for the sparse
    # C reward to be learnable; anneals to the DesignRules default (1.0).
    commit_snap: float = 1.0


STAGES = [
    StageSpec(layers=2, size=20.0,
              components={"PASSIVE_SMALL": 3},
              commit_snap=2.5),
    StageSpec(layers=2, size=25.0,
              components={"PASSIVE_SMALL": 4, "HEADER_4": 1},
              extra_keepouts=1, commit_snap=2.0),
    StageSpec(layers=2, size=28.0,
              components={"PASSIVE_SMALL": 5, "PASSIVE_LARGE": 1, "HEADER_6": 1},
              extra_keepouts=2, bottom_mount_prob=0.3, commit_snap=1.5),
    StageSpec(layers=4, size=32.0,
              components={"PASSIVE_SMALL": 4, "PASSIVE_LARGE": 2, "HEADER_6": 1,
                          "IC_SOIC8": 1},
              n_diff_pairs=1, extra_keepouts=2, bottom_mount_prob=0.3),
    StageSpec(layers=6, size=38.0,
              components={"PASSIVE_SMALL": 5, "PASSIVE_LARGE": 2, "HEADER_8": 1,
                          "IC_SOIC8": 2},
              n_diff_pairs=2, extra_keepouts=3, bottom_mount_prob=0.4),
    StageSpec(layers=6, size=45.0,
              components={"PASSIVE_SMALL": 4, "PASSIVE_LARGE": 1, "HEADER_6": 1,
                          "IC_SOIC8": 1, "IC_QFN16": 1},
              n_diff_pairs=2, extra_keepouts=4, bottom_mount_prob=0.5),
]


def stackup_roles(num_layers: int) -> List[int]:
    """Assign each layer a role. Mirrors real stack-ups: 2-layer boards have
    no dedicated planes; 4/6-layer boards sandwich signal layers around
    dedicated power/ground planes."""
    if num_layers <= 2:
        return [LAYER_ROLE_SIGNAL] * num_layers
    if num_layers == 4:
        return [LAYER_ROLE_SIGNAL, LAYER_ROLE_POWER, LAYER_ROLE_POWER, LAYER_ROLE_SIGNAL]
    if num_layers == 6:
        return [LAYER_ROLE_SIGNAL, LAYER_ROLE_POWER, LAYER_ROLE_SIGNAL,
                LAYER_ROLE_SIGNAL, LAYER_ROLE_POWER, LAYER_ROLE_SIGNAL]
    roles = [LAYER_ROLE_POWER] * num_layers
    roles[0] = roles[-1] = LAYER_ROLE_SIGNAL
    return roles


_HS_TRACE_WIDTH = 0.20     # matched width for differential pairs
_POWER_TRACE_WIDTH = 0.35  # wider than signal, for current-carrying nets
_DIFF_PITCH = 0.30         # perpendicular P/N spacing (mm)
_HS_PAD_R = 0.3            # bare diff-pair pad radius (not tied to a footprint)


def generate_board(stage: int, rng: np.random.Generator,
                   rules: DesignRules | None = None) -> Board:
    spec = STAGES[min(stage, len(STAGES) - 1)]
    rules = rules or DesignRules(commit_snap=spec.commit_snap)
    board = Board(width=spec.size, height=spec.size, num_layers=spec.layers,
                  rules=rules, layer_roles=stackup_roles(spec.layers))

    margin = 1.5 + rules.board_clearance
    gap = 0.5  # minimum clearance between any two bounding boxes (mm)
    # (center, half_w, half_h) for every placed obstacle -- components,
    # decoupling caps, diff-pair pads, extra keep-outs all go through this
    # one spacing check. Rectangles, not bounding circles: headers in
    # particular are long, thin strips, and a circular bound would demand a
    # clearance radius equal to half the strip's *length* in every
    # direction, making them nearly unplaceable next to anything else.
    # Circular items (bare diff-pair pads, extra keep-outs) just pass
    # hw == hh == radius.
    occupied: List[Tuple[np.ndarray, float, float]] = []

    def fits(center: np.ndarray, hw: float, hh: float) -> bool:
        if not (margin + hw <= center[0] <= spec.size - margin - hw
                and margin + hh <= center[1] <= spec.size - margin - hh):
            return False
        return all(abs(center[0] - c[0]) >= hw + chw + gap
                  or abs(center[1] - c[1]) >= hh + chh + gap
                  for c, chw, chh in occupied)

    def place_center(hw: float, hh: float) -> np.ndarray:
        for _ in range(500):
            p = np.array([rng.uniform(margin + hw, spec.size - margin - hw),
                         rng.uniform(margin + hh, spec.size - margin - hh)])
            if fits(p, hw, hh):
                return p
        raise RuntimeError("board generator: could not place component (density too high)")

    # ---- place components ---------------------------------------------------
    # candidate pin slot: world xy, layer, pad radius, whether it's a
    # designated power pin. Deferred -- these only become real Pad objects
    # once we know which ones actually get netted (decoupling / diff-pair /
    # general pairing), so no pad is ever left without a net.
    slots: List[dict] = []
    for fp_name, count in spec.components.items():
        fp = FOOTPRINTS[fp_name]
        for _ in range(count):
            quadrant = int(rng.integers(0, 4))
            hw, hh = ((fp.courtyard_hh, fp.courtyard_hw) if quadrant % 2
                     else (fp.courtyard_hw, fp.courtyard_hh))
            center = place_center(hw, hh)
            occupied.append((center, hw, hh))
            mount_layer = spec.layers - 1 if rng.random() < spec.bottom_mount_prob else 0
            if fp.body_radius > 0:
                board.add_keepout(center[0], center[1], fp.body_radius)
            pads, _ = place_footprint(fp, tuple(center), quadrant)
            for i, (px, py) in enumerate(pads):
                slots.append({"xy": np.array([px, py]), "layer": mount_layer,
                              "r": fp.pad_radius, "power": i in fp.power_pins})

    # ---- decoupling caps: one PASSIVE_SMALL per power pin --------------------
    cap_fp = FOOTPRINTS["PASSIVE_SMALL"]
    power_slots = [s for s in slots if s["power"]]
    # net spec: pads + signal_type + optional pair_id + trace width
    net_specs: List[dict] = []

    for pwr in power_slots:
        quadrant = int(rng.integers(0, 4))
        hw, hh = ((cap_fp.courtyard_hh, cap_fp.courtyard_hw) if quadrant % 2
                 else (cap_fp.courtyard_hw, cap_fp.courtyard_hh))
        center = None
        for _ in range(50):
            offset = rng.uniform(-1.0, 1.0, size=2)
            offset *= (2.0 + rng.uniform(0, 2.0)) / (np.linalg.norm(offset) + 1e-6)
            cand = pwr["xy"] + offset
            if fits(cand, hw, hh):
                center = cand
                break
        if center is None:
            continue  # no room for this decoupling cap -- skip the pin entirely
        occupied.append((center, hw, hh))
        pads, _ = place_footprint(cap_fp, tuple(center), quadrant)
        # Mount on the SAME layer as the power pin -- a decoupling cap sits
        # right next to the pin it bypasses, same side of the board.
        cap_a = {"xy": np.array(pads[0]), "layer": pwr["layer"], "r": cap_fp.pad_radius}
        cap_b = {"xy": np.array(pads[1]), "layer": pwr["layer"], "r": cap_fp.pad_radius}
        net_specs.append({"a": pwr, "b": cap_a, "signal_type": 1, "pair_id": None,
                          "width": _POWER_TRACE_WIDTH})
        slots.append({**cap_b, "power": False})  # other leg joins the general pool

    slots = [s for s in slots if not s["power"]]  # power pins consumed above

    # ---- differential pairs: bare pad pairs, not tied to a footprint --------
    signal_layers = [i for i, r in enumerate(board.layer_roles) if r == LAYER_ROLE_SIGNAL]
    for _ in range(spec.n_diff_pairs):
        a = b = None
        for _ in range(200):
            cand = rng.uniform(margin, spec.size - margin, size=2)
            if fits(cand, _HS_PAD_R, _HS_PAD_R):
                a = cand
                break
        if a is None:
            continue
        for _ in range(200):
            cand = rng.uniform(margin, spec.size - margin, size=2)
            d = np.linalg.norm(cand - a)
            if 3.0 <= d <= 0.6 * spec.size and fits(cand, _HS_PAD_R, _HS_PAD_R):
                b = cand
                break
        if b is None:
            continue
        direction = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        perp = np.array([-direction[1], direction[0]]) * (_DIFF_PITCH / 2.0)
        layer = int(rng.choice(signal_layers))
        a_p, a_n, b_p, b_n = a + perp, a - perp, b + perp, b - perp
        for c in (a_p, a_n, b_p, b_n):
            occupied.append((c, _HS_PAD_R, _HS_PAD_R))
        pair_id = len(net_specs)  # unique tag; net_specs only grows from here
        p_a = {"xy": a_p, "layer": layer, "r": _HS_PAD_R}
        p_b = {"xy": b_p, "layer": layer, "r": _HS_PAD_R}
        n_a = {"xy": a_n, "layer": layer, "r": _HS_PAD_R}
        n_b = {"xy": b_n, "layer": layer, "r": _HS_PAD_R}
        net_specs.append({"a": p_a, "b": p_b, "signal_type": 2, "pair_id": pair_id,
                          "width": _HS_TRACE_WIDTH})
        net_specs.append({"a": n_a, "b": n_b, "signal_type": 2, "pair_id": pair_id,
                          "width": _HS_TRACE_WIDTH})

    # ---- general pool: pair remaining pads, preferring a sane distance -----
    # window (3mm .. 60% of the board) so straight-line routing is routable
    # but non-trivial -- same rule the old floating-pad generator used. Pure
    # random pairing (no window) can degenerate into near-zero-length nets
    # (two pads of the same tiny component) or paths that happen to graze a
    # foreign obstacle, either of which the naive canonical-frame oracle
    # (tests/test_env.py) chokes on.
    rng.shuffle(slots)
    if len(slots) % 2 == 1:
        slots.pop()
    min_d, max_d = 3.0, 0.6 * spec.size
    used = [False] * len(slots)
    for i in range(len(slots)):
        if used[i]:
            continue
        partner = None
        for j in range(i + 1, len(slots)):
            if not used[j] and min_d <= np.linalg.norm(slots[i]["xy"] - slots[j]["xy"]) <= max_d:
                partner = j
                break
        if partner is None:  # window unsatisfiable -- take any remaining pad
            partner = next(j for j in range(i + 1, len(slots)) if not used[j])
        used[i] = used[partner] = True
        net_specs.append({"a": slots[i], "b": slots[partner], "signal_type": 0,
                          "pair_id": None, "width": rules.trace_width})

    # ---- a few random extra keep-outs for congestion variety -----------------
    for _ in range(spec.extra_keepouts):
        for _ in range(200):
            r = float(rng.uniform(1.0, 2.0))
            p = rng.uniform(margin + r, spec.size - margin - r, size=2)
            if fits(p, r, r):
                board.add_keepout(p[0], p[1], r)
                occupied.append((p, r, r))
                break

    # ---- realize pads + nets on the board -------------------------------------
    assert 2 * len(net_specs) <= N_MAX_PINS, \
        "raise N_MAX_PINS or shrink this stage's component mix"
    for ns in net_specs:
        a, b = ns["a"], ns["b"]
        ia = board.add_pad(a["xy"][0], a["xy"][1], a["r"], a["layer"], a["layer"],
                           net_id=len(board.nets))
        ib = board.add_pad(b["xy"][0], b["xy"][1], b["r"], b["layer"], b["layer"],
                           net_id=len(board.nets))
        board.add_net(ia, ib, signal_type=ns["signal_type"], pair_id=ns["pair_id"],
                      trace_width=ns["width"])

    return board
