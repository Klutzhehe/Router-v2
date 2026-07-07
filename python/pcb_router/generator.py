"""Programmatic curriculum board generator.

Stage 0 is trivially routable (few nets, one layer pair, no keep-outs);
later stages add nets, layers, keep-outs, and cross-layer pad pairs that
force via usage. Caps respect N_MAX_PINS so observations stay fixed-size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .board import Board
from .config import DesignRules, N_MAX_PINS


@dataclass
class StageSpec:
    layers: int
    size: float             # square board edge (mm)
    n_nets: int
    n_keepouts: int
    cross_layer_prob: float  # chance a net's second pad sits on the bottom layer
    pad_r: float = 0.5


STAGES = [
    StageSpec(layers=2,  size=20.0, n_nets=3,  n_keepouts=0, cross_layer_prob=0.0),
    StageSpec(layers=2,  size=25.0, n_nets=6,  n_keepouts=2, cross_layer_prob=0.0),
    StageSpec(layers=2,  size=25.0, n_nets=8,  n_keepouts=2, cross_layer_prob=0.3),
    StageSpec(layers=4,  size=30.0, n_nets=12, n_keepouts=4, cross_layer_prob=0.3),
    StageSpec(layers=6,  size=40.0, n_nets=20, n_keepouts=8, cross_layer_prob=0.4),
    StageSpec(layers=12, size=50.0, n_nets=30, n_keepouts=12, cross_layer_prob=0.5),
]


def generate_board(stage: int, rng: np.random.Generator,
                   rules: DesignRules | None = None) -> Board:
    spec = STAGES[min(stage, len(STAGES) - 1)]
    rules = rules or DesignRules()
    assert 2 * spec.n_nets <= N_MAX_PINS, "raise N_MAX_PINS for this stage"

    board = Board(width=spec.size, height=spec.size,
                  num_layers=spec.layers, rules=rules)

    margin = spec.pad_r + rules.board_clearance + 1.0
    # Pads must be far enough apart that a trace + clearance fits between any two.
    min_sep = 2 * spec.pad_r + 2 * rules.trace_clearance + 2 * rules.trace_width + 0.5

    placed: list[np.ndarray] = []

    def place_point(min_dist_from=None, max_dist_from=None, anchor=None):
        for _ in range(500):
            p = rng.uniform(margin, spec.size - margin, size=2)
            if placed and np.min(np.linalg.norm(np.array(placed) - p, axis=1)) < min_sep:
                continue
            if anchor is not None:
                d = np.linalg.norm(p - anchor)
                if not (min_dist_from <= d <= max_dist_from):
                    continue
            placed.append(p)
            return p
        raise RuntimeError("board generator: could not place pad (density too high)")

    for _ in range(spec.n_nets):
        a = place_point()
        # Pair pad 3..60% of the board away: routable but non-trivial.
        b = place_point(min_dist_from=3.0, max_dist_from=0.6 * spec.size, anchor=a)
        layer_b = spec.layers - 1 if rng.random() < spec.cross_layer_prob else 0
        ia = board.add_pad(a[0], a[1], spec.pad_r, 0, 0, net_id=len(board.nets))
        ib = board.add_pad(b[0], b[1], spec.pad_r, layer_b, layer_b, net_id=len(board.nets))
        board.add_net(ia, ib, signal_type=int(rng.integers(0, 3)))

    for _ in range(spec.n_keepouts):
        for _ in range(200):
            r = float(rng.uniform(1.0, 2.5))
            p = rng.uniform(margin + r, spec.size - margin - r, size=2)
            # Keep-outs must not swallow pads or sit within a pad's escape zone.
            if placed and np.min(np.linalg.norm(np.array(placed) - p, axis=1)) < r + spec.pad_r + 1.5:
                continue
            board.add_keepout(p[0], p[1], r)
            break

    return board
