"""Shared constants and configuration.

This module is the single source of truth for the action-space contract
(previously mirrored in cpp/include/pcb/action_masker.hpp — the C++ port
must match these numbers exactly).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---- Action space -----------------------------------------------------------
N_ACTION_TYPES = 3
A_EXTEND, A_VIA, A_COMMIT = 0, 1, 2
N_ANGLE_BINS = 64                    # 5.625 degrees per bin
MAX_LAYERS = 12

# ---- Fixed observation dims (padded for trivial batching) -------------------
N_MAX_PINS = 64                      # max pads per board (netlist graph nodes)
P_MAX = 256                          # max points in the egocentric cloud
NODE_FEAT_DIM = 8
POINT_FEAT_DIM = 10
HEAD_FEAT_DIM = 19


@dataclass
class DesignRules:
    """All lengths in millimetres."""
    trace_width: float = 0.15
    trace_clearance: float = 0.15    # copper-to-copper, different nets
    via_drill: float = 0.30
    via_annular: float = 0.15        # via pad radius = drill/2 + annular
    via_clearance: float = 0.20
    min_segment_length: float = 0.05
    board_clearance: float = 0.25    # copper to board edge
    commit_snap: float = 1.0         # COMMIT legal within this of target pad

    @property
    def via_pad_radius(self) -> float:
        return self.via_drill / 2.0 + self.via_annular


@dataclass
class RewardWeights:
    """Constants from docs/reward-function.md."""
    C: float = 10.0                  # per-net completion
    B: float = 50.0                  # terminal completion-ratio bonus
    F: float = 20.0                  # terminal failure penalty (any net unrouted)
    D: float = 50.0                  # DRC safety net (should never fire)
    lam1: float = 1.0                # normalized length (detour factor)
    lam2: float = 0.5                # via usage
    lam3: float = 5.0                # impedance mismatch   (physics hook)
    lam4: float = 2.0                # differential skew    (physics hook)
    lam5: float = 1.0                # crosstalk            (physics hook)
    beta: float = 3.0                # potential-based shaping weight; MUST stay
                                     # > lam1 or moving toward the target earns
                                     # net-zero and "don't move" becomes a local
                                     # optimum (see docs/reward-function.md)
    gamma: float = 0.995             # discount (also used by PPO)


@dataclass
class EnvConfig:
    lookahead: float = 6.0           # max EXTEND reach per step (mm)
    obs_window: float = 8.0          # egocentric point-cloud radius (mm)
    max_steps_per_net: int = 96
    cap_sample_spacing: float = 1.0  # point-cloud samples along traces (mm)
    rules: DesignRules = field(default_factory=DesignRules)
    reward: RewardWeights = field(default_factory=RewardWeights)
