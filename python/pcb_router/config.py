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
N_ANGLE_BINS = 128                   # 2.8125 degrees per bin
N_DIST_BINS = 5                      # 5 discrete distance steps
DIST_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0] # fractions of max distance
MAX_LAYERS = 12

# ---- Fixed observation dims (padded for trivial batching) -------------------
N_MAX_PINS = 64                      # max pads per board (netlist graph nodes)
P_MAX = 256                          # max points in the egocentric cloud
NODE_FEAT_DIM = 10                   # was 8: + per-pad trace_width, + per-pad layer_role
POINT_FEAT_DIM = 10
HEAD_FEAT_DIM = 22                   # position(2) + target_offset(2) + distance(1) + layer_onehot(12)
                                     # + completion(1) + budget(1) + prev_heading_cos_sin(2)
                                     # + stackup_mismatch(1) = 22

# ---- Stack-up layer roles ----------------------------------------------------
# A layer's "role" governs the stack-up penalty in env.step: routing a
# non-power net's copper on a POWER-role layer is legal (never masked) but
# costs RewardWeights.lam_stackup, same normalized-length shape as lam1.
# SIGNAL-role layers are general-purpose -- nobody is penalized there.
LAYER_ROLE_SIGNAL = 0
LAYER_ROLE_POWER = 1


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
    lam2: float = 2.0                # via usage (expensive: prefer routing around over abusing vias)
    lam3: float = 5.0                # impedance mismatch   (physics hook)
    lam4: float = 2.0                # differential skew    (physics hook)
    lam5: float = 1.0                # crosstalk            (physics hook)
    lam_turn: float = 0.3            # turn-angle penalty (radians normalized)
    lam_straight: float = 0.05       # bonus for continuing in the same direction
    lam_efficiency: float = 2.0      # terminal bonus: reward proportional to hpwl/actual_length (1.0 = perfect straight route)
    lam_stackup: float = 0.5         # non-power net dwelling on a POWER-role layer
    beta: float = 1.5                # potential-based shaping weight; MUST stay
                                     # > lam1 or moving toward the target earns
                                     # net-zero and "don't move" becomes a local
                                     # optimum (see docs/reward-function.md)
    # NOTE: shaping is deliberately undiscounted -- beta*(phi' - phi), no gamma
    # inside (see the annuity note in docs/reward-function.md and env.step).
    # The RL discount lives in PPOConfig.gamma only.


@dataclass
class EnvConfig:
    lookahead: float = 6.0           # max EXTEND reach per step (mm)
    obs_window: float = 8.0          # egocentric point-cloud radius (mm)
    max_steps_per_net: int = 96
    cap_sample_spacing: float = 1.0  # point-cloud samples along traces (mm)
    rules: DesignRules = field(default_factory=DesignRules)
    reward: RewardWeights = field(default_factory=RewardWeights)
