"""Board state: pads, nets, traces, vias, keep-outs in continuous space.

Geometry is kept as Python lists during an episode and compiled to flat
NumPy arrays on demand (cached, invalidated on mutation) for the vectorized
kernels in geometry.py. Kind codes: 0=trace, 1=pad, 2=via, 3=keepout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np

from .config import DesignRules, LAYER_ROLE_SIGNAL

KIND_TRACE, KIND_PAD, KIND_VIA, KIND_KEEPOUT = 0, 1, 2, 3


@dataclass
class Pad:
    x: float
    y: float
    r: float
    layer_lo: int
    layer_hi: int
    net_id: int


@dataclass
class Net:
    pins: Tuple[int, int]        # pad indices (2-pin nets in v1)
    hpwl: float                  # half-perimeter wirelength lower bound
    # physics stubs, consumed by the PhysicsEvaluator hook:
    z_required: float = 0.0
    signal_type: int = 0         # 0=signal, 1=power, 2=high-speed
    trace_width: float = 0.15    # per-net copper width (mm); set by the generator
    pair_id: Optional[int] = None  # shared id for a differential pair's P/N nets


@dataclass
class Board:
    width: float
    height: float
    num_layers: int
    rules: DesignRules = field(default_factory=DesignRules)
    # One role per layer (LAYER_ROLE_SIGNAL / LAYER_ROLE_POWER); defaults to
    # all-signal (no dedicated planes) until the generator assigns a real
    # stack-up. Consumed by env.step's stack-up penalty.
    layer_roles: List[int] = field(default_factory=list)

    pads: List[Pad] = field(default_factory=list)
    nets: List[Net] = field(default_factory=list)
    # traces: (ax, ay, bx, by, half_width, layer, net_id)
    traces: List[Tuple[float, float, float, float, float, int, int]] = field(default_factory=list)
    # vias: (x, y, pad_r, layer_lo, layer_hi, net_id)
    vias: List[Tuple[float, float, float, int, int, int]] = field(default_factory=list)
    # keepouts: (x, y, r) -- all layers, net_id -1
    keepouts: List[Tuple[float, float, float]] = field(default_factory=list)

    _version: int = 0
    _cache_version: int = -1
    _cache: SimpleNamespace = None

    def __post_init__(self):
        if not self.layer_roles:
            self.layer_roles = [LAYER_ROLE_SIGNAL] * self.num_layers

    # ------------------------------------------------------------- mutation
    def add_pad(self, x, y, r, layer_lo, layer_hi, net_id) -> int:
        self.pads.append(Pad(x, y, r, layer_lo, layer_hi, net_id))
        self._version += 1
        return len(self.pads) - 1

    def add_net(self, pin_a: int, pin_b: int, **kw) -> int:
        pa, pb = self.pads[pin_a], self.pads[pin_b]
        hpwl = abs(pa.x - pb.x) + abs(pa.y - pb.y)
        self.nets.append(Net(pins=(pin_a, pin_b), hpwl=max(hpwl, 1.0), **kw))
        return len(self.nets) - 1

    def add_trace(self, ax, ay, bx, by, half_width, layer, net_id):
        self.traces.append((ax, ay, bx, by, half_width, layer, net_id))
        self._version += 1

    def add_via(self, x, y, layer_lo, layer_hi, net_id):
        self.vias.append((x, y, self.rules.via_pad_radius, layer_lo, layer_hi, net_id))
        self._version += 1

    def add_keepout(self, x, y, r):
        self.keepouts.append((x, y, r))
        self._version += 1

    # -------------------------------------------------------------- queries
    def arrays(self) -> SimpleNamespace:
        """Flat obstacle arrays (cached). Discs = pads + vias + keepouts;
        capsules = traces."""
        if self._cache is not None and self._cache_version == self._version:
            return self._cache

        disc_rows = (
            [(p.x, p.y, p.r, p.layer_lo, p.layer_hi, p.net_id, KIND_PAD) for p in self.pads]
            + [(x, y, r, lo, hi, net, KIND_VIA) for (x, y, r, lo, hi, net) in self.vias]
            + [(x, y, r, 0, self.num_layers - 1, -1, KIND_KEEPOUT) for (x, y, r) in self.keepouts]
        )
        if disc_rows:
            d = np.array(disc_rows, dtype=np.float64)
            disc_c, disc_r = d[:, 0:2], d[:, 2]
            disc_llo, disc_lhi = d[:, 3].astype(int), d[:, 4].astype(int)
            disc_net, disc_kind = d[:, 5].astype(int), d[:, 6].astype(int)
        else:
            disc_c = np.zeros((0, 2)); disc_r = np.zeros(0)
            disc_llo = disc_lhi = disc_net = disc_kind = np.zeros(0, dtype=int)

        if self.traces:
            t = np.array(self.traces, dtype=np.float64)
            cap_a, cap_b, cap_r = t[:, 0:2], t[:, 2:4], t[:, 4]
            cap_layer, cap_net = t[:, 5].astype(int), t[:, 6].astype(int)
        else:
            cap_a = cap_b = np.zeros((0, 2)); cap_r = np.zeros(0)
            cap_layer = cap_net = np.zeros(0, dtype=int)

        self._cache = SimpleNamespace(
            disc_c=disc_c, disc_r=disc_r, disc_llo=disc_llo, disc_lhi=disc_lhi,
            disc_net=disc_net, disc_kind=disc_kind,
            cap_a=cap_a, cap_b=cap_b, cap_r=cap_r,
            cap_layer=cap_layer, cap_net=cap_net,
        )
        self._cache_version = self._version
        return self._cache

    def outline_inset(self, inset: float):
        lo = np.array([inset, inset])
        hi = np.array([self.width - inset, self.height - inset])
        return lo, hi
