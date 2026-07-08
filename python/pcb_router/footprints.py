"""Small component-footprint library for the board generator.

Each Footprint is pad offsets (relative to a component origin) + a
rectangular courtyard extent, both defined before rotation. Placement
rotates by a random multiple of 90 degrees, which keeps everything
axis-aligned -- no new geometry primitives, just coordinate swaps/negation.

Courtyards are NOT drawn as a single rectangle obstacle (the engine only
knows discs and capsules); generator.py approximates each courtyard as a
short chain of keep-out discs along its long axis, reusing the existing
disc-keepout code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Footprint:
    name: str
    pad_offsets: List[Tuple[float, float]]  # (dx, dy) mm, pre-rotation
    pad_radius: float
    courtyard_hw: float   # half-width (x extent), pre-rotation -- placement spacing only
    courtyard_hh: float   # half-height (y extent), pre-rotation -- placement spacing only
    power_pins: List[int] = field(default_factory=list)  # indices into pad_offsets
    # A single keep-out disc at the component center, radius 0 = none. Only
    # packages with real empty board area under the body (ICs) get one --
    # passives/headers rely on their own pad discs + placement spacing, since
    # their "body" sits too close to their own pads to safely keep out
    # without risking swallowing a pad (rotation-invariant since it's centered).
    body_radius: float = 0.0


def _header(n: int, pitch: float = 2.54, pad_radius: float = 0.4) -> Footprint:
    span = (n - 1) * pitch
    xs = [i * pitch - span / 2.0 for i in range(n)]
    return Footprint(
        name=f"HEADER_{n}",
        pad_offsets=[(x, 0.0) for x in xs],
        pad_radius=pad_radius,
        courtyard_hw=span / 2.0 + 0.5,
        courtyard_hh=1.0,
    )


def _qfn16(pitch: float = 0.5, pad_radius: float = 0.15) -> Footprint:
    """16 pins around a 4-sided package, 4 pins per side."""
    side_off = [(-1.5 + i) * pitch for i in range(4)]  # 4 positions per side
    body = 2.0  # half-extent of the package body each side's pins sit against
    offsets: List[Tuple[float, float]] = []
    offsets += [(body, y) for y in side_off]            # right side
    offsets += [(x, body) for x in side_off]            # top side
    offsets += [(-body, y) for y in reversed(side_off)]  # left side
    offsets += [(x, -body) for x in reversed(side_off)]  # bottom side
    return Footprint(
        name="IC_QFN16",
        pad_offsets=offsets,
        pad_radius=pad_radius,
        courtyard_hw=2.5,
        courtyard_hh=2.5,
        power_pins=[0, 8],   # one pin on the right side, one on the left
        body_radius=1.3,     # comfortably inside the pad ring on every side
    )


def _rotate(dx: float, dy: float, quadrant: int) -> Tuple[float, float]:
    """Rotate an offset by quadrant*90 degrees (0..3)."""
    q = quadrant % 4
    if q == 0:
        return dx, dy
    if q == 1:
        return -dy, dx
    if q == 2:
        return -dx, -dy
    return dy, -dx


def place_footprint(fp: Footprint, center: Tuple[float, float], quadrant: int):
    """World-space pad positions and rotated courtyard half-extents."""
    cx, cy = center
    pads = [(cx + rx, cy + ry)
            for (rx, ry) in (_rotate(dx, dy, quadrant) for dx, dy in fp.pad_offsets)]
    hw, hh = fp.courtyard_hw, fp.courtyard_hh
    if quadrant % 2 == 1:
        hw, hh = hh, hw
    return pads, (hw, hh)


FOOTPRINTS = {
    "PASSIVE_SMALL": Footprint(
        name="PASSIVE_SMALL",
        pad_offsets=[(-0.5, 0.0), (0.5, 0.0)],
        pad_radius=0.35,
        courtyard_hw=0.8, courtyard_hh=0.4,
    ),
    "PASSIVE_LARGE": Footprint(
        name="PASSIVE_LARGE",
        pad_offsets=[(-1.0, 0.0), (1.0, 0.0)],
        pad_radius=0.5,
        courtyard_hw=1.6, courtyard_hh=0.8,
    ),
    "HEADER_4": _header(4),
    "HEADER_6": _header(6),
    "HEADER_8": _header(8),
    "IC_SOIC8": Footprint(
        name="IC_SOIC8",
        pad_offsets=[(-1.905, -2.0), (-0.635, -2.0), (0.635, -2.0), (1.905, -2.0),
                     (-1.905, 2.0), (-0.635, 2.0), (0.635, 2.0), (1.905, 2.0)],
        pad_radius=0.3,
        courtyard_hw=2.5, courtyard_hh=2.0,
        power_pins=[4],
        body_radius=1.3,   # clear of both pin rows (y=+-2.0) with margin
    ),
    "IC_QFN16": _qfn16(),
}
