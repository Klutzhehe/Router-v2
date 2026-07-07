"""Continuous-space geometry kernel (pure NumPy, gridless).

Every obstacle on the board reduces to two primitives:
  disc    -- (center, radius): pads, vias, round keep-outs
  capsule -- (segment a-b, radius): routed traces (thick segments)

The core query is a *swept-disc cast*: the routing head is a disc of radius
(trace_width/2 + clearance) moving along a ray; we need the distance to first
contact with each inflated obstacle. All functions are vectorized over
(K rays x M obstacles). Units: millimetres.
"""

from __future__ import annotations

import numpy as np

INF = np.inf
_EPS = 1e-12
_TOL = 1e-6


def unit_dirs(n_bins: int) -> np.ndarray:
    """(n_bins, 2) unit vectors; bin i points at angle 2*pi*i/n_bins."""
    ang = np.arange(n_bins) * (2.0 * np.pi / n_bins)
    return np.stack([np.cos(ang), np.sin(ang)], axis=-1)


def point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points p (...,2) to segments a-b (broadcastable, (...,2))."""
    ab = b - a
    l2 = (ab * ab).sum(-1)
    ap = p - a
    u = np.where(l2 > _EPS, (ap * ab).sum(-1) / np.maximum(l2, _EPS), 0.0)
    u = np.clip(u, 0.0, 1.0)
    closest = a + u[..., None] * ab
    d = p - closest
    return np.sqrt((d * d).sum(-1))


def ray_disc_first_hit(origin: np.ndarray, dirs: np.ndarray,
                       centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """First-contact distance t (K, M) of rays vs discs.

    origin (2,), dirs (K,2) unit, centers (M,2), radii (M,).
    t = 0 if the origin is already inside a disc; inf if the ray never hits.
    """
    K = dirs.shape[0]
    if centers.shape[0] == 0:
        return np.full((K, 0), INF)
    m = origin[None, :] - centers                     # (M,2)
    b = dirs @ m.T                                    # (K,M)
    c0 = (m * m).sum(-1) - radii ** 2                 # (M,)
    h = b * b - c0[None, :]
    sqrt_h = np.sqrt(np.maximum(h, 0.0))
    t1 = -b - sqrt_h                                  # near root
    t = np.where(h < 0.0, INF, np.where(t1 >= 0.0, t1, INF))
    return np.where(c0[None, :] <= 0.0, 0.0, t)


def ray_capsule_first_hit(origin: np.ndarray, dirs: np.ndarray,
                          seg_a: np.ndarray, seg_b: np.ndarray,
                          radii: np.ndarray) -> np.ndarray:
    """First-contact distance t (K, M) of rays vs capsules (segment + radius).

    First contact with a capsule boundary happens either on an end-cap arc
    (ray-vs-disc at each endpoint) or on a straight side (crossing of the two
    offset lines at signed distance +-r from the axis). We evaluate all four
    candidates and keep the smallest one that actually lies on the capsule.
    """
    K, M = dirs.shape[0], seg_a.shape[0]
    if M == 0:
        return np.full((K, 0), INF)

    ab = seg_b - seg_a
    L = np.sqrt((ab * ab).sum(-1))                    # (M,)
    degenerate = L < 1e-9
    ab_hat = ab / np.maximum(L, 1e-9)[:, None]
    n = np.stack([-ab_hat[:, 1], ab_hat[:, 0]], -1)   # unit normal (M,2)

    s0 = ((origin[None, :] - seg_a) * n).sum(-1)      # signed offset (M,)
    sd = dirs @ n.T                                   # d(offset)/dt (K,M)
    safe = np.abs(sd) > _EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = (radii[None, :] - s0[None, :]) / sd
        tm = (-radii[None, :] - s0[None, :]) / sd
    tp = np.where(safe, tp, INF)
    tm = np.where(safe, tm, INF)

    tda = ray_disc_first_hit(origin, dirs, seg_a, radii)
    tdb = ray_disc_first_hit(origin, dirs, seg_b, radii)

    cands = np.stack([tp, tm, tda, tdb], axis=-1)     # (K,M,4)
    finite = np.isfinite(cands) & (cands >= 0.0)
    t_eval = np.where(finite, cands, 0.0)
    p = origin + t_eval[..., None] * dirs[:, None, None, :]          # (K,M,4,2)
    d_seg = point_seg_dist(p, seg_a[None, :, None, :], seg_b[None, :, None, :])
    on_capsule = finite & (d_seg <= radii[None, :, None] + _TOL)
    t = np.where(on_capsule, t_eval, INF).min(axis=-1)               # (K,M)

    # Origin already inside the inflated capsule -> zero legal distance.
    d0 = point_seg_dist(origin[None, :], seg_a, seg_b)               # (M,)
    t = np.where((d0 <= radii)[None, :], 0.0, t)

    if degenerate.any():
        t = np.where(degenerate[None, :], tda, t)
    return t


def rect_inset_exit(origin: np.ndarray, dirs: np.ndarray,
                    lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Largest t (K,) keeping origin + t*dir inside the box [lo, hi].

    Used for the board outline: lo/hi are the outline inset by
    (trace_width/2 + board_clearance). Returns 0 if origin is outside.
    """
    if np.any(origin < lo) or np.any(origin > hi):
        return np.zeros(dirs.shape[0])
    with np.errstate(divide="ignore"):
        t_hi = np.where(dirs > _EPS, (hi - origin) / dirs, INF)
        t_lo = np.where(dirs < -_EPS, (lo - origin) / dirs, INF)
    t = np.minimum(t_hi, t_lo).min(axis=-1)
    return np.maximum(t, 0.0)
