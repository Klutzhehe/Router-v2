"""Brute-force validation of the analytic swept-disc casts."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcb_router import geometry as geo  # noqa: E402


def brute_force_first_hit(origin, direction, dist_fn, r, t_max=20.0, dt=1e-3):
    """March along the ray until dist(point) <= r."""
    ts = np.arange(0.0, t_max, dt)
    pts = origin[None, :] + ts[:, None] * direction[None, :]
    d = dist_fn(pts)
    hits = np.nonzero(d <= r)[0]
    return ts[hits[0]] if hits.size else np.inf


def test_ray_disc(n_trials=300, seed=1):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_trials):
        origin = rng.uniform(-5, 5, 2)
        ang = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(ang), np.sin(ang)])
        c = rng.uniform(-5, 5, size=(1, 2))
        r = np.array([rng.uniform(0.2, 2.0)])
        t = geo.ray_disc_first_hit(origin, d[None, :], c, r)[0, 0]
        bf = brute_force_first_hit(origin, d, lambda p: np.linalg.norm(p - c[0], axis=1), r[0])
        if np.isfinite(t) or np.isfinite(bf):
            err = abs(min(t, 20.0) - min(bf, 20.0))
            worst = max(worst, err)
            assert err < 5e-3, f"disc mismatch: analytic {t}, brute {bf}"
    print(f"test_ray_disc OK (worst err {worst:.2e})")


def test_ray_capsule(n_trials=300, seed=2):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_trials):
        origin = rng.uniform(-5, 5, 2)
        ang = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(ang), np.sin(ang)])
        a = rng.uniform(-5, 5, size=(1, 2))
        b = a + rng.uniform(-4, 4, size=(1, 2))
        r = np.array([rng.uniform(0.2, 1.5)])
        t = geo.ray_capsule_first_hit(origin, d[None, :], a, b, r)[0, 0]
        bf = brute_force_first_hit(
            origin, d, lambda p: geo.point_seg_dist(p, a[0], b[0]), r[0])
        if np.isfinite(t) or np.isfinite(bf):
            err = abs(min(t, 20.0) - min(bf, 20.0))
            worst = max(worst, err)
            assert err < 5e-3, f"capsule mismatch: analytic {t}, brute {bf}"
    print(f"test_ray_capsule OK (worst err {worst:.2e})")


def test_rect_exit():
    lo, hi = np.array([0.0, 0.0]), np.array([10.0, 10.0])
    origin = np.array([5.0, 5.0])
    dirs = geo.unit_dirs(8)
    t = geo.rect_inset_exit(origin, dirs, lo, hi)
    assert abs(t[0] - 5.0) < 1e-9          # +x
    assert abs(t[2] - 5.0) < 1e-9          # +y
    assert abs(t[1] - 5.0 * np.sqrt(2)) < 1e-9  # diagonal
    assert geo.rect_inset_exit(np.array([-1.0, 5.0]), dirs, lo, hi).max() == 0.0
    print("test_rect_exit OK")


if __name__ == "__main__":
    test_ray_disc()
    test_ray_capsule()
    test_rect_exit()
    print("geometry: all tests passed")
