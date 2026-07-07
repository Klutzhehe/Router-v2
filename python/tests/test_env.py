"""Legality fuzz: random *masked* actions must never trigger a DRC flag,
and independently re-checked clearances must hold for all placed copper."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcb_router import geometry as geo                       # noqa: E402
from pcb_router.config import EnvConfig                      # noqa: E402
from pcb_router.env import RoutingEnv                        # noqa: E402
from pcb_router.generator import generate_board              # noqa: E402


def random_legal_action(masks, rng):
    legal_types = np.nonzero(masks["type"])[0]
    t = int(rng.choice(legal_types))
    angle = int(rng.choice(np.nonzero(masks["angle"])[0])) if masks["angle"].any() else 0
    layer = int(rng.choice(np.nonzero(masks["layer"])[0])) if masks["layer"].any() else 0
    return (t, angle, float(rng.random()), layer)


def verify_clearances(board):
    """Independent O(n^2) DRC over final copper: every trace segment must
    clear every foreign obstacle by trace_clearance (minus float safety)."""
    arr = board.arrays()
    rules = board.rules
    worst = np.inf
    for (ax, ay, bx, by, hw, lay, net) in board.traces:
        a, b = np.array([ax, ay]), np.array([bx, by])
        # against foreign discs on this layer
        sel = (arr.disc_net != net) & (arr.disc_llo <= lay) & (arr.disc_lhi >= lay)
        if sel.any():
            d = geo.point_seg_dist(arr.disc_c[sel], a, b) - arr.disc_r[sel] - hw
            worst = min(worst, d.min())
        # against foreign capsules on this layer (sampled along our axis)
        selc = (arr.cap_net != net) & (arr.cap_layer == lay)
        if selc.any():
            for u in np.linspace(0, 1, 16):
                p = a + u * (b - a)
                d = geo.point_seg_dist(p[None, :], arr.cap_a[selc], arr.cap_b[selc]) \
                    - arr.cap_r[selc] - hw
                worst = min(worst, d.min())
    assert worst == np.inf or worst >= rules.trace_clearance - 1e-3, \
        f"clearance violated: min gap {worst:.6f} < {rules.trace_clearance}"
    return worst


def test_masked_actions_are_legal(episodes=5, seed=7):
    rng = np.random.default_rng(seed)
    for stage in (0, 1, 3):
        for ep in range(episodes):
            env = RoutingEnv(lambda r, s=stage: generate_board(s, r),
                             cfg=EnvConfig(), seed=seed + ep + 100 * stage)
            obs, masks = env.reset()
            done, steps = False, 0
            while not done and steps < 3000:
                a = random_legal_action(masks, rng)
                obs, masks, r, done, info = env.step(a)
                assert np.isfinite(r)
                steps += 1
            assert done, "episode did not terminate"
            assert info["drc"] == 0, f"DRC fired {info['drc']} times"
            gap = verify_clearances(env.board)
            print(f"stage {stage} ep {ep}: {info['nets_done']}/{info['nets_total']} "
                  f"nets by random walk, {steps} steps, min gap "
                  f"{'inf' if gap == np.inf else f'{gap:.3f}mm'}, DRC=0")
    print("test_masked_actions_are_legal OK")


def test_obs_shapes(seed=3):
    env = RoutingEnv(lambda r: generate_board(0, r), seed=seed)
    obs, masks = env.reset()
    assert obs["node_feats"].shape == (64, 8)
    assert obs["adj"].shape == (64, 64)
    assert obs["points"].shape == (256, 10)
    assert obs["head_state"].shape == (19,)
    assert masks["type"].shape == (3,) and masks["angle"].shape == (64,)
    assert masks["layer"].shape == (12,)
    assert masks["type"].any(), "stage-0 start must have a legal action"
    print("test_obs_shapes OK")


if __name__ == "__main__":
    test_obs_shapes()
    test_masked_actions_are_legal()
    print("env: all tests passed")
