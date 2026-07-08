"""Legality fuzz: random *masked* actions must never trigger a DRC flag,
and independently re-checked clearances must hold for all placed copper."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcb_router import geometry as geo                       # noqa: E402
from pcb_router.config import EnvConfig, N_MAX_PINS           # noqa: E402
from pcb_router.env import RoutingEnv                         # noqa: E402
from pcb_router.generator import STAGES, generate_board       # noqa: E402


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


def test_masked_actions_are_legal(episodes=3, seed=7):
    rng = np.random.default_rng(seed)
    for stage in range(len(STAGES)):
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
    assert obs["node_feats"].shape == (64, 10)
    assert obs["adj"].shape == (64, 64)
    assert obs["points"].shape == (256, 10)
    assert obs["head_state"].shape == (22,)
    assert masks["type"].shape == (3,) and masks["angle"].shape == (64,)
    assert masks["layer"].shape == (12,)
    assert masks["type"].any(), "stage-0 start must have a legal action"
    print("test_obs_shapes OK")


def test_generator_smoke(seed=11):
    """Board generation invariants across every curriculum stage: pad/net
    count stays within N_MAX_PINS, every pad resolves to a valid net, and
    differential pairs (where present) share trace width and route
    back-to-back once RoutingEnv splices the net order."""
    for stage in range(len(STAGES)):
        rng = np.random.default_rng(seed + stage)
        board = generate_board(stage, rng)
        n_pads, n_nets = len(board.pads), len(board.nets)
        assert n_pads == 2 * n_nets
        assert n_pads <= N_MAX_PINS, f"stage {stage}: {n_pads} pads > N_MAX_PINS"
        for p in board.pads:
            assert 0 <= p.net_id < n_nets, f"stage {stage}: pad has invalid net_id"

        by_pair = {}
        for net in board.nets:
            if net.pair_id is not None:
                by_pair.setdefault(net.pair_id, []).append(net)
        for pid, nets in by_pair.items():
            assert len(nets) == 2, f"stage {stage} pair {pid}: expected 2 nets, got {len(nets)}"
            assert nets[0].trace_width == nets[1].trace_width, \
                f"stage {stage} pair {pid}: mismatched trace width"

        env = RoutingEnv(lambda r, s=stage: generate_board(s, r), cfg=EnvConfig(),
                         seed=seed + stage)
        env.board = board
        order = env._splice_diff_pairs(sorted(range(n_nets), key=lambda i: board.nets[i].hpwl))
        pos = {idx: i for i, idx in enumerate(order)}
        for pid, nets in by_pair.items():
            i0 = board.nets.index(nets[0])
            i1 = board.nets.index(nets[1])
            assert abs(pos[i0] - pos[i1]) == 1, \
                f"stage {stage} pair {pid}: nets not adjacent in route order"
        print(f"stage {stage}: pads={n_pads} nets={n_nets} "
              f"keepouts={len(board.keepouts)} diff_pairs={len(by_pair)}")
    print("test_generator_smoke OK")


if __name__ == "__main__":
    test_obs_shapes()
    test_generator_smoke()
    test_masked_actions_are_legal()
    print("env: all tests passed")
