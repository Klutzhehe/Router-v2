"""Legality fuzz: random *masked* actions must never trigger a DRC flag,
and independently re-checked clearances must hold for all placed copper."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcb_router import geometry as geo                       # noqa: E402
from pcb_router.config import (DIST_FRACTIONS, EnvConfig,     # noqa: E402
                               N_ANGLE_BINS, N_MAX_PINS)
from pcb_router.env import RoutingEnv                         # noqa: E402
from pcb_router.generator import STAGES, generate_board       # noqa: E402

_DIRS = geo.unit_dirs(N_ANGLE_BINS)


def random_legal_action(masks, rng):
    legal_types = np.nonzero(masks["type"])[0]
    t = int(rng.choice(legal_types))
    angle = int(rng.choice(np.nonzero(masks["angle"])[0])) if masks["angle"].any() else 0
    layer = int(rng.choice(np.nonzero(masks["layer"])[0])) if masks["layer"].any() else 0
    # dist is a discrete bin index; sampling all bins matters because the
    # long bins (full max_distance) are where float-robustness bugs live.
    return (t, angle, int(rng.integers(len(DIST_FRACTIONS))), layer)


def check_frame_alignment(env):
    """Canonical bin 0, decoded to world coordinates, must point at the
    target to within half an angle bin -- the invariant the whole
    target-aligned frame rests on."""
    if env.mask is None or env.head is None:
        return
    to_tgt = np.array([env.head.target_x - env.head.x,
                       env.head.target_y - env.head.y])
    d = np.linalg.norm(to_tgt)
    if d < 1e-9:
        return
    cos = float(_DIRS[env.mask.frame_offset] @ to_tgt) / d
    assert cos >= np.cos(np.pi / N_ANGLE_BINS) - 1e-9, \
        f"frame_offset {env.mask.frame_offset} misaligned: cos={cos:.6f}"


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
                check_frame_alignment(env)
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


def test_canonical_frame_oracle(boards=30, seed=11):
    """The point of the target-aligned frame: the trivial policy 'COMMIT when
    legal, else take the legal bin nearest canonical 0 with the step length
    closest to the remaining distance' should route most stage-0 nets
    without any learning. If completion craters well below the threshold,
    straight-line routing is not expressible as the constant action the
    canonicalization promises -- that's the bug this test exists to catch.

    It won't hit 100%: this oracle has no lookahead, so when an obstacle
    caps max_distance in its best-aimed bin from both sides, it can settle
    into a stable back-and-forth 2-cycle straddling the obstacle (confirmed
    by tracing a failing seed -- the head bounces between two fixed points
    forever, each hop the full capped distance). Real component footprints
    (courtyards, a sibling pin ~1mm from every route start/end) make this
    more likely than the old floating-random-pad generator did. That's a
    known limitation of a lookahead-free heuristic, not a masking/DRC bug --
    test_masked_actions_are_legal separately guarantees DRC=0 regardless."""
    done_nets = total_nets = 0
    for ep in range(boards):
        env = RoutingEnv(lambda r: generate_board(0, r),
                         cfg=EnvConfig(), seed=seed + ep)
        obs, masks = env.reset()
        done, steps = False, 0
        while not done and steps < 2000:
            if masks["type"][2]:                       # COMMIT
                a = (2, 0, 0, 0)
            elif masks["type"][0]:                     # EXTEND toward target
                legal = np.nonzero(masks["angle"])[0]
                bin_ = int(legal[np.argmin(np.minimum(legal,
                                                      N_ANGLE_BINS - legal))])
                d_rem = np.hypot(env.head.target_x - env.head.x,
                                 env.head.target_y - env.head.y)
                mn = env.board.rules.min_segment_length
                dmax = env.mask.max_distance[bin_]
                step_mm = mn + np.asarray(DIST_FRACTIONS) * (dmax - mn)
                j = int(np.argmin(np.abs(step_mm - d_rem)))
                a = (0, bin_, j, 0)
            else:                                      # burn budget legally
                t = int(np.nonzero(masks["type"])[0][0])
                lay = int(np.nonzero(masks["layer"])[0][0]) \
                    if masks["layer"].any() else 0
                a = (t, 0, 0, lay)
            obs, masks, r, done, info = env.step(a)
            steps += 1
        assert done, "oracle episode did not terminate"
        assert info["drc"] == 0, f"oracle triggered DRC {info['drc']} times"
        done_nets += info["nets_done"]
        total_nets += info["nets_total"]
    rate = done_nets / total_nets
    print(f"test_canonical_frame_oracle: {done_nets}/{total_nets} nets "
          f"({rate:.0%}) by the bin-0 oracle")
    assert rate >= 0.7, f"oracle completion {rate:.0%} < 70%"


def test_canonical_obs_alignment(seed=13, max_steps=300):
    """After the canonical-frame rotation, the target's row in the egocentric
    point cloud must lie on the +x axis to within half a bin, in every state
    of a random walk."""
    rng = np.random.default_rng(seed)
    env = RoutingEnv(lambda r: generate_board(0, r), cfg=EnvConfig(), seed=seed)
    obs, masks = env.reset()
    half_bin = np.pi / N_ANGLE_BINS
    checked, done, steps = 0, False, 0
    while not done and steps < max_steps:
        if env.head is not None:
            d = np.hypot(env.head.target_x - env.head.x,
                         env.head.target_y - env.head.y)
            if d > 1e-6:
                rows = np.nonzero(obs["points"][:, 9] > 0.5)[0]
                assert len(rows) >= 1, "target point missing from cloud"
                x, y = obs["points"][rows[0], 0:2]
                assert x > 0 and abs(np.arctan2(y, x)) <= half_bin + 1e-6, \
                    f"target at canonical angle {np.degrees(np.arctan2(y, x)):.2f} deg"
                hs = obs["head_state"]
                assert hs[2] > 0 and hs[2] >= abs(hs[3]), \
                    "head_state target direction not canonical"
                checked += 1
        a = random_legal_action(masks, rng)
        obs, masks, r, done, info = env.step(a)
        steps += 1
    assert checked > 0
    print(f"test_canonical_obs_alignment OK ({checked} states checked)")


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
    test_canonical_obs_alignment()
    test_canonical_frame_oracle()
    test_masked_actions_are_legal()
    print("env: all tests passed")
