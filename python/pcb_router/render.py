"""Board renderer + policy rollout demo.

    python -m pcb_router.render --stage 0 --out board.png                # random walk
    python -m pcb_router.render --stage 0 --checkpoint checkpoints/router_stage0.pt \
        --out routed.png --deterministic

Draws pads/traces colored per net, vias as ringed dots, keep-outs in gray.
Top-layer copper is solid; inner/bottom layers get dashed lines and lower alpha.
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

from .board import Board                  # noqa: E402
from .config import EnvConfig             # noqa: E402
from .env import RoutingEnv               # noqa: E402
from .generator import generate_board     # noqa: E402

_CMAP = plt.get_cmap("tab20")


def _net_color(net_id: int):
    return _CMAP(net_id % 20)


def render_board(board: Board, path: str, title: str = "", completed=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1, board.width + 1)
    ax.set_ylim(-1, board.height + 1)
    ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle((0, 0), board.width, board.height,
                               fill=False, ec="black", lw=1.5))

    for (x, y, r) in board.keepouts:
        ax.add_patch(plt.Circle((x, y), r, fc="0.75", ec="0.4", zorder=1))

    for (ax_, ay, bx, by, hw, layer, net) in board.traces:
        style = "-" if layer == 0 else "--"
        alpha = 1.0 if layer == 0 else 0.55
        ax.plot([ax_, bx], [ay, by], style, color=_net_color(net),
                lw=max(2 * hw * 72 / 8, 1.6), alpha=alpha,
                solid_capstyle="round", zorder=2)

    for (x, y, r, lo, hi, net) in board.vias:
        ax.add_patch(plt.Circle((x, y), r, fc="white", ec=_net_color(net),
                                lw=1.5, zorder=4))
        ax.plot(x, y, ".", color="black", ms=3, zorder=5)

    completed = set(completed or [])
    for p in board.pads:
        ec = "green" if p.net_id in completed else "black"
        ax.add_patch(plt.Circle((p.x, p.y), p.r, fc=_net_color(p.net_id),
                                ec=ec, lw=1.5, zorder=3))
        if p.layer_lo > 0:
            ax.text(p.x, p.y, str(p.layer_lo), ha="center", va="center",
                    fontsize=6, zorder=6)

    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def rollout_and_render(stage: int, out: str, checkpoint: str | None,
                       seed: int, deterministic: bool):
    env = RoutingEnv(lambda r: generate_board(stage, r),
                     cfg=EnvConfig(), seed=seed)
    obs, masks = env.reset()

    if checkpoint:
        import torch
        from .model import DualStreamRouter
        from .ppo import to_torch
        model = DualStreamRouter()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.eval()
        done = False
        while not done:
            t_masks = {k: torch.from_numpy(v).unsqueeze(0)
                       for k, v in masks.items()}
            a, _, _ = model.act(to_torch(obs, "cpu"), t_masks,
                                deterministic=deterministic)
            obs, masks, _, done, info = env.step(
                (int(a.action_type), int(a.angle_bin),
                 float(a.dist_frac), int(a.layer)))
        who = "policy"
    else:
        rng = np.random.default_rng(seed)
        done = False
        while not done:
            legal_t = np.nonzero(masks["type"])[0]
            t = int(rng.choice(legal_t))
            ang = int(rng.choice(np.nonzero(masks["angle"])[0])) if masks["angle"].any() else 0
            lay = int(rng.choice(np.nonzero(masks["layer"])[0])) if masks["layer"].any() else 0
            obs, masks, _, done, info = env.step((t, ang, float(rng.random()), lay))
        who = "random walk"

    title = (f"stage {stage} | {who} | "
             f"{info['nets_done']}/{info['nets_total']} nets | DRC={info['drc']}")
    print(title)
    render_board(env.board, out, title=title, completed=env.completed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--out", default="board.png")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    rollout_and_render(args.stage, args.out, args.checkpoint, args.seed,
                       args.deterministic)
