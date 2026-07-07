"""Live training-visualization helpers for the Colab notebook.

Kept separate from render.py (which is the headless CLI renderer) because
this module assumes an interactive/inline matplotlib backend and redraws
in place -- importing it does not touch the backend at all.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

from .board import Board
from .render import _net_color


class TrainingMonitor:
    """Call `.update(record)` once per epoch (= one PPO rollout+update).
    Redraws a 3-panel figure in place: completion rate, episode return,
    and policy entropy / value loss, each vs. total env steps."""

    def __init__(self, advance_at: float = 0.95):
        self.history: List[Dict] = []
        self.advance_at = advance_at

    def update(self, record: Dict):
        self.history.append(record)
        steps = [h["steps_done"] for h in self.history]
        stages = np.array([h["stage"] for h in self.history])

        # Colab's inline backend auto-closes the figure after each display
        # (InlineBackend.close_figures=True), so reusing one fig/axes across
        # calls silently stops rendering after the first epoch. Build a
        # fresh figure every time instead, same as show_board_inline below.
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        ax.plot(steps, [h["completion"] for h in self.history], color="tab:blue")
        ax.axhline(self.advance_at, color="gray", ls="--", lw=1,
                  label=f"advance @ {self.advance_at:.0%}")
        ax.set_title("net completion rate (rolling 20 ep)")
        ax.set_xlabel("env steps"); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8)

        ax = axes[1]
        ax.plot(steps, [h["ep_return"] for h in self.history], color="tab:orange")
        ax.set_title("mean episode return"); ax.set_xlabel("env steps")

        ax = axes[2]
        ax.plot(steps, [h["entropy"] for h in self.history], color="tab:green",
               label="entropy")
        ax.set_ylabel("entropy"); ax.set_xlabel("env steps")
        ax2 = ax.twinx()
        ax2.plot(steps, [h["v_loss"] for h in self.history], color="tab:red",
                alpha=0.6, label="value loss")
        ax2.set_ylabel("value loss")
        ax.set_title("entropy (green) / value loss (red)")

        # Mark curriculum stage transitions on all three panels.
        change_idx = np.nonzero(np.diff(stages))[0]
        for i in change_idx:
            for a in axes:
                a.axvline(steps[i + 1], color="k", ls=":", lw=1)

        fig.suptitle(
            f"steps {steps[-1]:,}  |  stage {stages[-1]}  |  "
            f"completion {self.history[-1]['completion']:.1%}  |  "
            f"return {self.history[-1]['ep_return']:.1f}")
        fig.tight_layout()

        clear_output(wait=True)
        plt.show()
        plt.close(fig)


def show_board_inline(board: Board, title: str = "", completed=None, figsize=(6, 6)):
    """Render one board snapshot inline (no file I/O) -- a policy-rollout
    'photo' to sanity-check what the agent is actually doing."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-1, board.width + 1); ax.set_ylim(-1, board.height + 1)
    ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle((0, 0), board.width, board.height,
                               fill=False, ec="black", lw=1.5))
    for (x, y, r) in board.keepouts:
        ax.add_patch(plt.Circle((x, y), r, fc="0.75", ec="0.4", zorder=1))
    for (ax_, ay, bx, by, hw, layer, net) in board.traces:
        style = "-" if layer == 0 else "--"
        ax.plot([ax_, bx], [ay, by], style, color=_net_color(net),
                lw=max(2 * hw * 72 / 8, 1.6),
                alpha=1.0 if layer == 0 else 0.55, solid_capstyle="round", zorder=2)
    for (x, y, r, lo, hi, net) in board.vias:
        ax.add_patch(plt.Circle((x, y), r, fc="white", ec=_net_color(net),
                                lw=1.5, zorder=4))
    completed = set(completed or [])
    for p in board.pads:
        ec = "green" if p.net_id in completed else "black"
        ax.add_patch(plt.Circle((p.x, p.y), p.r, fc=_net_color(p.net_id),
                                ec=ec, lw=1.5, zorder=3))
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    plt.show()
    plt.close(fig)
