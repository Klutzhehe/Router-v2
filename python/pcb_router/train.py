"""Curriculum training entry point.

    python -m pcb_router.train --stage 0 --total-steps 200000

The curriculum auto-advances when the rolling completion rate clears
--advance-at (default 0.95) over the last 20 episodes.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .config import EnvConfig
from .env import RoutingEnv
from .generator import STAGES, generate_board
from .model import DualStreamRouter
from .ppo import PPO, PPOConfig, collect_rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=200_000)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--advance-at", type=float, default=0.95)
    ap.add_argument("--resume", default=None, help="checkpoint .pt to load")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    torch.manual_seed(args.seed)

    stage = args.stage
    env = RoutingEnv(lambda rng: generate_board(stage, rng),
                     cfg=EnvConfig(), seed=args.seed)
    model = DualStreamRouter()
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed from {args.resume}")
    ppo = PPO(model, PPOConfig(), device=device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    completions = deque(maxlen=20)
    steps_done, carried = 0, (None, None)
    print(f"device={device} stage={stage} "
          f"({STAGES[min(stage, len(STAGES)-1)].n_nets} nets, "
          f"{STAGES[min(stage, len(STAGES)-1)].layers} layers)", flush=True)

    while steps_done < args.total_steps:
        t0 = time.time()
        buf, stats, carried = collect_rollout(env, model, args.rollout, device,
                                              *carried)
        upd = ppo.update(buf)
        steps_done += args.rollout
        completions.extend(stats["completions"])

        sps = args.rollout / (time.time() - t0)
        mean_ret = np.mean(stats["returns"]) if stats["returns"] else float("nan")
        mean_cmp = np.mean(completions) if completions else 0.0
        print(f"steps {steps_done:>8}  stage {stage}  "
              f"ep_return {mean_ret:8.2f}  completion {mean_cmp:5.1%}  "
              f"entropy {upd['entropy']:6.3f}  pi {upd['pi_loss']:+.4f}  "
              f"v {upd['v_loss']:8.3f}  {sps:6.0f} steps/s", flush=True)

        torch.save(model.state_dict(), save_dir / f"router_stage{stage}.pt")

        if (len(completions) == completions.maxlen
                and mean_cmp >= args.advance_at
                and stage < len(STAGES) - 1):
            stage += 1
            print(f"=== curriculum advance -> stage {stage} ===", flush=True)
            env = RoutingEnv(lambda rng, s=stage: generate_board(s, rng),
                             cfg=EnvConfig(), seed=args.seed + stage)
            completions.clear()
            carried = (None, None)

    print("training complete")


if __name__ == "__main__":
    main()
