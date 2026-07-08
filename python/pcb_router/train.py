"""Curriculum training entry point.

    python -m pcb_router.train --stage 0 --total-steps 200000 --n-envs 16

The curriculum auto-advances only once the rolling completion rate clears
--advance-at (default 0.99) over a full --advance-window episodes AND stays
there for --advance-streak consecutive PPO updates in a row -- a single
lucky rollout no longer promotes the stage (see CLAUDE.md / the curriculum
overhaul notes for why: every stage here is a 2+-layer board, so "mostly
works" isn't good enough before moving on).

Checkpoints are a single .pt file per run containing model weights,
optimizer state, curriculum stage, and step count (see ppo.save_checkpoint) --
resuming with --resume restores all of it, not just the weights.

--n-envs > 1 batches model.act across N parallel boards each step instead of
1 -- on a GPU this is the difference between the GPU sitting mostly idle
(single-sample forward passes dominated by launch/transfer overhead) and it
actually being used. Costs more host RAM (N boards' worth of obstacle
arrays); has no effect on the RL algorithm or reward math, only throughput.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .config import EnvConfig
from .env import RoutingEnv, VecRoutingEnv
from .generator import STAGES, generate_board
from .model import DualStreamRouter
from .ppo import (PPO, PPOConfig, collect_rollout, collect_rollout_vec,
                  load_checkpoint, save_checkpoint)


def make_env(stage: int, n_envs: int, seed: int):
    factory = lambda rng, s=stage: generate_board(s, rng)
    if n_envs > 1:
        return VecRoutingEnv(factory, n_envs, cfg=EnvConfig(), seed=seed + stage)
    return RoutingEnv(factory, cfg=EnvConfig(), seed=seed + stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=200_000)
    ap.add_argument("--rollout", type=int, default=8192,
                    help="total env-steps collected per PPO update "
                         "(split across --n-envs when > 1)")
    ap.add_argument("--n-envs", type=int, default=16,
                    help="parallel boards per step; >1 batches model.act "
                         "for GPU throughput (try 16-32 on a Colab GPU)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-dir", default="checkpoints")
    ap.add_argument("--run-name", default="router",
                    help="checkpoint/log file prefix, e.g. router -> router.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--advance-at", type=float, default=0.99,
                    help="rolling completion rate required before advancing")
    ap.add_argument("--advance-window", type=int, default=50,
                    help="episodes averaged into the rolling completion rate")
    ap.add_argument("--advance-streak", type=int, default=3,
                    help="consecutive qualifying updates required to advance "
                         "(prevents a single lucky window from promoting)")
    ap.add_argument("--resume", default=None, help="checkpoint .pt to load")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{args.run_name}.pt"
    log_path = save_dir / f"{args.run_name}_history.jsonl"

    stage = args.stage
    model = DualStreamRouter()
    ppo = PPO(model, PPOConfig(), device=device)

    steps_done = 0
    completions = deque(maxlen=max(args.advance_window, args.n_envs))
    consecutive_hits = 0
    resume_from = args.resume or (str(ckpt_path) if ckpt_path.exists() else None)
    if resume_from:
        ckpt = load_checkpoint(resume_from, model, ppo, device=device)
        stage = ckpt.get("stage", stage)
        steps_done = ckpt.get("steps_done", 0)
        completions.extend(ckpt.get("completions", []))
        consecutive_hits = ckpt.get("consecutive_hits", 0)
        print(f"resumed from {resume_from}: stage={stage} steps_done={steps_done} "
              f"consecutive_hits={consecutive_hits}")

    env = make_env(stage, args.n_envs, args.seed)
    steps_per_env = max(args.rollout // args.n_envs, 1) if args.n_envs > 1 else args.rollout

    carried = (None, None)
    print(f"device={device} stage={stage} n_envs={args.n_envs} "
          f"({STAGES[min(stage, len(STAGES)-1)].layers} layers)", flush=True)

    while steps_done < args.total_steps:
        t0 = time.time()
        if args.n_envs > 1:
            buf, stats, carried = collect_rollout_vec(env, model, steps_per_env,
                                                       device, *carried, ppo.cfg)
            steps_this_update = steps_per_env * args.n_envs
        else:
            buf, stats, carried = collect_rollout(env, model, args.rollout, device,
                                                  *carried, ppo.cfg)
            steps_this_update = args.rollout
        upd = ppo.update(buf, steps_done)
        steps_done += steps_this_update
        completions.extend(stats["completions"])

        sps = steps_this_update / (time.time() - t0)
        mean_ret = np.mean(stats["returns"]) if stats["returns"] else float("nan")
        mean_cmp = np.mean(completions) if completions else 0.0
        drc_total = int(sum(stats["drc"]))
        # Net count is RNG-dependent now (component placement can fail, the
        # general pool trims an odd leftover pad) -- no static per-stage
        # count to read, so average whatever finished this rollout.
        mean_nets_total = np.mean(stats["nets_total"]) if stats["nets_total"] else float("nan")
        mean_nets_routed = mean_cmp * mean_nets_total
        print(f"steps {steps_done:>8}  stage {stage}  "
              f"ep_return {mean_ret:8.2f}  completion {mean_cmp:5.1%} "
              f"({mean_nets_routed:.2f}/{mean_nets_total:.1f} nets)  "
              f"entropy {upd['entropy']:6.3f}  pi {upd['pi_loss']:+.4f}  "
              f"v {upd['v_loss']:8.3f}  clip {upd['clip_frac']:5.1%}  drc {drc_total}  "
              f"commit_rate {stats['commit_rate']:5.1%}  {sps:6.0f} steps/s", flush=True)
        if upd["clip_frac"] > 0.3:
            print(f"  ! clip_frac={upd['clip_frac']:.1%} is high -- updates may be too "
                  f"large/destabilizing. If ep_return/completion crash and stay down "
                  f"after a run of high clip_frac, that's why.", flush=True)
        if drc_total > 0:
            print(f"  !! DRC={drc_total} this rollout -- geometry-kernel bug, "
                  f"not a hyperparameter issue. See CLAUDE.md invariant #2.", flush=True)

        with open(log_path, "a") as f:
            f.write(json.dumps({"steps_done": steps_done, "stage": stage,
                                "ep_return": mean_ret, "completion": mean_cmp,
                                "drc": drc_total, "commit_rate": stats["commit_rate"],
                                **upd}) + "\n")

        # Consistency gate: the window must be full AND clear advance_at,
        # AND that has to hold for advance_streak updates in a row -- a
        # single lucky window no longer promotes the stage.
        qualifies = len(completions) == completions.maxlen and mean_cmp >= args.advance_at
        consecutive_hits = consecutive_hits + 1 if qualifies else 0
        save_checkpoint(ckpt_path, model, ppo, stage, steps_done, completions,
                        consecutive_hits=consecutive_hits)

        if consecutive_hits >= args.advance_streak and stage < len(STAGES) - 1:
            stage += 1
            print(f"=== curriculum advance -> stage {stage} "
                  f"(after {args.advance_streak} consecutive qualifying updates) ===", flush=True)
            env = make_env(stage, args.n_envs, args.seed)
            completions.clear()
            consecutive_hits = 0
            carried = (None, None)
            save_checkpoint(ckpt_path, model, ppo, stage, steps_done, completions,
                            consecutive_hits=consecutive_hits)

    print("training complete")


if __name__ == "__main__":
    main()
