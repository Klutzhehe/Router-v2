# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Gridless PCB autorouter trained with PPO. The engine is **pure Python + NumPy**
(chosen so the whole project trains on Google Colab with zero installs); the
agent is pure PyTorch (deliberately no torch_geometric — Colab doesn't ship
it). See PROJECT.md for the full architecture; docs/reward-function.md for the
reward math and constants.

## Commands

All commands run from `python/`:

```bash
python tests/test_geometry.py      # analytic casts vs brute-force marching
python tests/test_env.py           # legality fuzz + independent DRC re-check
python tests/test_model_ppo.py     # model + PPO smoke tests
python -m pcb_router.train --stage 0 --total-steps 200000   # train
python -m pcb_router.train --stage 1 --resume checkpoints/router_stage0.pt
```

Tests are plain scripts with main guards (no pytest dependency). They insert
`python/` into sys.path themselves.

## Architecture in one pass

`generator.py` builds a curriculum board → `env.py` runs the MDP: one net at a
time (ascending HPWL), each step `masker.py` computes which of the 64
directions / via layers / commit are legal via swept-disc casts in
`geometry.py` against the disc+capsule obstacle arrays in `board.py` →
`model.py` (netlist GNN + egocentric PointNet → masked hybrid actor + critic)
picks an action → `ppo.py`/`train.py` learn from it. All obstacles are discs
(pads/vias/keep-outs) or capsules (traces); there is no grid anywhere.

## Invariants (the things that break silently)

1. **Legal-by-construction distances.** The actor emits a *fraction* ∈ (0,1);
   `env.py` maps it to `[min_segment_length, mask.max_distance[bin]]`. Never
   interpret the policy's distance output as absolute millimetres.
2. **A DRC flag during training is a geometry bug, not a tuning problem.**
   Masking makes illegal actions unsamplable; `info["drc"]` > 0 means fix
   `geometry.py`/`masker.py`. `tests/test_env.py` fuzzes exactly this and
   independently re-checks clearances on the final copper.
3. **Action-space constants live in `config.py`** (`N_ANGLE_BINS=64`,
   `MAX_LAYERS=12`, `N_ACTION_TYPES=3`) and are consumed by masker, env, and
   model. `cpp/include/pcb/action_masker.hpp` is a *specification for a future
   C++ port* — not built, not linked — and must be kept in sync with config.py
   if either changes.
4. **Observations are fixed-size padded tensors** (`N_MAX_PINS=64`, `P_MAX=256`)
   so PPO batching is trivial. Raising curriculum scale means raising these
   caps, which changes model input shapes and invalidates old checkpoints.
5. **Reward changes go through docs/reward-function.md first.** Length
   penalties are HPWL-normalized (detour factor) — keep them that way or
   curriculum transfer breaks. Shaping must stay potential-based.
6. **γ appears twice** — `RewardWeights.gamma` (shaping) and `PPOConfig.gamma`
   (GAE). Keep them equal.

## Conventions

- Units: millimetres, float64 in the engine, float32 in observations.
  Layer 0 = top copper; vias span an inclusive `[layer_lo, layer_hi]`.
- Angle bin i points at heading `2π·i/64`; indexing shared by masker, env
  decoder, and actor head.
- The env is deterministic given the board; all randomness lives in the
  generator and the policy (enables search-based inference later).
