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
python -m pcb_router.train --stage 0 --total-steps 200000 --save-dir checkpoints --run-name router
```

Re-running with the same `--save-dir`/`--run-name` auto-resumes (weights,
optimizer state, curriculum stage, and step count all round-trip through
`ppo.save_checkpoint`/`load_checkpoint`) — no separate `--resume` flag needed
unless loading a checkpoint under a different name. For Colab, use
`PCB_Router_Training.ipynb` at the repo root instead of the CLI: same
checkpoint format, plus live per-epoch plots and inline board renders.

Tests are plain scripts with main guards (no pytest dependency). They insert
`python/` into sys.path themselves.

## Architecture in one pass

`generator.py` builds a curriculum board → `env.py` runs the MDP: one net at a
time (ascending HPWL), each step `masker.py` computes which of the 128
directions / via layers / commit are legal via swept-disc casts in
`geometry.py` against the disc+capsule obstacle arrays in `board.py` →
`model.py` (netlist GNN + egocentric PointNet → masked hybrid actor + critic)
picks an action → `ppo.py`/`train.py` learn from it. All obstacles are discs
(pads/vias/keep-outs) or capsules (traces); there is no grid anywhere.

## Invariants (the things that break silently)

1. **Legal-by-construction distances.** The actor emits a distance *bin*;
   `env.py` maps `DIST_FRACTIONS[bin]` into `[min_segment_length,
   mask.max_distance[angle]]`. Never interpret the policy's distance output
   as absolute millimetres.
2. **A DRC flag during training is a geometry bug, not a tuning problem.**
   Masking makes illegal actions unsamplable; `info["drc"]` > 0 means fix
   `geometry.py`/`masker.py`. `tests/test_env.py` fuzzes exactly this and
   independently re-checks clearances on the final copper.
3. **Action-space constants live in `config.py`** (`N_ANGLE_BINS=128`,
   `MAX_LAYERS=12`, `N_ACTION_TYPES=3`) and are consumed by masker, env, and
   model. `cpp/include/pcb/action_masker.hpp` is a *specification for a future
   C++ port* — not built, not linked — and must be kept in sync with config.py
   if either changes.
4. **Observations are fixed-size padded tensors** (`N_MAX_PINS=64`, `P_MAX=256`)
   so PPO batching is trivial. Raising curriculum scale means raising these
   caps, which changes model input shapes and invalidates old checkpoints.
5. **Reward changes go through docs/reward-function.md first.** Length
   penalties are HPWL-normalized (detour factor) — keep them that way or
   curriculum transfer breaks. Shaping must stay potential-based *within a
   net*, and the shaping term is deliberately omitted on net-boundary
   transitions — never reintroduce a boundary payout (a Φ=0 refund at
   timeouts paid the agent for failing far from the target and taught
   wall-hugging; see the boundary note in docs/reward-function.md).
6. **γ lives only in `PPOConfig.gamma` (GAE).** The shaping term is
   *deliberately undiscounted* — `beta * (phi_after - phi_before)`, no γ on
   phi_after. Putting γ back pays a positive per-step annuity
   `beta*(1-γ)*d/HPWL` for standing still far from the target and taught
   far-wall loitering (see the annuity note in docs/reward-function.md).
7. **Checkpoints are a dict, not a bare state_dict.** `save_checkpoint`/
   `load_checkpoint` in `ppo.py` round-trip `{model, optimizer, stage,
   steps_done, completions, history}`. `load_checkpoint` back-compat-loads a
   bare state_dict (old format) but treats it as stage 0 / step 0 — don't
   assume an old checkpoint file carries stage/step info.
8. **`render.py` vs `viz.py`**: `render.py` is the headless CLI tool and only
   sets the `Agg` matplotlib backend when run as `__main__`, specifically so
   importing it from a notebook doesn't clobber the inline backend `viz.py`
   depends on. Don't move that `matplotlib.use("Agg")` call back to
   module-import time.

## Conventions

- Units: millimetres, float64 in the engine, float32 in observations.
  Layer 0 = top copper; vias span an inclusive `[layer_lo, layer_hi]`.
- Angle actions live in a **target-aligned canonical frame**. World bin i
  points at heading `2π·i/64`; `masker.py` rolls the angle arrays by
  `ActionMask.frame_offset` so canonical bin 0 points at the current target,
  `env.step` decodes world bin = `(bin + frame_offset) % 64`, and `env._obs`
  rotates the point cloud and head-state direction by the same *quantized*
  angle. Never mix canonical and world bin indices; `tests/test_env.py`
  asserts the alignment invariant every fuzz step.
- The env is deterministic given the board; all randomness lives in the
  generator and the policy (enables search-based inference later).
