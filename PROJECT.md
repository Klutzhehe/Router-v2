# Gridless PCB Routing AI — Full Project Document

An autonomous PCB routing engine that learns to route boards with
reinforcement learning in **continuous, gridless vector space** — no pixel
grids, no fixed routing lattice. Traces can leave a pad at any angle and any
length; legality is enforced by computational geometry, not discretization.

---

## 1. Design decisions (and why)

### Python-first engine (the Colab question)

Google Colab *can* compile C++ (it is a Linux VM with g++), but iterating on
native code inside a notebook is slow and painful. So the engine is **pure
Python + NumPy**, written so that every hot path is a vectorized array
operation (the same math as a C++ kernel, executed by NumPy's C internals).
The entire project runs on a stock Colab instance with **zero installs** —
`numpy` and `torch` are preinstalled.

The C++ design is preserved in `cpp/include/pcb/action_masker.hpp` as the
specification for a later 1:1 performance port (pybind11). The Python
`masker.py` implements exactly that interface.

### Legal-by-construction actions (the core trick)

A binary action mask over a *continuous* action space is mathematically
ill-posed — you cannot enumerate an uncountable set. The resolution:

- **Direction** is discretized into 64 angle bins → maskable Categorical.
- **Distance** stays continuous, but the policy emits a **fraction ∈ (0,1)**
  of the per-bin *maximum legal distance* that the geometry engine computes
  with a swept-disc cast.

Every sampled action is therefore legal **by construction**. The DRC penalty
in the reward exists only as a float-tolerance assertion; in testing it fires
exactly zero times (see §6).

### One net at a time, easy-first

The agent routes nets sequentially in ascending-HPWL order. Completed copper
becomes an obstacle for subsequent nets. (Net-ordering as a learned decision,
and rip-up/re-route, are roadmap items — §8.)

---

## 2. Architecture — the three pillars

```mermaid
flowchart LR
    subgraph ENV["Pillar 1 — Environment (headless CAD engine)"]
        GEN[generator.py<br/>curriculum boards] --> BRD[board.py<br/>pads / traces / vias<br/>continuous polygons]
        BRD --> MSK[masker.py<br/>swept-disc casts<br/>64-bin action mask]
        BRD --> PHY[PhysicsEvaluator hook<br/>2.5D EM solver API]
        MSK --> ENVV[env.py<br/>step / reset / reward]
        PHY --> ENVV
    end
    subgraph AGENT["Pillar 2 — Agent (dual-stream actor-critic)"]
        GNNS[DenseGNN<br/>netlist graph] --> FUSE[fusion trunk]
        PNT[BoardPointNet<br/>egocentric point cloud] --> FUSE
        HDS[head state] --> FUSE
        FUSE --> ACT["actor: type × angle × dist × layer"]
        FUSE --> CRT[critic: board-score baseline]
    end
    subgraph TRAIN["Pillar 3 — Training (PPO + curriculum)"]
        PPOO[ppo.py<br/>GAE + clipped surrogate] --> TRN[train.py<br/>auto-advancing curriculum]
    end
    ENVV -- "obs + mask" --> AGENT
    ACT -- "masked action" --> ENVV
    AGENT --> PPOO
```

### Pillar 1 — Environment (`python/pcb_router/`)

| File | Role |
|---|---|
| `geometry.py` | Vectorized kernel: ray-vs-disc, ray-vs-capsule first-contact casts, board-outline clipping. Validated against brute-force ray marching. |
| `board.py` | Continuous-space board state. All copper reduces to **discs** (pads, vias, keep-outs) and **capsules** (thick trace segments). Lists compile to cached flat NumPy arrays. |
| `masker.py` | Dynamic action masking. Per step: 64 max-legal-distance casts (one AABB broad-phase window + local narrow phase — the "one windowed R-tree query" design), via-fit checks per layer, commit legality. |
| `env.py` | Gym-style MDP. Reward implements `docs/reward-function.md` exactly. `PhysicsEvaluator` is the 2.5D EM-solver hook (returns zeros until a solver/surrogate plugs in). |
| `generator.py` | Programmatic curriculum boards, stage 0 → 5 (see §5). |

### Pillar 2 — Agent (`python/pcb_router/model.py`)

- **Logical stream** — `DenseGNN`: 3 rounds of message passing over the
  netlist (nodes = pads with position/layer/net-status/impedance/signal-type
  features; edges = required nets). Dense padded adjacency (≤64 pins) keeps
  batching trivial and needs no torch_geometric.
- **Physical stream** — `BoardPointNet`: shared MLP + symmetric max-pool over
  an **egocentric** point cloud (≤256 points) sampled from nearby copper,
  keep-outs, and the target pad, with layer offsets and same-net flags.
  Egocentric framing is deliberate: local congestion matters most, and it
  makes the policy translation-invariant.
- **Fusion** — `[graph_global | current_net_embed | board_global | head_state]`
  → MLP trunk → four actor heads + critic:
  - type: masked Categorical over {EXTEND, PLACE_VIA, COMMIT_NET}
  - angle: masked Categorical over 64 bins
  - distance: **Beta distribution** over the legal fraction (naturally bounded)
  - layer: masked Categorical over via targets
- Masking: `logits + (1 − mask)·(−1e9)` before sampling; joint log-prob is the
  sum of the four components (standard parameterized-action PPO).

### Pillar 3 — Training (`ppo.py`, `train.py`)

Standard PPO: GAE(λ=0.95), clip 0.2, entropy bonus, value-loss coefficient
0.5, Adam 3e-4, γ=0.995 (matches the reward spec). The trainer auto-advances
the curriculum stage when the rolling completion rate over 20 episodes clears
95%, and checkpoints every rollout.

---

## 3. Reward function

Full math with all constants: [`docs/reward-function.md`](docs/reward-function.md).
Summary: +C per completed net; per-step length penalty **normalized by the
net's HPWL lower bound** (so reward scale survives curriculum growth — the
agent is punished for its *detour factor*, not raw millimetres); via penalty;
potential-based shaping toward the target (policy-invariant); terminal
completion bonus/failure penalty plus impedance / skew / crosstalk penalties
from the physics hook.

---

## 4. Repository map

```
Router2/
├── PROJECT.md                    ← this document
├── CLAUDE.md                     ← working notes for AI pair-programmers
├── PCB_Router_Training.ipynb     ← Colab notebook: train with live viz + Drive checkpoints
├── requirements.txt              ← numpy, torch (both preinstalled on Colab)
├── docs/reward-function.md       ← reward math, single source of truth
├── cpp/include/pcb/action_masker.hpp   ← C++ port specification (future)
└── python/
    ├── pcb_router/
    │   ├── config.py             ← action-space contract + design rules
    │   ├── geometry.py           ← vectorized continuous-space kernel
    │   ├── board.py              ← board state (discs + capsules)
    │   ├── masker.py             ← dynamic action masking
    │   ├── env.py                ← RL environment + physics hook
    │   ├── footprints.py         ← component footprint library (passives, headers, ICs)
    │   ├── generator.py          ← curriculum board generator (places footprints, not random pads)
    │   ├── model.py              ← DualStreamRouter (pure torch)
    │   ├── ppo.py                ← PPO + GAE + rollout collection + checkpoint I/O
    │   ├── train.py              ← curriculum training entry point (CLI)
    │   ├── render.py             ← headless CLI board renderer (Agg backend)
    │   └── viz.py                ← inline/live training viz for the notebook
    └── tests/
        ├── test_geometry.py      ← analytic casts vs brute-force marching
        ├── test_env.py           ← legality fuzz + independent DRC check
        └── test_model_ppo.py     ← model/PPO smoke tests
```

---

## 5. Curriculum

Boards are assembled from a small component footprint library
(`footprints.py`: passives, headers, ICs) placed with realistic pin pitch
and courtyard keep-outs — not floating random pads. Net counts are
therefore **RNG-dependent** (component placement, decoupling-cap placement,
and diff-pair placement can each fail to find room and get skipped), so the
table below gives approximate counts, not exact ones. 4+ layer boards get a
real stack-up (`generator.stackup_roles`): dedicated power/ground layers
that are legal to route through but cost `RewardWeights.lam_stackup` if a
non-power net dwells there — this, not just adding more layers, is what
curbs "hop to an empty layer to dodge congestion."

| Stage | Layers | Stack-up | Board | ~Nets | Diff pairs |
|---|---|---|---|---|---|
| 0 | 2 | none | 20×20 mm | 3 | — |
| 1 | 2 | none | 25×25 mm | 6 | — |
| 2 | 2 | none (bottom-mount pads force vias) | 28×28 mm | 9 | — |
| 3 | 4 | 1 dedicated power/gnd layer | 32×32 mm | 16 | 1 |
| 4 | 6 | 2 dedicated power/gnd layers | 38×38 mm | 25 | 2 |
| 5 | 6 (cap) | 2 dedicated power/gnd layers | 45×45 mm | 27 | 2 |

The layer cap is 6 (was 12) — deliberately, so the curriculum stays focused
on same-layer congestion-solving rather than letting the agent dodge
obstacles by hopping to an ever-larger pool of empty layers. Curriculum
advancement itself is also stricter than a single lucky rollout: `train.py`
requires the rolling completion rate to clear `--advance-at` (default 0.99)
over a full `--advance-window` (default 50) episodes, *and* hold for
`--advance-streak` (default 3) consecutive PPO updates before promoting.

Scaling beyond (1000-pin BGA, multi-pin Steiner nets) requires raising
`N_MAX_PINS`/`P_MAX` in `config.py` and the roadmap items in §8.

---

## 6. Verification status (all run, all passing)

- **Geometry**: analytic first-contact distances match brute-force ray
  marching over 600 randomized cases (worst error < 1e-3 mm = the march
  resolution).
- **Legality fuzz**: 15 episodes × up to 1152 random *masked* actions across
  stages 0/1/3 → **0 DRC flags**, and an independent O(n²) re-check of the
  final copper confirms every gap ≥ 0.150 mm (the clearance rule).
- **Model/PPO**: forward, act/evaluate log-prob consistency, and a full PPO
  update produce finite losses.
- **Learning** (measured, 49k steps of PPO on stage 0, laptop CPU): rolling
  completion rate **4.8% → 35.0%**, mean episode return **−115 → −48**, policy
  entropy 4.92 → 4.25, with the curve still rising at cutoff. The random-walk
  baseline sits at ~0–5% completion. Rendered rollouts show the agent
  connecting nets (wastefully — the length/via penalties haven't disciplined
  it yet) with zero DRC violations throughout. Longer runs and the §8 roadmap
  items (parallel envs, imitation warm-start) are the path to saturating
  stage 0 and advancing the curriculum.

Run everything:

```bash
cd python
python tests/test_geometry.py
python tests/test_env.py
python tests/test_model_ppo.py
```

### Expected training progress, per stage, and what to do if it stalls

Extrapolated from the single measured run above (stage 0 only) — treat step
counts as rough budgets, not guarantees. The full diagnostic checklist (entropy
collapse, flat completion + flat pi_loss, exploding value loss, etc.) lives in
[`PCB_Router_Training.ipynb`](PCB_Router_Training.ipynb) since that's where
you'll actually be watching it happen.

| Stage | Board | Rough steps to ~90%+ completion |
|---|---|---|
| 0 | ~3 nets, 2 layers | 100k–400k |
| 1 | ~6 nets, 2 layers, keep-outs | 200k–600k |
| 2 | ~9 nets, bottom-mount pads (forces vias) | 300k–800k |
| 3 | ~16 nets, 4 layers, stack-up + 1st diff pair | 500k–1.5M |
| 4 | ~25 nets, 6 layers | 1M–3M |
| 5 | ~27 nets, 6 layers (cap) | 2M+, likely needs imitation warm-start (§8) |

See §5 for the full curriculum table and why the layer cap is 6, not 12.

The one hard invariant, independent of budget: **DRC count must always be 0**.
If it isn't, stop and treat it as a geometry-kernel bug, not a training issue.

---

## 7. How to run

### Locally

```bash
cd python
python -m pcb_router.train --stage 0 --total-steps 200000
# resume / continue on a later stage:
python -m pcb_router.train --stage 1 --resume checkpoints/router_stage0.pt
```

### On Google Colab

Use [`PCB_Router_Training.ipynb`](PCB_Router_Training.ipynb) at the repo root —
open it directly from GitHub in Colab (File → Open notebook → GitHub tab, or
`https://colab.research.google.com/github/Klutzhehe/Router-v2/blob/main/PCB_Router_Training.ipynb`).
It mounts Drive, clones/pulls the repo, trains with live per-epoch plots
(completion rate, return, entropy/value loss) and periodic inline board
snapshots, and checkpoints every epoch to Drive (model weights + optimizer
state + curriculum stage + step count, so re-running after a disconnect
resumes exactly, not just from bare weights). It also documents, per
curriculum stage, roughly how many steps to expect before ~90%+ completion
and what to do if a stage stalls short of that (see the notebook's "Expected
behavior per curriculum stage" section, reproduced in outline in §6 above).

Nothing to `pip install` — `numpy` and `torch` ship with Colab. Checkpoints
and the training history log land in `/content/drive/MyDrive/Router-v2-checkpoints/`,
not in the repo (see `.gitignore`) — Drive is the right home for that, not git.

For CLI-only use (no notebook), `pcb_router.train` supports the same resumable
checkpoint format:

```bash
python -m pcb_router.train --stage 0 --total-steps 500000 \
    --save-dir /content/drive/MyDrive/Router-v2-checkpoints --run-name router
# re-running with the same --save-dir/--run-name auto-resumes (weights +
# optimizer state + stage + step count), no separate --resume flag needed
# unless you want to load a checkpoint under a different name.
```

### Throughput expectations

The bare environment runs ~190 steps/s on a laptop CPU (profiled and
vectorized: the observation builder was the hotspot at 78% of step time, not
the geometry casts); with the policy in the loop, single-env training collects
~60–70 steps/s on CPU. On a GPU, single-env collection is *worse* than that
sounds: batch size 1 barely exercises the GPU (expect single-digit % GPU
utilization, a few hundred MB of VRAM) because Python/PCIe-transfer overhead
per step dominates the tiny forward pass. `VecRoutingEnv` +
`collect_rollout_vec` (`env.py`/`ppo.py`) fix this — batch `model.act` across
`N_ENVS` parallel boards each step (`--n-envs` on the CLI, `N_ENVS` in the
notebook; try 16–32 on a Colab GPU). GAE is computed correctly per-environment
(stored as `(steps_per_env, n_envs, ...)`, not naively flattened) before
handing a normally-shaped flat buffer to the unmodified `PPO.update`. The
remaining optimization ladder: shrink `P_MAX`, and ultimately the C++ port of
`masker.py`+`geometry.py` (the header is already written), which is where
"millions of frames per second" becomes realistic — and the mask, not single
collision checks, is the thing to optimize there too.

---

## 8. Roadmap

1. ~~**Vectorized envs**~~ — done: `VecRoutingEnv`/`collect_rollout_vec` batch
   `model.act` across N parallel boards (`--n-envs`/`N_ENVS`).
2. **Imitation warm-start** — A* teacher on the same DRC engine, behavior-clone
   its trajectories before PPO. Pure RL from scratch will not conquer large
   boards in hackathon time; the teacher doubles as the evaluation baseline
   the RL agent must beat.
3. **Rendering** — matplotlib/SVG board renderer for debugging and demos.
4. **Rip-up & re-route** — add a TEAR action or episode-level restarts so one
   badly placed net cannot permanently block another.
5. ~~**Differential pairs (length-mismatch half)**~~ — done: `generator.py`
   places real P/N pairs (`Net.pair_id`, matched pitch/width), `env.py`
   splices them to route back-to-back, and `PhysicsEvaluator.evaluate`
   measures real skew (routed-length mismatch) feeding `lam4`. **Still
   open**: paired routing heads (simultaneous P+N routing via a doubled
   action/observation space) — a materially larger architecture change than
   the length-mismatch proxy above.
6. **Physics solver** — implement the `PhysicsEvaluator` hook: start with a
   Hammerstad/Jensen microstrip impedance approximation (cheap, per-trace),
   graduate to a real 2.5D solver on subsampled episodes or a learned
   surrogate.
7. **C++ port** — pybind11 module implementing `action_masker.hpp`; drop-in
   replacement for `masker.py`.
8. **Search at inference** — the env is deterministic by design, so beam
   search / MCTS over the trained policy can squeeze out extra completion
   rate when routing a real board offline.
