# Reward Function Specification

Multi-objective reward for the gridless PCB routing agent. All symbols and
default constants here are the single source of truth — the C++ evaluators and
the Python training loop must both read from this spec.

## Notation

| Symbol | Meaning |
|---|---|
| $s_t, a_t$ | State and action at step $t$ |
| $\Delta\ell_t$ | Copper length (mm) added by action $a_t$ |
| $\mathrm{HPWL}_n$ | Half-perimeter wirelength lower bound of net $n$ (scale normalizer) |
| $N, N_c$ | Total nets / nets completed at episode end |
| $Z_i, Z_i^*$ | Achieved / required impedance of high-speed net $i$ (from 2.5D solver) |
| $\Delta t_p$ | Intra-pair skew of differential pair $p$; $\tau_{\max}$ = allowed skew |
| $\mathrm{XT}_{ij}$ | Crosstalk coupling metric between nets $i, j$ (solver hook) |
| $d_{\mathrm{geo}}(s)$ | Euclidean distance from routing head to current target pad |
| $\gamma$ | Discount factor |

## Per-step reward

$$
r_t \;=\;
\underbrace{C \cdot \mathbb{1}\!\left[\text{net } n \text{ completed at } t\right]}_{\text{net completion}}
\;-\; \underbrace{\lambda_1 \frac{\Delta\ell_t}{\mathrm{HPWL}_n}}_{\text{normalized length}}
\;-\; \underbrace{\lambda_2 \cdot \mathbb{1}\!\left[a_t = \mathrm{PLACE\_VIA}\right]}_{\text{via usage}}
\;-\; \underbrace{D \cdot \mathbb{1}\!\left[\text{DRC violation}\right]}_{\text{safety net}}
\;+\; \underbrace{\beta\left(\gamma\,\Phi(s_{t+1}) - \Phi(s_t)\right)}_{\text{potential shaping}}
$$

with the potential function

$$
\Phi(s) = -\,\frac{d_{\mathrm{geo}}(s)}{\mathrm{HPWL}_n}
$$

Notes:

- **Length is normalized per net** by $\mathrm{HPWL}_n$ so that a 5-component
  curriculum board and a 1000-pin BGA produce rewards on the same scale.
  The agent is effectively penalized on its *detour factor*, not raw mm.
- **Shaping is potential-based** (Ng, Harada & Russell, 1999), so it changes
  the learning dynamics but not the optimal policy.
- **The DRC term should never fire** — the action mask makes illegal moves
  unsamplable. It exists as a safety net for floating-point edge cases in the
  geometry kernel; if it fires more than ~0 times per million steps, that is a
  C++ bug, not a training signal.

## Terminal reward (step $T$)

$$
R_T \;=\;
B\,\frac{N_c}{N}
\;-\; F \cdot \mathbb{1}\!\left[N_c < N\right]
\;-\; \lambda_3 \!\!\sum_{i \in \mathcal{H}} \frac{\left|Z_i - Z_i^*\right|}{Z_i^*}
\;-\; \lambda_4 \!\!\sum_{p \in \mathcal{P}} \max\!\left(0,\; \left|\Delta t_p\right| - \tau_{\max}\right)
\;-\; \lambda_5 \!\!\sum_{(i,j)} \mathrm{XT}_{ij}
$$

where $\mathcal{H}$ is the set of impedance-controlled nets and $\mathcal{P}$
the set of differential pairs. The $\lambda_3, \lambda_4, \lambda_5$ terms are
produced by the 2.5D field-solver hook (`PhysicsEvaluator` API in the C++
engine); when the solver is disabled they are zero.

## Total return

$$
G = \sum_{t=0}^{T-1} \gamma^t r_t \;+\; \gamma^T R_T
$$

## Default constants (curriculum stage 1)

| Constant | Value | Rationale |
|---|---|---|
| $C$ | $+10$ | Dominant positive signal per net |
| $B$ | $+50$ | Full-board completion bonus (scaled by ratio) |
| $F$ | $-20$ | Failure penalty when any net unrouted |
| $D$ | $-50$ | Should never fire (see above) |
| $\lambda_1$ | $1.0$ | Detour factor of 2 costs 1/10 of a net completion |
| $\lambda_2$ | $0.5$ | Via ≈ 0.5 detour-units |
| $\lambda_3$ | $5.0$ | Per unit of relative impedance error |
| $\lambda_4$ | $2.0$ | Per ps of excess skew (tune per stack-up) |
| $\lambda_5$ | $1.0$ | Solver-metric dependent |
| $\beta$ | $1.0$ | Shaping weight |
| $\gamma$ | $0.995$ | Long horizons on large boards |

Curriculum stages may anneal $\lambda_{3..5}$ from 0 upward: early training
optimizes connectivity + length only; physics penalties switch on once the
agent reliably completes boards.
