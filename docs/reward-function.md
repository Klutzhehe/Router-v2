# Reward Function Specification

Multi-objective reward for the gridless PCB routing agent. All symbols and
default constants here are the single source of truth — the C++ evaluators and
the Python training loop must both read from this spec.

## Notation

| Symbol | Meaning |
|---|---|
| $s_t, a_t$ | State and action at step $t$ |
| $\Delta\ell_t$ | Copper length (mm) added by action $a_t$ |
| $\Delta\theta_t$ | Turn angle (radians) from previous segment heading to current segment heading |
| $\mathrm{HPWL}_n$ | Half-perimeter wirelength lower bound of net $n$ (scale normalizer) |
| $N, N_c$ | Total nets / nets completed at episode end |
| $Z_i, Z_i^*$ | Achieved / required impedance of high-speed net $i$ (from 2.5D solver) |
| $\Delta\ell_p$ | Intra-pair routed-length mismatch of differential pair $p$ (mm) — a proxy for timing skew; see note below |
| $\mathrm{XT}_{ij}$ | Crosstalk coupling metric between nets $i, j$ (solver hook) |
| $d_{\mathrm{geo}}(s)$ | Euclidean distance from routing head to current target pad |
| $\gamma$ | Discount factor |

## Per-step reward

$$
r_t \;=\;
\underbrace{C \cdot \mathbb{1}\!\left[\text{net } n \text{ completed at } t\right]}_{\text{net completion}}
\;-\; \underbrace{\lambda_1 \frac{\Delta\ell_t}{\mathrm{HPWL}_n}}_{\text{normalized length}}
\;-\; \underbrace{\lambda_2 \cdot \mathbb{1}\!\left[a_t = \mathrm{PLACE\_VIA}\right]}_{\text{via usage}}
\;-\; \underbrace{\lambda_{\text{turn}} \left(\frac{\Delta\theta_t}{\pi}\right)^{\!2}}_{\text{turn penalty}}
\;-\; \underbrace{D \cdot \mathbb{1}\!\left[\text{DRC violation}\right]}_{\text{safety net}}
\;+\; \underbrace{\beta\left(\Phi(s_{t+1}) - \Phi(s_t)\right)}_{\text{potential shaping (undiscounted, see note)}}
$$

with the potential function

$$
\Phi(s) = -\,\frac{d_{\mathrm{geo}}(s)}{\mathrm{HPWL}_n}
$$

Notes:

- **Length is normalized per net** by $\mathrm{HPWL}_n$ so that a 5-component
  curriculum board and a 1000-pin BGA produce rewards on the same scale.
  The agent is effectively penalized on its *detour factor*, not raw mm.
- **The shaping difference is deliberately undiscounted** — $\beta(\Phi(s_{t+1}) - \Phi(s_t))$,
  *not* the textbook $\beta(\gamma\Phi(s_{t+1}) - \Phi(s_t))$. With the
  $\gamma$ inside, a stationary agent collects $\beta(1-\gamma)\,|\Phi|$ per
  step — since $\Phi \le 0$, that is a **positive annuity proportional to
  distance from the target** (measured $+0.0167$/step at $d/\mathrm{HPWL}=1.1$
  with $\beta=3$; up to $\sim+0.10$/step in a far corner on a short net —
  a full net-completion of free reward over one 96-step budget). The trained
  policy converged to exactly that harvest: climb to the far board edge,
  loiter along it, then descend to the target late in the budget. The
  $\gamma$ factor exists only to make full PBRS exactly policy-invariant
  under discounting, a guarantee the boundary rule below already forgoes —
  dropping it makes "stand still" worth exactly $0$ and removes the annuity
  entirely.
- **Shaping is potential-based *within a net*, and the shaping term is
  omitted entirely on net-boundary transitions** (COMMIT, budget timeout, or
  the engine skipping an unroutable net). $\Phi(s) = -d_{\mathrm{geo}}(s)/\mathrm{HPWL}_n$
  is defined relative to whichever net is "current," so each net's routing
  has its own potential function and the boundary transition has no
  well-defined PBRS term. Both "obvious" boundary conventions were tried and
  measured to be actively harmful:
  1. *Evaluating $\Phi(s_{t+1})$ on the next net* leaks that net's unrelated
     geometry into the reward (a completing COMMIT paid ~8.0 instead of
     ~10.3 of its own $C$).
  2. *The textbook convention $\Phi(\text{boundary}) = 0$* refunds
     $+\beta\,d_{\mathrm{end}}/\mathrm{HPWL}_n$ at every budget timeout — a
     concentrated reward for being **far** from the target when the net is
     abandoned (measured +3.3 for one wall-hugging trajectory; up to ~+28 on
     stage 0 for short nets — more than completing pays). This is
     policy-invariant in exact theory (the wander that inflated
     $d_{\mathrm{end}}$ was charged incrementally on the way out), but with
     GAE($\lambda$) truncating credit and an imperfect critic, the spike
     lands on the wander/wall-jiggle actions immediately before it and
     *teaches* wall-hugging.
  Omitting the boundary term keeps the incremental progress payments and
  simply never refunds the residual. The un-refunded residual is equivalent
  to a terminal penalty of $\beta\,d_{\mathrm{end}}/\mathrm{HPWL}_n$ per
  unfinished net: failure graded by final distance to target — a gradient
  the flat $F$ penalty cannot provide. Near-target COMMITs have ~zero
  residual, so completions still pay their full $C$. This is a deliberate,
  bounded deviation from strict PBRS; any future change to shaping must not
  reintroduce a positive boundary payout.
- **β must exceed λ₁.** With β = λ₁ the shaping bonus for moving toward the
  target *exactly cancels* the length penalty: progress earns zero immediate
  reward, every other movement is negative, and "take minimum-length steps
  until the budget runs out" becomes a strong local optimum. This was observed
  empirically (stage 0, 174k steps): the policy walked to the board edge,
  aimed *away* from the target (mean cos −0.36 vs the target direction), and
  collapsed its step length to ~0.15 mm while completion decayed to ~5%.
  With β > λ₁ each mm of progress toward the target is net-positive
  ((β−λ₁)·δ/HPWL), standing still earns 0, and the local optimum dissolves —
  while the shaping term stays potential-based, hence still policy-invariant.
- **The DRC term should never fire** — the action mask makes illegal moves
  unsamplable. It exists as a safety net for floating-point edge cases in the
  geometry kernel; if it fires more than ~0 times per million steps, that is a
  C++ bug, not a training signal.
- **Turn penalty is quadratic in the normalized angle, not linear.** A linear
  penalty prices a 10° wobble and a 170° hairpin at the same per-degree
  rate — the marginal cost of the first 10° of a turn equals the marginal
  cost of the last 10° of a 90° turn. That barely disciplines jagged zigzag
  or sharp-cornered detour loops (observed directly: a stage-0 render after
  500k+ steps still routing one net through an unnecessary multi-mm loop).
  Squaring the normalized angle keeps a full 180° reversal priced the same
  as before ($1^2=1$) while a 90° corner drops to 1/4 of a reversal's cost
  (not 1/2) and small course corrections (already a small fraction, squared
  smaller still) stay close to free — sharp turns are singled out instead
  of taxing every turn at a uniform rate.
- **Stack-up penalty ($\lambda_{\text{turn}}$'s neighbor, $\lambda_{\text{stackup}}$,
  not shown above — see the constants table)**: boards with 4+ layers assign
  each layer a role, `LAYER_ROLE_SIGNAL` or `LAYER_ROLE_POWER`
  (`generator.stackup_roles`), mirroring a real stack-up's dedicated
  ground/power planes. Routing is never blocked by this — a non-power net
  (`signal_type != 1`) is legal on a `POWER`-role layer, but pays
  $\lambda_{\text{stackup}}\cdot\Delta\ell_t/\mathrm{HPWL}_n$ per step it
  dwells there (same normalized-length shape as $\lambda_1$). Power/ground
  nets pay nothing on power layers, and nobody pays anything on
  `SIGNAL`-role layers (general-purpose, matches real boards where short
  power stubs on a signal layer are normal). Via barrels merely *crossing* a
  power layer are not penalized, only copper actually routed there.

## Terminal reward (step $T$)

$$
R_T \;=\;
B\,\frac{N_c}{N}
\;-\; F \cdot \mathbb{1}\!\left[N_c < N\right]
\;-\; \lambda_3 \!\!\sum_{i \in \mathcal{H}} \frac{\left|Z_i - Z_i^*\right|}{Z_i^*}
\;-\; \lambda_4 \cdot \mathrm{mean}_{p \in \mathcal{P}} \left|\Delta\ell_p\right|
\;-\; \lambda_5 \!\!\sum_{(i,j)} \mathrm{XT}_{ij}
$$

where $\mathcal{H}$ is the set of impedance-controlled nets and $\mathcal{P}$
the set of completed differential pairs (`Net.pair_id` groups). $\lambda_3$
(impedance) and $\lambda_5$ (crosstalk) are still produced by the stubbed
2.5D field-solver hook (`PhysicsEvaluator.evaluate`) and are zero until a
real solver plugs in. **$\lambda_4$ (skew) is real, not stubbed**: for each
pair where both nets completed, $\Delta\ell_p$ is the absolute difference
between the two nets' total routed trace length (summed straight from
`board.traces`), averaged across all completed pairs — the "length mismatch
is nearly free to compute" simplification from `PROJECT.md`'s roadmap. This
is a routed-length proxy for timing skew, not an actual propagation-delay
calculation (no trace-velocity model exists yet); a pair with one net
unfinished contributes no skew term (nothing to fairly compare against).

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
| $\lambda_{\text{turn}}$ | $0.3$ | Quadratic in angle: a 180° reversal costs 0.3, a 90° corner costs 0.075 (1/4, not 1/2), small corrections cost next to nothing |
| $\lambda_{\text{stackup}}$ | $0.5$ | Non-power net dwelling on a `POWER`-role layer (4+ layer boards only) |
| $\lambda_3$ | $5.0$ | Per unit of relative impedance error (stubbed at 0 pending a solver) |
| $\lambda_4$ | $2.0$ | Per mm of intra-pair routed-length mismatch (real, see terminal-reward note) |
| $\lambda_5$ | $1.0$ | Solver-metric dependent |
| $\beta$ | $1.5$ | Shaping weight — **must exceed $\lambda_1$** (see note above); was $3.0$, lowered to tighten steering cone and reduce wander |
| $\gamma$ | $0.995$ | Long horizons on large boards. **GAE/PPO discount only** (`PPOConfig.gamma`) — the shaping term is undiscounted by design (see note above) |

Curriculum stages may anneal $\lambda_{3..5}$ from 0 upward: early training
optimizes connectivity + length only; physics penalties switch on once the
agent reliably completes boards.
