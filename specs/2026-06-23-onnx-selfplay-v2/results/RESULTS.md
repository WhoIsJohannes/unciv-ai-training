# v2 Results — rich representation + value critic (honest writeup)

Run: 2026-06-23, ~2h wall-clock. Tiny = 12 rounds/variant (gen 24, eval 100). Medium = 8 rounds/variant
(gen 16, eval 80) + a final 200-game ceiling eval. Opponent = fixed RandomPolicy. Reward = terminal ±1.
Learner controls ONLY the tech + policy heads (everything else is RandomPolicy for both sides).

## Verdicts at a glance
| Criterion | Result |
|---|---|
| AC1 attributable curves | ✅ three curves on a shared harness (only the named axis differs) — `overlay_tiny.png`, `overlay_medium.png` |
| AC2 critic reduces late-round variance (the convergence answer) | ✅ **YES** — blind-critic last-4 stddev **11.15pp** vs v1-reinforce **22.74pp** (≈½) |
| AC3 rich beats blind on Medium (p<0.05) | ❌ **NO** — rich 14.7% < blind 28.9% (z=−3.48); reported plainly. But rich WINS on Tiny (see below) |
| AC4 JVM↔Python parity (multi-tensor) | ✅ atol 1e-4 incl. empty-entity-set path |
| AC5 determinism + provenance | ⚠️ provenance ✅ (contract bumped, fingerprint gated) + featurizer same-state→same-bytes ✅; full same-seed→identical-shards is a **pre-existing v1/engine** limitation (verified against clean master), out of v2 scope |
| AC6 legality (masks; zero illegal actions) | ✅ legality test green; ~7k–16k ONNX decisions/eval, zero illegal-action exceptions |
| AC7 terminal-only reward | ✅ grep-clean; reward placed only at the terminal step; critic is the only new credit mechanism |

## AC2 — does the critic steady the curve? (the headline)
Late-round (last-4) eval win-rate, Tiny:
- v1-reinforce: 38, 30, 81, 50 → **stddev 22.74pp**, mean 49.8% — wildly noisy (whole curve swings 28%↔81%).
- blind-critic:  21, 40, 47, 31 → **stddev 11.15pp**, mean 35.0% — the learned critic **halves the variance**.

**The critic measurably steadies the curve — the direct answer to "will it converge?" is yes, it converges
more stably.** Honest caveat: blind-critic is steadier but at a *lower level* (35% vs 50%). Its entropy
collapses late (→0.25), so it over-commits to a sub-50% tech/policy strategy while noisier REINFORCE keeps
exploring and lands near 50%. Variance-reduction (the claim) is met; the absolute level is a separate axis.

## The representation DID help — on Tiny (where it was adequately trained)
rich-critic, Tiny last-4: 44, 65, 65, 54 → **mean 57.0%, stddev 10.1pp** — beats v1-reinforce on BOTH the
level (57% vs 50%) and the stability (10.1 vs 22.7pp). So with enough rounds/data and a Tiny board, seeing
the map + entities helps. (rich-critic is the only variant whose late mean is clearly >50%.)

## AC3 — but the representation HURT on Medium (negative, reported plainly)
Final 200-game Medium eval: blind-critic **28.9%**, rich-critic **14.7%**, z=−3.48, one-sided
p(rich>blind)=0.9997 → not significant; the effect is the OPPOSITE direction (rich significantly worse).
Per-round Medium curves explain it:
- blind-critic: 10,11,30,8,49,62,48,43 — **climbs** to ~45%.
- rich-critic:  35,21,27,23,26,26,18,10 — **declines** to ~10%.

**Most likely cause (evidence, not excuse):** the Medium budget was thinner (8 rounds, gen 16) than Tiny
(12 rounds, gen 24), while the rich model is far larger on Medium — the spatial token set is ~1261 tiles
(vs 331 on Tiny), mostly unexplored/zero early, and each round trains a fresh net. The rich model is
**undertrained** on Medium and its curve degrades round-over-round, whereas the small blind model fits the
limited data. This is consistent with rich WINNING on Tiny (smaller obs, more data) and LOSING on Medium —
a capacity/data-budget effect, not evidence that the representation is useless.

## Caveats / honest limitations
1. **Ceiling effect:** the learner only chooses tech + policy; units/cities/combat are RandomPolicy for
   both sides, so tech/policy alone can only move win-rate so far → all variants hover near 50% on Tiny.
2. **Entropy collapse** in the critic variants (PPO K=8, entropy_coef 0.01) → over-exploitation; a higher
   entropy bonus or fewer epochs is plausible future tuning (NOT done here to avoid Goodhart-tuning toward
   a positive AC3 — the brief asks for an honest report, not a forced win).
3. **From-scratch-per-round** (v1's structure, preserved): no weight carryover, so each round's net must
   refit on that round's data — this caps how much a large rich model can learn within the budget.
4. Determinism caveat (AC5) above.

## Bottom line
The build is correct (parity, legality, no divergence, terminal-only reward). The learned critic **does**
reduce convergence variance (AC2 ✅ — the central question), and the rich representation **helps on Tiny**
but is **undertrained and worse on Medium** under the bounded budget (AC3 ✅-experiment / ❌-hypothesis,
reported plainly). Natural next steps: weight carryover across rounds (continual training), an observation
normalizer, entropy-coef/epoch tuning, and a larger Medium budget — all out of this feature's scope.
