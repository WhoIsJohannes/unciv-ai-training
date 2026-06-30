# RESULTS — selfplay-v7-construction (first per-entity head: city PRODUCTION)

## The question
v5's policy (tech + policy only) sits BELOW the symmetric 50% break-even. v7 gives the net its first
**per-entity** lever — per-city production. **Does controlling production move the learner toward/past 50%
on Medium?**

## Answer: NO — construction control (as implemented) HURTS. It does not approach 50%.
4-arm Medium run (16 rounds, K=4, continual; 200-game ceiling @ seed 4242424). Construction is the only
axis within each rung pair. **Construction-ON does not beat OFF in either rung; neither crosses 50%.**

| rung | OFF ceiling | ON ceiling | Δ (ON−OFF) | z | p (one-sided ON>OFF) | construction helps? | ON crosses 50%? |
|---|---|---|---|---|---|---|---|
| **small** | 96/204 = **47.1%** | 28/204 = **13.7%** | **−33.3 pp** | −7.32 | ≈1.0 | **NO** (catastrophic) | no (z=−10.4) |
| **medium** | 82/204 = **40.2%** | 75/204 = **36.8%** | **−3.4 pp** | −0.71 | 0.76 | **NO** (n.s.) | no (z=−3.8) |

**Ship recommendation (analyze_v7):** SHIP INFRA ONLY — no directional win; default `--control-construction OFF`.

## Interpretation
- **The machinery is correct; the control POLICY is what fails.** All 5 pre-run ACs are green (legality,
  parity, no-op bit-identical, throughput 1.957×, schema). The ON arms genuinely controlled construction —
  a silent fallback would make ON≈OFF (~47%), not crater small-ON to 13.7%. So this is a real behavioral
  finding, not a measurement artifact.
- **Likely cause — per-turn override → CHURNING.** The policy re-picks every city's production every turn.
  An under-trained net (sparse terminal-only ±1 reward over a large per-city action space) emits a noisy
  per-turn target, so cities **switch construction every turn instead of committing-until-done** like the
  heuristic. Production is stored per-item, so nothing is *lost* — but nothing gets *finished* efficiently,
  so the civ falls behind. The heuristic's commit-until-done is a strong, hard-to-beat baseline.
- **Capacity matters.** The small rung (GNN-only, embed_dim 8) churns **catastrophically** (47% → 14%,
  worse than the 28.9% blind baseline). The medium rung (attention, embed_dim 16) churns far less and
  recovers most of the heuristic's value (40% → 37%, not significant) — but still does not beat it.
- **Net:** per-turn, terminal-reward-trained per-city construction is destructive at low capacity and
  merely non-beneficial at higher capacity. It is NOT the lever that crosses 50% — at least not in this
  design.

## v7.1 investigation (per "understand why before the 10h re-run") — ROOT CAUSE FOUND
The churning hypothesis was tested and **disproven**. Two findings:
1. **Cadence is not the cause.** v7.1 changed per-turn override → commit-until-done (decide only at
   construction completion; heuristic disabled for controlled civs). This cut decisions/step 5.8 → 0.6
   and **fixed the training instability** (entropy 0.39→1.13 healthy like OFF; no grad-norm 957 spike).
   But small-ON got **WORSE: 2.0% (4/204)** — commit-until-done amplifies bad choices (the city is stuck
   building each pick). So churning/objective-domination was a symptom, not the root cause.
2. **Root cause = UNIT-BIAS from weak credit assignment.** The net over-builds UNITS (military) and
   starves BUILDINGS (economy), beyond mask availability:
   - small-ON (2%): chose 58% units vs 42% offered → **+16pp unit-bias**.
   - medium-ON (37%): chose 43% units vs 35% offered → **+8pp unit-bias** (more capacity → less bias → better).
   Economy buildings pay off over a long horizon — too hard to credit-assign from a terminal-only ±1
   reward — so the net defaults to "build a unit now," stalling its economy. The hand-tuned heuristic
   knows to build economy (OFF 47%). The unit-bias magnitude tracks the harm (small +16%→2%, medium +8%→37%).

**Conclusion: construction control via from-scratch RL with terminal-only reward learns a unit-biased,
economy-starving policy that the heuristic beats. Not a cadence bug, not a mechanical bug — a sparse-reward
credit-assignment limit over a large per-city action space.** A real fix needs imitation warm-start (init
the head from the heuristic) OR a denser/shaped reward (the latter is FROZEN out of v7 scope). Neither is
a quick tweak; further full runs would only re-confirm the negative.

## v7.2 — the real fix: potential-based reward shaping (PBRS) — IN PROGRESS
The user reframed sharply: the problem is that **the reward is too far in the future** (long-horizon
credit), and asked for the principled fix. A design panel (7 RL approaches → adversarial vetting →
synthesis) confirmed: among reward-shaping / critic-decomposition / return-redistribution / horizon
options, **PBRS is the only one that both (a) actually shortens the credit horizon AND (b) provably keeps
the objective honest.** The critic/decomposition alternatives were vetted down to "bias-neutral baseline
shifts that reach ~47% neutral at best" or "myopic economy proxies that flip the failure to UNDER-building
military."

**Mechanism (committed):** record a per-step log-stabilized economy potential
`Φ = ln(1+Σprod)+ln(1+Σfood)+ln(1+Σscience)+ln(1+#techs)` (SampleSchema VERSION 5→6, `BLOCK_PHI`), and add
the shaping reward `F = γ·Φ(s')−Φ(s)` to each step in `_optimize_actor_critic` (`--reward-shaping-coef`,
0 ⇒ terminal-only). **Ng-Harada policy-invariance:** the `F` terms telescope to a constant
(`coef·(γ^(L-1)·Φ_{L-1} − Φ_0)`), so the optimal "win the game" policy is UNCHANGED — only the credit
timing shifts (~250 → ~5 steps), so a Granary's economic payoff registers immediately instead of 200 turns
later. `ln` keeps Φ bounded so F can't drift-dominate the terminal ±1. Tests: telescopes-to-constant
(policy-invariance), coef=0/phi=None no-op, schema-6 round-trip; Φ recorded (range 1.1–17, grows w/ economy).

**Honesty note:** this RELAXES the frozen "no shaping" rule, but only on the principled policy-invariant
side the spec allows (user-approved). It does NOT change what's optimal; it only speeds credit.

**Staged validation:**
1. **Honesty gate (RUNNING):** small-rung, construction OFF, `--replay-window 1` (on-policy, clean PBRS),
   control (coef 0) vs treatment (coef 0.1). Gate: treatment must be **non-regressive** (PBRS must not bias
   the already-working tech/policy heads off-baseline) AND show **faster value-loss drop** (proves it
   injects usable signal). A regression ⇒ the potential is mis-specified → re-tune before the construction run.
2. **Efficacy (pending green gate):** construction-ON + PBRS small-arm vs OFF (47%) — does shortened credit
   let the net learn good construction and BEAT the heuristic?

## What this does NOT rule out (explicit follow-ups — require a bigger effort, not run here)
1. **Decision cadence** — decide at construction-completion (the natural commit point) rather than every
   turn, to remove churning. (The plan's perpetual-only gate was inert because the heuristic keeps cities
   busy; a completion-triggered gate is the untested middle ground.)
2. **Reward** — terminal-only ±1 gives near-zero per-decision credit for a high-frequency per-city lever;
   a denser signal (or a construction-progress potential) may be needed. (NOTE: out of v7 scope — v7 froze
   "terminal-only, no shaping".)
3. **Capacity** — the medium rung's near-parity suggests larger nets may learn coherent construction.

## Machinery validated (all pre-run ACs green — the deliverable that ships)
| AC | Result |
|---|---|
| #1 Legality | ✅ 0 illegal / 1106 decisions; recorded==applied via mask |
| #2 Parity | ✅ ORT-vs-torch construction logits atol 1e-4 + constructionId mask-space round-trip |
| #4 Schema | ✅ VERSION 4→5 lockstep; new VARIABLE blocks round-trip free; v4 refused |
| #5 No-op | ✅ construction-OFF bit-identical to v6 (zero-summand oracle), incl. K=4 replay |
| #6 Throughput | ✅ bench-onnx ON ratio 1.957× ≥ 0.70 |
| #3 Effect | ❌ construction-ON does NOT beat OFF (small −33pp p≈1.0; medium −3.4pp p=0.76); neither crosses 50% |

## Ship disposition (per user decision D-C5)
No directional win → **ship the validated per-entity infrastructure, default `--control-construction OFF`**
(already the SampleConfig/run_loop default — no change needed). The infra (per-entity decision loop,
variable recorded blocks, schema v5, the seam, the per-city net head + ONNX path, the training summand) is
correct and is **reused by the next per-entity heads** (promotion / great-person / diplomatic-vote), which
were the cheap follow-ups this feature was built to enable. Construction itself is a clean negative; its
follow-up is the cadence/reward/capacity work above, not this feature.

## Run provenance
4 arms, structured/Medium, 16 rounds, gen 16 / eval 80, turn-cap 250, K=4, micro-batch 256, continual;
ceilings 200 games @ eval-seed 4242424 (ON arms evaluated WITH construction control). Wall clock ~10h
(`training-runs/v7/acceptance_v7_compare.json`). seed/config held constant across the OFF/ON pairs.
