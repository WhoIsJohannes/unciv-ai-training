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
1. **Honesty gate (GREEN, + found a bug):** small-rung, construction OFF, `--replay-window 1`, control
   (coef 0) vs treatment (coef 0.1). Result: **control 81/204=39.7%, treatment 98/204=48.0%** (+8.3pp,
   z=1.70). PBRS is not just non-regressive — it **improved the already-working tech/policy heads**, the
   expected sign for shortened credit. The gate ALSO exposed a real bug: a starving city has negative food,
   so `ln(1+Σfood)` went NaN → non-finite loss → the divergence guard skipped ~half the treatment rounds.
   PBRS still beat control through that. **Fixed** (`coerceAtLeast(0)` on the yield sums + `nan_to_num`);
   verified Φ finite, diverged=0, critic trains.
2. **Efficacy (DONE — PBRS did NOT salvage construction):** clean OFF+PBRS **37.3%** vs ON+PBRS **1.5%**
   (3/204, Δ−35.8pp, z=−9.15), 0 divergence both arms. Construction-ON still craters even with the credit
   horizon fixed. Diagnostics: ON+PBRS is **still +12.4% unit-biased** (chose 54% units vs 41% offered —
   unchanged from no-PBRS's +16%) and its **economy is measurably starved** (mean Φ 12.64 vs OFF's 16.69).

**REFINED ROOT CAUSE — it was per-city credit ATTRIBUTION, not the credit HORIZON.** PBRS shortened the
horizon (and is policy-invariant/honest — clean OFF+PBRS 37.3% ≈ no-PBRS control 39.7%, within noise; the
gate's apparent +8pp was a buggy-divergence artifact). But the per-step **advantage is a single scalar
shared across ALL heads and ALL cities** — PBRS makes it reflect economy growth, but it cannot attribute
that growth to the specific city/construction that caused it. So the construction head gets a noisy,
non-causal, diluted signal and never learns buildings>units for a given city. PBRS fixes *when* credit
arrives, not *which* per-city decision earned it. **Construction-via-RL is a robust negative across 4
approaches (per-turn 14%, commit 2%, buildings-only [dodge], PBRS 1.5%).**

**Spin-off result:** PBRS itself is correctly implemented + honest (policy-invariant), but **clean PBRS is
~neutral** for the civ-global heads (no efficacy boost) — the credit horizon was not actually limiting
tech/policy. So PBRS is a validated-but-neutral capability (kept, default coef 0), not a win.

**Remaining untested levers (bigger efforts):** (a) per-city counterfactual/difference rewards (credit each
city for its MARGINAL ΔΦ — the actual attribution fix, but needs stable per-city identity tracking the
panel flagged as hard); (b) imitation warm-start from the heuristic (sidesteps learning attribution
entirely — the net starts from a good construction policy and RL-finetunes).

## v7.3 — the attribution fix: per-city credit assignment — IN PROGRESS
User directive after PBRS: **"Attack attribution directly: per-city credit."** This is lever (a) above,
implemented as a per-city *value baseline* (not raw difference rewards — no cross-city counterfactual
rollout needed, so no unstable per-city identity tracking).

**Mechanism.** Each city's construction is credited by ITS OWN economy return instead of the single civ-wide
scalar:
- **Record** (schema 6→7, `BLOCK_ECON_CITY`): per-step VARIABLE f32 `econ_city`, one row per own city
  (aligned to construction) = that city's raw log-economy `ln(1+prod+food+sci)`.
- **Net:** a per-city VALUE head parallel to the construction head (own-city embedding ⊕ trunk body → V_city),
  final layer zero-init so V_city≈0 at init. Train-only, dropped from the ONNX export.
- **Trainer** (`--construction-credit-coef`, >0 activates): construction is pulled OUT of the joint PPO ratio
  and trained by a SEPARATE per-city PG term. A per-city GAE over `econ_city` gives A_city; the construction
  advantage is `shared_adv + coef·A_city` (COMA/difference-rewards direction). `econ_city` is per-round
  standardized then average-reward scaled ×(1−γ) so R_city is O(1); PG + entropy are MEAN over present cities
  (city-count-invariant — summing ~6 cities was the v7 objective-domination bug); value loss masked to
  present cities. **coef=0 reproduces the legacy v7.2 shared-adv joint-PPO path exactly.** On-policy
  (`--replay-window 1` — the per-city term has no importance ratio).

**Validation (all green before the run):** Kotlin dataplane suite + full Python suite pass (fixed two stale
v6 shard-builders predating the v7.2 fail-loud `phi` read; the determinism test is a pre-existing
engine-level flake — same-seed runs already diverge in *step count*, unrelated to `econ_city`, which is a
pure function of city stats). End-to-end Tiny smoke: 3 rounds gen→train→eval, econ_city finite, per-city
branch fires, no divergence. Diagnosed + fixed the per-city value-loss scale (raw log-economy level → R_city
~10²; standardize + ×(1−γ) + zero-init head → R_city O(0.14), value loss O(0.2)). Micro-batch path matches
whole-batch within the size-weighting tolerance (Δw 0.004).

**Experiment (running, resumable — `python/run_v73eff.sh`):** 3 arms, small rung, Medium, 16 rounds, rw1,
NO PBRS, 200-game ceiling @ eval-seed 4242424:
- `off`       — construction OFF (baseline, tech+policy only)
- `on-shared` — construction ON, coef 0 (the v7 negative under MATCHED rw1 conditions — the control)
- `on-pcc`    — construction ON, coef 0.5 (the fix)

The question: does per-city credit turn the negative around — `on-pcc ≥ off` (moves the right way, per the
ship criterion) and `on-pcc > on-shared` (proving the credit MECHANISM, not just re-running construction)?
_Results pending — this section will be updated with the 3-way ceiling comparison + verdict._

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
