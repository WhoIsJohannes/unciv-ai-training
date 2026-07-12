---
name: selfplay-v7-construction-negative
description: RESOLVED TO A WIN — construction control was a robust negative (collapse) UNTIL v7.4 BC-clone + KL-leash: bckl 52.3% vs off 37.5%, +14.8pp, t=2.51 SIGNIFICANT (n=8), crosses the 50% break-even. Recipe = clone heuristic then leashed-finetune.
metadata: 
  node_type: memory
  type: project
  originSessionId: f6780e61-0544-41d1-ba19-6b764050f0f5
---

**★ RESOLVED TO A WIN (v7.4).** Controlling per-city production — the original v7 question ("does it move the
learner toward/past 50%?") — is now a STATISTICALLY SIGNIFICANT POSITIVE. Winning recipe: behavior-clone the
heuristic's picks (tight: `--bc-epochs 120`, acc 0.67) then RL-finetune with a KL-to-clone leash
(`--construction-kl-coef 0.5`) + entropy 0.02. Replicated **off vs bckl × 8 seeds** (small/Medium/16-round/
rw1/mb0): **off 37.5%±4.4 vs bckl 52.3%±4.8; paired Δ=+14.8pp±5.9, t=2.51, p≈0.02, 7/8 seeds positive** —
construction control significantly beats NOT controlling it, and crosses the 50% break-even. `run_v74kl.sh`.

**Why it took 4 negatives first + the arc:** raw construction-on collapses to ~0% (mode collapse — sparse
terminal reward over a 251-wide action space + losing ~every game → all-negative advantages → net builds 99%
military units, cities stop growing at ~1.6 vs heuristic's 7). v7.3 per-city credit didn't fix it. KEY unlock
= two data-driven findings: (1) the on-net's builds are 99% units (diagnosed, not guessed); (2) the
sophisticated 403-line heuristic is only ~RANDOM-level at winning (41.7% random vs 36.9% heuristic
construction) → real headroom above it. So the fix isn't better sparse-reward RL — it's DON'T START FROM
SCRATCH: clone the (mediocre-but-decent) heuristic for a non-collapsed start + a positive-advantage foothold,
then leashed-finetune to climb above it. Progression: BC(20ep,acc0.27)→collapse-solved+parity(bc−off −4.4,
n.s.); BC(120ep,acc0.67)→bc−off +6.6 (n.s., 1 drift outlier); +KL-leash+8seeds→**+14.8pp SIGNIFICANT**.
The weak-clone stage was pure undertraining (acc climbs 0.27@20→0.67@120, no plateau).

**Shippable recipe (opt-in behind flags):** `--control-construction on --bc-pretrain-dir <heuristic-gen dir>
--bc-epochs 120 --construction-kl-coef 0.5 --entropy-coef 0.02` (+ a one-time control-OFF gen for the BC
dataset). Infra: schema-8 `construction_current` recording, `bc_pretrain_construction`, `--bc-pretrain-dir`,
KL leash. Generalizes to the next per-entity heads. See [[selfplay-baseline-variance]] (why replication +
paired diffs are mandatory — the eval is high-variance).

**v7.4 BEHAVIOR-CLONING warm-start — the collapse is SOLVED, but construction is only NEUTRAL (not a win).**
The construction net collapses BELOW random (99% units, ~0%) because it loses ~every game → all-negative
advantages → mode collapse with no positive-signal foothold. Diagnosed with data: on-arm builds 99% units /
2 types, ~1.6 cities (vs heuristic's 7). Also measured (gen random+control-on): the HEURISTIC construction is
only ~RANDOM-level (41.7% random vs 36.9% heuristic, tech/policy random) — it's a sophisticated 403-line
cost-effectiveness/victory-aware policy but NOT better than random at WINNING, so plausible headroom above it.
BC clones the heuristic's picks (record `construction_current` = current build, schema 7→8; supervised CE on
the construction head; `run_loop --bc-pretrain-dir`) to give a non-collapsed ~heuristic start + a
positive-advantage foothold.
**Replicated 3-arm × 4-seed result (off / on=collapse / bc):** off 42.4%±5.1 · on 3.2%±1.9 · bc 38.0%±15.0.
Paired: **bc−on = +34.8pp (t=2.31, SIGNIFICANT → BC escapes the collapse)**; **bc−off = −4.4pp (t=−0.37,
WITHIN NOISE → parity, inconclusive)**; on−off = −39.2pp (t=−9.6, raw construction control collapses).
**Verdict: BC solves the collapse + makes construction SAFE (neutral vs baseline), but does NOT beat it.**
Caveats both point the same way: (1) bc is WILDLY variable (6.4%→73.5%! SE 15) — at seed 4000 it hit 73.5%
(+29pp above off = real headroom captured), at 3000 barely escaped (6.4%); so bc−off is underpowered. (2) The
clone is WEAK — bc_acc stalled at 0.27 on Medium (vs 0.54 Tiny), so bc is a noisy heuristic copy that RL
finetunes from a shaky start. Untested lever: a TIGHTER clone (more BC epochs / lower LR → higher acc) should
cut the variance + give RL a stronger launchpad → decisive test of whether the seed-4000 upside is real.
Infra (schema-8 recording, bc_pretrain_construction, --bc-pretrain-dir + entropy leash) validated + reused by
future heads. See [[selfplay-baseline-variance]].

**v7.3 per-city credit — NEGATIVE, confirmed by a REPLICATED experiment (the definitive result).** After the
single-seed scare turned out to be [[selfplay-baseline-variance]] (NOT a regression), the whole thing was
re-tested properly: 3 arms × 4 seeds, paired per-seed diffs (`python/run_v73rep.sh` + `analyze_v73rep.py`,
small/Medium/16-round/mb0/rw1, 200-game ceiling @ 4242424):
- **off 25.7%±7.7** (high variance: 17.2/8.8/41.7/35.3) · **on-shared 0.4%±0.4** · **on-pcc 0.5%±0.3**.
- Paired **on-pcc − off = −25.2pp** (t≈−3.27, >2·SE — significant DESPITE OFF's variance, because both
  construction arms are ~0 at EVERY seed incl. the one where OFF hits 41.7%). **on-pcc − on-shared = +0.1pp**
  (one extra win at one seed; noise).
- **Two firm conclusions:** (1) controlling production is a REAL catastrophic negative (LOW-variance ~0%,
  unlike the noisy OFF baseline) — confirms v7 under replication, more severely (~26%→0.5%). (2) Per-city
  credit adds NOTHING (≡ shared-adv) — the attribution hypothesis is FALSIFIED; how construction credit is
  assigned doesn't matter, controlling ~250-wide per-city production just breaks the learner.
- SHIP: per-city-credit INFRA (schema-7 econ_city, per-city value head, per-city GAE + PPO ratio,
  `--construction-credit-coef`; 14 tests green) is built+validated+correct, default OFF (coef 0 ≡ v7.2). NEXT
  per-entity heads (promotion/GP/vote) have far smaller action spaces — may work where construction can't.


**v7 (first per-entity head: per-city PRODUCTION/construction) result: NEGATIVE.** Built + validated on
branch `selfplay-v7-construction` (5/6 ACs green: legality, parity, no-op bit-identical, throughput 1.957×,
schema v5). But the EFFECT is negative — construction-ON does NOT beat OFF and does not cross 50% (Medium,
16 rounds, K=4, 200-game ceiling @ seed 4242424):
- small rung: OFF 47.1% → ON **13.7%** (Δ −33pp, z=−7.32) — catastrophic.
- medium rung: OFF 40.2% → ON **36.8%** (Δ −3.4pp, z=−0.71, n.s.).

**ROOT CAUSE (investigated, not guessed): UNIT-BIAS from weak credit assignment.** The net over-builds
UNITS (military) and starves BUILDINGS (economy) beyond mask availability — small-ON chose 58% units vs 42%
offered (+16pp bias → 2%); medium chose 43% vs 35% (+8pp → 37%, more capacity → less bias → less harm).
Economy buildings pay off over a long horizon; terminal-only ±1 reward can't credit-assign that over a
~250-wide per-city action space, so the net defaults to "build a unit now," stalling its economy. The
hand-tuned heuristic knows to build economy (OFF 47%).

**Cadence was tested and is NOT the cause.** v7.1 changed per-turn override → commit-until-done (decide only
at completion; heuristic disabled for controlled civs). That FIXED the training instability (per-turn's
entropy runaway + grad-norm-957 spike from the ~6 construction terms/step dominating the joint PPO objective)
but made the result WORSE (2%) — committing amplifies bad picks. So per-turn churn/objective-domination was a
symptom; unit-bias is the root.

**The machinery is correct** (a fallback would make ON≈OFF, not crater). The per-entity infra (decision loop,
variable recorded blocks, schema v5, seam, per-city net head + ONNX path, training summand) is validated and
is the durable deliverable — **reused by the next per-entity heads (promotion / GP / diplomatic-vote, which
have FAR smaller action spaces and may work where construction doesn't)**.

**v7.2 PBRS DONE — did NOT salvage construction (ON+PBRS 1.5%).** PBRS is honest+correctly-built but
clean-NEUTRAL on tech/policy (37% vs no-PBRS 40%, within noise; the honesty-gate's +8pp was a NaN-
divergence-bug artifact — a starving city's negative food made ln(1+Σfood)=NaN; fixed via coerceAtLeast(0)
+ nan_to_num). ON+PBRS still +12% unit-biased + economy starved (Φ 12.6 vs OFF 16.7). **REFINED ROOT CAUSE:
the blocker is per-city credit ATTRIBUTION, NOT the credit HORIZON** — the per-step GAE advantage is ONE
scalar shared across all heads+cities, so PBRS-shaped economy credit can't be attributed to the specific
city/construction that earned it → construction head gets a non-causal diluted signal, keeps over-building
units. CONSTRUCTION-VIA-RL IS A ROBUST NEGATIVE across 4 approaches (per-turn 14%, commit 2%, buildings-only
[dodge], PBRS 1.5%). Untested bigger levers: per-city counterfactual/difference rewards (the real
attribution fix, hard — needs stable per-city identity); imitation warm-start from the heuristic. PBRS kept
(default coef 0, neutral-honest). The NEXT per-entity heads (promotion/GP/vote) avoid per-CITY attribution.

(history) **v7.2 PBRS (user-directed: "reward too far in the future").** A design-panel-vetted fix:
potential-based reward shaping. Record a per-step log-stabilized economy potential
Φ=ln(1+Σprod)+ln(1+Σfood)+ln(1+Σsci)+ln(1+#techs) (schema 5→6, BLOCK_PHI) and add F=γΦ(s')−Φ(s) to each
step (`--reward-shaping-coef`, default 0). Ng-Harada policy-invariant (F telescopes to a constant → optimal
"win" policy unchanged; only credit TIMING shifts ~250→~5 steps). User APPROVED relaxing the frozen
"no-shaping" rule on this policy-invariant side. Committed (b246778fd). Staged validation: (1) honesty gate
— PBRS on tech/policy, construction OFF, replay-window 1, coef 0 vs 0.1, must be non-regressive + faster
value-fit; (2) on green, construction-ON+PBRS vs OFF (47%) efficacy. v7.1 cadence + v7.1b buildings-only
were tested+rejected (cadence made it worse; buildings-only dodges the credit problem). Default
`--control-construction OFF` until proven. See [[selfplay-roadmap-bottleneck]].
