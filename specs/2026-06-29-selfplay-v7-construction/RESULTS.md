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

## What this does NOT rule out (explicit v7.1 follow-ups — not run here)
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
