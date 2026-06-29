# RESULTS — selfplay-v7-construction (first per-entity head: city PRODUCTION)

## The question
v5's policy (tech + policy only) sits BELOW the symmetric 50% break-even (small-rung 40.7%). v7 gives the
net its first **per-entity** lever — per-city production. **Does controlling production move the learner
toward/past 50% on Medium?** Ship criterion (D-C5): ship if construction-ON provably beats OFF (p<0.05)
within a rung, **even below 50%**; crossing 50% is a reported milestone, not a gate.

## Machinery validated (all pre-run ACs green — BEFORE the Medium run, per the acceptance ordering)
| AC | Result |
|---|---|
| #1 LEGALITY | ✅ gen-ON smoke: 1106 per-city decisions, **0 illegal**; recorded action ∈ legal mask, recorded==applied |
| #2 PARITY | ✅ JVM/ORT construction logits == torch reference, atol 1e-4; `constructionId` inverts the 0-indexed mask space (round-trip asserted) |
| #4 SCHEMA | ✅ VERSION 4→5 lockstep (Kotlin+Python); two new VARIABLE blocks round-trip with no reader change; old v4 shards refused |
| #5 NO-OP | ✅ construction-OFF is a **bit-identical** no-op vs v6 (deterministic zero-summand oracle: max\|Δw\|<1e-6), incl. under K=4 replay |
| #6 THROUGHPUT | ✅ bench-onnx construction-ON ratio **1.957 ≥ 0.70** (per-city head cheaper than the heuristic cost-effectiveness eval) |
| determinism | ✅ single RNG draw/decision, identical order; the byte-identical-shards test is a pre-existing flake on this box (clean master also non-identical), NOT a v7 regression |

## Experiment (AC#3 — EFFECT) — IN PROGRESS
4 arms, all `--replay-window 4 --continual --micro-batch-steps 256`, structured/Medium, 16 rounds,
gen 16 / eval 80, turn-cap 250; construction is the ONLY axis within each rung pair:

| arm | rung | construction |
|---|---|---|
| structured-small-off | small | OFF (== v6 tech+policy) |
| structured-small-on | small | ON |
| structured-medium-off | medium | OFF |
| structured-medium-on | medium | ON |

Then per-arm 200-game Medium ceiling @ eval-seed 4242424 (ON arms eval **with** construction control), and
`analyze_v7`: per-rung two-proportion one-sided z-test (H1 ON>OFF) + each arm vs the 50% break-even.

Launch: `THREADS=12 OUT_ROOT=../training-runs/v7 python/run_v7.sh` (resumable; a bench-onnx PR5 pre-gate runs first).

### Results (to be filled when the run completes)
- small: OFF=__/200 · ON=__/200 · Δ=__ · z=__ p=__ → construction_helps=__ · ON crosses 50%? __
- medium: OFF=__/200 · ON=__/200 · Δ=__ · z=__ p=__ → construction_helps=__ · ON crosses 50%? __
- **Ship recommendation**: _(analyze_v7 — SHIP if directional win in ≥1 rung)_

## Interpretation framing (read before the numbers land)
Construction is now a **modeled head**: the learner uses its net, the opponent (RandomPolicy) uses
uniform-legal — consistent with how tech/policy are modeled (the opponent is "random" on every modeled
head). So the ON-vs-OFF delta measures "net-driven construction beats a random construction baseline, net
of everything." A clean directional win (ON>OFF, p<0.05) means the net learns useful production policy and
is the ship signal — whether or not it crosses the 50% symmetric break-even (the milestone). A null/negative
result is reported honestly; the per-entity infra still ships (default OFF) for the next promotion / great-
person / diplomatic-vote heads, which reuse it.
