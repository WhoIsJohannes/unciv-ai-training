# onnx-selfplay-v2 — SHIPPED

**Feature:** v2 self-play upgrade — a learned value critic (actor-critic + GAE, standard PPO) and a model
that consumes the FULL emitted observation (spatial tile-set + entity token sets) via a grown multi-tensor
ONNX contract (v1→v2). All v1 infra reused; reward stays terminal-only ±1.

**Verdict:** SHIPPED (local squash-merge to master; solo repo, no GitHub PR).
Diff vs master:  75 files changed, 18504 insertions(+), 63 deletions(-).

## The two questions answered
- **Will it converge?** YES — the learned critic **halves late-round win-rate variance** (blind-critic
  last-4 stddev 11.15pp vs v1-reinforce 22.74pp). AC2 met.
- **Does seeing the board help?** Mixed/honest: rich-critic is **best on Tiny** (late mean ~57%, low
  variance) but **undertrained and worse on Medium** under the bounded budget (14.7% vs 28.9%, reported
  plainly — AC3 negative, not tuned to a win).

## Acceptance
AC1 ✅ three curves · AC2 ✅ critic reduces variance · AC3 ❌-hypothesis (reported plainly) ·
AC4 ✅ JVM↔Python multi-tensor parity (atol 1e-4) · AC5 ⚠️ provenance+featurizer-determinism ✅,
full-shard determinism pre-existing v1/engine gap · AC6 ✅ legality (0 illegal actions) ·
AC7 ✅ terminal-only reward.

## Files (by role)
- **trainer (python):** model.py (value head + rich net), train.py (compute_gae + PPO+GAE optimizer),
  dataset.py (trajectories), features.py (multi-tensor assembly), contract.py + export_onnx.py (v2),
  run_loop.py (variants/resume/map-size), analyze.py (AC analysis), run_acceptance.sh.
- **bridge (kotlin):** OnnxPolicy.kt (multi-tensor + v1/v2 gate), SelfPlayRunner.kt (map-size + rich parity),
  SampleSchema.kt (OnnxContract v2).
- **tests:** test_gae.py, test_v2_units.py, test_parity.py (+rich).
- **results:** results/RESULTS.md + acceptance-report.md + overlay plots + curve CSVs.

## Council
Intake 42 + plan 41 + ship 37 findings; all critical+major addressed (decisions R1–R12 + ship fixes:
tensor leak, NaN poisoning, defensive bridge, provenance inventory, fail-loud map-size).

## Open / follow-ups (cleanup-opportunities.md)
1. Observation normalizer (training quality). 2. Centralize training hyperparam magic constants.
3. (research) weight carryover across rounds + larger Medium budget to retest the AC3 hypothesis.
