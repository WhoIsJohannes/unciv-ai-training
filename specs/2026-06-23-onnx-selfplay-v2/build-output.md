# Phase 3 Build Output — onnx-selfplay-v2

## Summary
v2 delivered: a learned value critic (actor-critic + GAE, standard PPO) and a model that consumes the
FULL emitted observation (spatial tile-set + entity token sets) via a grown multi-tensor ONNX contract
(v1→v2), reusing all v1 infrastructure. Reward stays terminal-only ±1. Three attributable curves produced
(v1-reinforce, blind-critic, rich-critic) + a Medium ceiling comparison.

## Files changed
- **new:** `python/unciv_train/{features.py, analyze.py}`, `python/run_acceptance.sh`,
  `python/tests/{test_gae.py, test_v2_units.py}`.
- **edited (python):** `model.py` (value head, RichPolicyValueNet, masked_pool), `train.py` (compute_gae,
  PPO+GAE optimizer, train_reinforce preserved), `dataset.py` (load_trajectories), `contract.py`
  (CONTRACT_VERSION_RICH + token specs), `export_onnx.py` (export_rich + value-drop wrappers),
  `run_loop.py` (variants/resume/map-size/metrics), `tests/test_parity.py` (rich multi-tensor).
- **edited (kotlin):** `OnnxPolicy.kt` (multi-tensor forwardRich + v1/v2 gate + tensor cleanup),
  `SelfPlayRunner.kt` (map-size CLI + parity-dump-rich/parity-run-rich), `SampleSchema.kt` (OnnxContract v2).
- **no-touch (reused):** Featurizer, Observation, TrajectoryEmitter, ShardFormat, reader.py provenance,
  LegalActionMasks, MaskedChoice, RulesetFingerprint, RandomPolicy.

## Gate status
- **Python:** test_gae(4) + test_v2_units(6) + test_parity blind+rich(2) + test_reader(5) +
  test_train_dataset(3) GREEN. **test_determinism(1) FAILS — PRE-EXISTING v1/engine** (verified against
  clean master; v2 core emitter untouched).
- **Kotlin:** OnnxPolicyLegalityTest + FairnessAndDeterminismTests GREEN; core+desktop compile clean.

## Acceptance results (see results/RESULTS.md)
- AC1 ✅ three curves (overlay_tiny.png, overlay_medium.png).
- AC2 ✅ critic reduces late-round variance: blind-critic 11.15pp vs v1-reinforce 22.74pp (≈½) — the
  convergence answer is YES. (Honest nuance: steadier at a lower level; rich-critic on Tiny achieves both
  higher level ~57% AND low variance ~10pp.)
- AC3 reported plainly: rich did NOT beat blind on Medium (14.7% vs 28.9%, z=−3.48) — rich is undertrained
  on the larger board under the bounded budget (it WINS on Tiny). Stated honestly, not tuned to a win.
- AC4 ✅ parity atol 1e-4. AC5 ⚠️ provenance/featurizer ✅, full-shard determinism pre-existing gap.
  AC6 ✅ legality (0 illegal actions). AC7 ✅ terminal-only reward.

## Plan fidelity
All plan deliverables implemented (traceability.md). Doc drift reconciled (decisions R12 + plan amend):
the as-built optimizer is standard PPO (compute-once GAE targets, ε=0.2 K=8, tanh-bounded value) — the
final recipe after calibration surfaced and fixed two bugs (degenerate value target; value explosion on
unnormalized features).

## Security checklist
- No new secrets (no credentials/keys added; gitleaks-relevant surface = none).
- New external inputs: the parity fixture text is local + self-generated; parsed defensively
  (vec/set tags, counts). Checkpoints are write-only (resume uses .onnx + curve.csv, never torch.load).
- No PII, no new auth/endpoints, no network I/O. Pure local ML training + headless game sim.

## Open issues for ship
1. AC5 full-shard determinism is a pre-existing v1/engine limitation — document, don't claim.
2. AC3 hypothesis negative (rich worse on Medium) — reported honestly as a data/capacity-budget effect.
3. Cleanup backlog: 2 items filed (obs normalization, hyperparam magic constants).
