# Phase 3 Checkpoint — onnx-selfplay-v2

## What was built (all 3 stages, validated)
- **Stage A (value critic, blind):** `PolicyNet` value head (tanh-bounded, dropped at export);
  `compute_gae`; standard-PPO `_optimize_actor_critic` (compute-once targets, ε=0.2, K=8,
  log-ratio clamp); `load_trajectories` (ordered, terminal-only reward, keeps no-action steps);
  `train_reinforce` preserved for the v1 baseline; `run_loop --variant/--resume/--map-size` +
  per-round metrics + checkpoints.
- **Stage B (rich rep + contract v2):** `features.py` (multi-tensor padding + masks);
  `RichPolicyValueNet` (masked mean+max pool, NaN-guarded); contract v2 lockstep
  (`contract.py` ↔ `SampleSchema.OnnxContract`); `export_rich` (multi-tensor, dynamic axes, value
  dropped); `OnnxPolicy.forwardRich`/`buildRichTensors` (v1+v2 gate, all tensors closed);
  `SelfPlayRunner parity-dump-rich/parity-run-rich`.
- **Stage C (eval):** map-size CLI threaded gen/eval (Tiny/Medium); `analyze.py` (AC1/AC2/AC3 +
  overlay plots, two-proportion test); `run_acceptance.sh` reproducible driver.

## Key files
new: `python/unciv_train/{features.py, analyze.py}`, `python/run_acceptance.sh`,
`python/tests/{test_gae.py, test_v2_units.py}`. edited: `model.py, train.py, dataset.py, contract.py,
export_onnx.py, run_loop.py`; `OnnxPolicy.kt, SelfPlayRunner.kt, SampleSchema.kt`; `test_parity.py`.

## Gate status
- Python: `test_gae` (4) + `test_v2_units` (6) + `test_parity` blind+rich (2) + `test_reader` (5) +
  `test_train_dataset` (3) all GREEN. **`test_determinism` FAILS — PRE-EXISTING v1/engine** (confirmed
  by reverting the 3 bridge files to master: same failure; core emitter untouched by v2).
- Kotlin: `OnnxPolicyLegalityTest` + `FairnessAndDeterminismTests` (featurizer same-state→same-bytes,
  mask parity, leakage, fingerprint) GREEN. desktop+core compile clean.
- Smokes: blind-critic loop end-to-end; rich-critic loop end-to-end (49 ONNX decisions, 0 illegal).

## Open issues
1. **AC5 determinism** is split: featurizer + provenance hold; full same-seed→identical-shards is a
   pre-existing engine limitation, out of v2 scope — report honestly, do NOT claim it.
2. Full training runs to acceptance: IN PROGRESS in the background (`run_acceptance.sh`); analysis +
   AC1/AC2/AC3 verdicts pending completion.

## Test spec / council
- Test spec: `agent-test-spec.md` (= `test_gae.py`). test_mode=integration.
- Council: intake 42 findings + plan 41 findings, all critical+major folded (decisions R1–R12).
