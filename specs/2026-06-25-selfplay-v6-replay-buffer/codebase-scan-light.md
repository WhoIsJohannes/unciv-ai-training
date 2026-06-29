# Light Codebase Scan — selfplay-v6-replay-buffer

Mode: BUILD. Thoroughness: medium. Scout via Explore agent + direct reads.

## 1. Experiment drivers (the multi-hour cost)
- `python/run_v5.sh` — v5 driver. Two arms SEQUENTIALLY on a 12/14-core box:
  - **ARM A (primary, the 40.7%)**: `--variant structured --rung small --map-size Medium --rounds 16 --gen-games 16 --eval-games 80 --turn-cap 250 --epochs 8 --lr 1e-3 --gamma 0.99 --lam 0.95 --value-coef 0.5 --entropy-coef 0.01 --clip-eps 0.2 --gen-seed 1000 --eval-seed 999000 --continual --resume --micro-batch-steps 0`
  - ARM B (medium rung): same but `--rung medium --micro-batch-steps 256` → 46.6%.
  - Then per-arm **200-game Medium ceiling eval at `--eval-seed 4242424`** + z-tests vs fixed v4 baselines via `analyze_v5.py` (blind baseline = 58/200 = 28.9%).
- Resumable: every arm uses `--resume`; re-launch continues from last completed round (curve.csv + ckpt/opt sidecars). → the long run can be launched once and resumed.
- **Wall-clock: multi-hour to overnight per arm** (16 rounds, each = JVM self-play gen of 16 games + 8 train epochs + 80-game eval at turn-cap 250).
- `run_acceptance.sh` = older v2 driver (same shape, Tiny+Medium).

## 2. Kotlin MaskedChoice test pattern (template for chooseWithLogp)
- `tests/src/com/unciv/logic/simulation/dataplane/OnnxPolicyLegalityTest.kt`:
  - `choiceIsAlwaysLegalOrMinusOne()` — 3000 fuzz trials, random logits/mask, both eval modes, asserts legality.
  - `emptyLegalSetReturnsMinusOne()`, `evalPicksHighestLegalLogit()`, `sampleStaysInLegalSupportUnderExtremeSkew()`.
  - Fixtures: `FloatArray` logits, `BooleanArray` mask, `Random(seed)`. → mirror for `chooseWithLogp` (assert idx legality + logp == log_softmax(masked)[idx]; empty → (-1, 0f)).

## 3. Python test patterns to mirror
- `test_microbatch.py`: builds synthetic `TrainTrajectory` via `_multi_step_trajectory`; asserts micro-batch == whole-batch (atol ~1e-5/1e-6). **`test_microbatch_noop_when_K_ge_N`** is the exact no-op template for the K=1 replay no-op.
- `test_continual_resume.py`: `_traj(dims, seed)` single/few-step trajectory; warm-from-memory == warm-from-disk (atol 1e-6); **AC6 ONNX parity** torch logits == exported ONNX (atol 1e-4).
- `test_gae.py`: hand-computed `compute_gae` cases (no shard). 
- `test_train_dataset.py`: **`_build_v2_shard`** fabricates a real shard with an `actions` FIXED f32 block `{"name":"actions","dtype":"<f4","kind":"fixed","perItem":0,"len":4}` = `[a_tech,a_policy,-1,-1]`; provenance-mismatch refusal tests. → template for adding a `behavior_logp` FIXED f32 block + a refusal test for v3→v4.
- `test_parity.py`: JVM ORT vs Python ORT logits (atol 1e-4) for contract v1/v2/v3 — the Kotlin↔Python contract-name lock. (Frozen; behavior_logp is post-inference so it does not touch this.)

## 4. Frozen contract surfaces (do NOT touch)
- `core/.../SampleSchema.kt:OnnxContract` + `python/unciv_train/contract.py`: `CONTRACT_VERSION=1/_RICH=2/_STRUCTURED=3`; `META_SCHEMA_VERSION/RULESET_FINGERPRINT/CONTRACT_VERSION/INPUT_WIDTH/TECH_WIDTH/POLICY_WIDTH/INPUT_NAMES`; `INPUT_NAME/OUTPUT_TECH/OUTPUT_POLICY`. Locked by `test_parity.py`. `export_onnx.py` stamps metadata; OnnxPolicy inference path (`buildRichTensors/forwardRich/forward`) byte-unchanged.

## 5. Shard write/read round-trip (behavior_logp will round-trip with NO reader change)
- Write: `core/.../TrajectoryEmitter.kt` (magic+version+header, record framing+CRC) → `Observation.writeBlock` (FIXED → `f32s(values)` little-endian via `ShardFormat.LeBuffer`).
- Read: `python/unciv_dataplane/reader.py:_decode_blocks` (105–140) is **layout-driven**: FIXED block reads `len*itemsize` bytes generically into `step.blocks[name]`. A new FIXED f32 block decodes with zero reader change.
- `dataset.py` reads `s.blocks["actions"][0/1]` → mirror for `s.blocks["behavior_logp"][0/1]`.

## 6. Greenfield check
- grep for `behavior_logp|off.policy|importance.sampling|b_logp` across *.py/*.kt → **NONE**. ("replay" exists only in unrelated UI VictoryScreenReplay.) v5 is on-policy only. v6 is a clean addition.

## Implication for this session
- Code + ALL unit tests (chooseWithLogp, K=1 no-op numerical equiv, off-policy-equivalence stored≈recomputed, behavior_logp shard round-trip, clip_eps guard, v3→v4 refusal) are FAST and are the gate.
- AC1 (no-op reproduces 40.7%) and AC2 (replay sample efficiency, z/p) require the **multi-hour Medium re-run** — resumable, mirrors run_v5.sh as a new run_v6.sh.
