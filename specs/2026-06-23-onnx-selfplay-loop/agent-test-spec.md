# Test Spec — ONNX policy bridge + self-play loop (test_mode=integration)

Framework: Kotlin = JUnit4 + `GdxTestRunner` (`tests/` module, headless); Python = pytest.
Each acceptance criterion maps to a concrete, runnable test. RED-first (TDD): tests reference
symbols the build creates; they fail now (compile/import error) and go green as the build lands.

## AC2 — LEGALITY (Kotlin): `tests/.../dataplane/OnnxPolicyLegalityTest.kt`
Mirror `FairnessAndDeterminismTests.maskParity_*`. With a tiny fixture model
(`tests/.../resources/policy-test.onnx`, random weights — legality is weight-independent):
- For many (state, turn) samples and both heads: `idx = onnxPolicy.chooseIndex(head, civ, mask, turn)`
  → assert `idx == -1 || mask[idx] == true` (NEVER an illegal index).
- Empty legal mask → `-1`. Unmodeled head ("greatPerson") → `-1`.
- Property holds for argmax (eval) and sampled (gen) modes.

## AC3 — DETERMINISM (Kotlin): `tests/.../dataplane/SelfPlayDeterminismTest.kt`
Mirror the determinism test (`assertArrayEquals(build(), build())`). Fixed model + fixed seed:
- Run two generations of the same Tiny game; assert identical trajectory `calculateChecksum`.
- Run two EVALs; assert identical (games, wins, winrate) — eval reproducible.

## AC4 — PARITY (Python ↔ JVM): `python/tests/test_parity.py`
The anti-drift test. A fixed obs vector + a fixed `policy.onnx`:
- JVM: `./gradlew selfPlay --args="parity <model> <obs_file> <out_file>"` writes JVM `tech_logits`/`policy_logits`.
- Python: load the SAME model via onnxruntime, run the SAME obs → reference logits.
- Assert `np.allclose(jvm, ref, atol=1e-4)` for both heads.

## AC6 — PROVENANCE (Python): `python/tests/test_train_dataset.py`  ← concrete RED test written now
- `unciv_train.dataset` refuses a shard whose `schema_version` ≠ expected (ProvenanceError).
- Refuses a shard whose `ruleset_fingerprint` ≠ expected.
- Accepts a matching VERSION-2 shard and extracts `(obs=concat(global,acting_civ), a_tech, a_policy,
  mask_tech, mask_policy, return)` with the terminal reward broadcast to the civ's non-terminal steps.

## AC1 + AC5 — LOOP + CURVE (smoke): driver dry-run
- `python -m unciv_train.run_loop --rounds 2 --gen-games 2 --eval-games 8 --dry-checks`
  produces a `curve.csv` with the right header and K rows, and `curve.png`. (Full K≥10 run is the
  deliverable, run after green.)

## Supporting unit test (Kotlin): `tests/.../dataplane/SimStatsTest.kt`
`SimStats.binomialTest` matches known values (regression guard for the extraction from `Simulation`).

## RED status (now)
`python/tests/test_train_dataset.py` is written and fails at import (`unciv_train` not built yet) —
verified RED below. The Kotlin tests + remaining Python tests are authored in Phase 3 RED-first.
