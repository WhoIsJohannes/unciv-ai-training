# Deep Codebase Scan — selfplay-v6-replay-buffer (collision-risk map)

Very-thorough scan. 5 critical collision sites (all mitigated by the task's design) + the
fixture-update surface forced by the VERSION 3→4 bump. No silent positional-index breakage.

## 1. RNG / determinism (choose → chooseWithLogp)
- ONLY production caller of `MaskedChoice.choose` = `OnnxPolicy.chooseIndex` (desktop OnnxPolicy.kt:105),
  one draw/decision, per-(civ,turn) RNG from `DataPlaneHooks.defaultRngFor` stateBasedRandom.
- `tests/.../OnnxPolicyLegalityTest.kt` calls `choose` directly at ~6 sites with fixed `Random(seed)`.
  Routing `choose` through `chooseWithLogp(...).first` (single draw, identical order) ⇒ these tests
  must still pass UNCHANGED (delegation preserves the stream). Add a NEW chooseWithLogp logp test.

## 2. Layout indexing — ALL name-keyed (safe to append behavior_logp)
- Kotlin: `Observation.blocks.first{it.name==…}`, Featurizer builds named list, no positional reads.
- Python: reader `_decode_blocks` iterates `layout` by name; dataset/features/contract all dict-keyed
  by block name. No code assumes block COUNT or ORDER. Appending behavior_logp after actions is safe.

## 3. 'actions' is appended at EMIT time (ShardRecorder.recordStep, DataPlaneHooks.kt:221), NOT in Featurizer
- behavior_logp MUST follow the SAME pattern: appended in recordStep right after the actions block
  (exactly as the task says). Building it in Featurizer would interleave it among entity tokens and
  break fixed-layout fixtures.

## 4. CONTRACT FREEZE — export stamps shard SCHEMA_VERSION into the model's META_SCHEMA_VERSION
- `export_onnx.py` sets `META_SCHEMA_VERSION = str(schema_version)` where `schema_version =
  contract.schema_version_from_schema(schema)` (run_loop ver), and `META_CONTRACT_VERSION =
  CONTRACT_VERSION*` (separate constant, UNCHANGED).
- ⇒ Bumping shard VERSION 3→4 makes the model's `schema_version` metadata become 4. OnnxPolicy checks
  `mSchema == expectedSchemaVersion` (live SampleSchema.VERSION = 4) ⇒ BOTH move to 4 in lockstep ⇒
  still MATCH within a v6 run. The CONTRACT_VERSION_STRUCTURED=3, the I/O names/widths, and META_* KEYS
  are byte-unchanged; the inference path (buildRichTensors/forwardRich/forward) is untouched.
- AC4 "byte-unchanged" is about the contract STRUCTURE (names/widths/keys/CONTRACT_VERSION*) — the
  META_SCHEMA_VERSION *value* legitimately moves 3→4 (the perishable-dataset gate working as intended).
- A v3 model will NOT load on v4 code (provenance gate) — INTENDED. --resume reloads warm net+opt
  sidecars (.pt), then re-exports a v4 model; it never loads a v3 .onnx under v4.

## 5. --resume / keep-shards + replay refill ordering
- Prune deletes `round_{r-keep_shards}` at the END of round r (run_loop.py:340), AFTER training.
- In-process deque is appended at LOAD time each round (before that round's prune) ⇒ ordering safe.
- Floor `keep_shards = max(args.keep_shards, replay_window-1)` keeps the last K rounds' round_*/ dirs.
- On --resume restart: refill deque from disk reading rounds [start-K .. start-1] BEFORE the loop;
  glob `round_{i}/*.bin`. Missing dir → warn + refill what exists (don't crash). Early fresh rounds
  naturally have < maxlen entries (deque handles it). Round 0 excluded from the window.

## 6. Tests forced to update by VERSION 3→4 (lockstep — REQUIRED for pytest green)
- `test_train_dataset.py`: `_build_v2_shard(version=2)` default → must build version=SCHEMA_VERSION(4);
  the v1-mismatch case `_build_v2_shard(version=1)` stays (still a mismatch vs 4). If any path feeds
  `load_trajectories`, the fixture layout must include a `behavior_logp` block.
- `test_v2_units.py`: `_shard_with_steps()` hardcodes `"schemaVersion": 2` + feeds load_trajectories
  (rich) → bump to 4 AND add behavior_logp block (else KeyError on `s.blocks["behavior_logp"]`).
- `test_parity.py` / `test_structured_smoke.py` / `test_continual_resume.py`: `export(_rich)(...,
  schema_version=2/3/CONTRACT_VERSION_STRUCTURED)` — these export+infer in-process (no shard reader
  gate); verify each still passes, parametrize from SCHEMA_VERSION where it's a real version (not the
  contract constant) to avoid drift.

## Top surprises
- export stamps the SHARD schema version into the MODEL metadata → the 3→4 bump flows into
  META_SCHEMA_VERSION (lockstep, still matches within a v6 run; v3 models intentionally refuse on v4).
- 'actions' (and so behavior_logp) is appended at SHARD-EMIT time in recordStep, not in the Featurizer.
- The VERSION bump silently breaks ~2 shard-fixture tests (hardcoded version=2/3 + no behavior_logp
  block) — these MUST be updated in lockstep or `pytest` goes red; this is the main hidden work item.
- Replay refill MUST read kept round dirs BEFORE prune deletes them; the keep_shards floor guarantees
  the window survives, and the in-process deque append-at-load ordering is already correct.
- No positional/index-based block access anywhere → appending behavior_logp is structurally safe.
