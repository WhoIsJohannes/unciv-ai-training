# Light Codebase Scan — v4 Structured Encoder

**Date:** 2026-06-24 · **Base:** master `5ac0a3cf6` (= v3 rich-critic) · **Mode:** BUILD · **Size:** L

Grounded map of the v3 rich-critic (contract-v2) pipeline that v4 extends. Line numbers
**verified against the current tree** (Explore scan + direct grep). The plan's cited line
numbers were checked: **no material drift** (all within ±2 lines).

---

## Contract pipeline (the spine v4 must move in lockstep)

Kotlin emit → shard → Python read → model → ONNX export → JVM bridge. The contract version
must change in lockstep across **all** of these:

| Concern | Kotlin | Python |
|---|---|---|
| Schema/layout version | `SampleSchema.VERSION = 2` (`SampleSchema.kt:22`) | reader refuses mismatch (perishable shards) |
| ONNX contract version | `OnnxContract.CONTRACT_VERSION_RICH = 2` (`SampleSchema.kt`) | `contract.CONTRACT_VERSION_RICH = 2` (`contract.py:18`) |
| Token-set names | `OnnxContract.RICH_TOKEN_NAMES` | `contract.RICH_TOKEN_NAMES` (`contract.py:29`) |
| Spatial channels | `SampleSchema.SPATIAL_CHANNELS` (13) + `NUM_SPATIAL_CHANNELS` (`:86-101`) | `token_specs_from_schema` reads `spatial_channels` (`contract.py:85-102`); fallback 13 (`:81-82`) |
| Fingerprint | `Vocab.canonicalSections()` hashes `"schema:spatialChannels"` → changing channels changes the fingerprint | gated on import (`OnnxPolicy.kt:52-63`) |

`RICH_TOKEN_NAMES = [spatial, own_units, opp_units, own_cities, opp_cities, civ_tokens]`.
Current widths: spatial **13**, own_units/opp_units **8**, own_cities/opp_cities **16**, civ_tokens **84**.
`MASK_SUFFIX = "_mask"`. ONNX export tensor order is positional/strict (`export_onnx.py:102-120`).

---

## Python model + training (DO NOT MODIFY the encoder-agnostic core)

- `model.py` — `RichPolicyValueNet` (`:79-113`); `_TokenEncoder` (`:66-76`); `masked_pool`
  (`:50-63`, mean‖max with NaN guards: mean / clamp(count,1), max over empty→0); value head
  near-zero init `_small_init_value_head` (`:20-26`). `forward(inputs:dict) → (tech_logits,
  policy_logits, tanh(value))` (`:108-113`). Reads keys: `global`, `acting_civ`, and per token
  set `name` + `name+"_mask"`. Constructor defaults `token_dim=32, hidden=256`.
  **This `forward` signature is the stable seam — v4's new encoder must preserve it.**
- `train.py` — `train_actor_critic_rich` (`:240-275`); encoder-agnostic core
  `_optimize_actor_critic` (`:115-194`); `compute_gae` (`:45-64`). Terminal-only reward,
  V(terminal)=0, PPO clip 0.2 (off by default), NaN/divergence guard restores last-good and
  never exports. Whole-round dense batch over all tiles at `train.py:265`. **DO NOT MODIFY
  these** (AC6). v4 only swaps the `nn.Module` passed in.
- `features.py` — `build_rich_batch` (`:33-48`), `build_rich_single` (`:51-69`, the parity
  reference), `_pad_token_set` (`:17-30`, pads to max(1,count) with presence mask).

## Python contract / export / dataset / loop

- `contract.py` — `CONTRACT_VERSION_RICH=2` (`:18`); `RICH_TOKEN_NAMES` (`:29`); `Dims` (`:45-56`);
  `token_specs_from_schema` (`:85-102`); `_TOKEN_WIDTH_FALLBACK` (`:81-82`); metadata keys (`:31-38`).
- `export_onnx.py` — `export_rich` (`:82-138`); `_RichPolicyOnly` policy-only wrapper (`:32-44`);
  strict `names` order (`:102-120`); dynamic axes batch `{0}` + ragged `{1:"n_<name>"}`; opset 17;
  stamps schema_version, ruleset_fingerprint, contract_version, widths, input_names.
- `dataset.py` — `RICH_TOKEN_BLOCKS` (`:37`); `_rich_step_blocks` (`:106-121`); `.rich` per-step
  blocks attached when rich=True (`:163`).
- `run_loop.py` — `train_round` variant dispatch (`:74-94`): `v1-reinforce` / `blind-critic` /
  `rich-critic`; export dispatch (`:212-222`); eval JSON `onnx_decisions` + `games` (`:226-231`).

## Kotlin dataplane (emit side)

- `Featurizer.kt`
  - `buildGlobal` (`:153-166`): head(5)=turns, era, tileCount, knownMajors, aliveMajors + demographics.
    **No map dims today** → D1.2 adds radius/worldWrap/shape here.
  - `buildActingCiv` (`:168-183`): head(13) + tech/policy/branch one-hots.
  - `writeCityToken` (`:185-208`, width 16): fields incl. `majority_religion` (vocab.religion+1)
    and `current_construction` (`:205`). **No tile position today** → D1.3 adds tile index.
  - `writeUnitToken` (`:210-223`, width 8): presence, isOwn, ownerSlot, unitTypeCat,
    health_bucket, capital_dx, capital_dy, promotion_count. **No tile index today** (units already
    *sorted* by `currentTile.zeroBasedIndex` but it isn't emitted) → D1.3 adds it.
  - `buildSpatial` (`:233-262`, 13 channels/tile): 0 visibility, 1 terrain_base, 2 terrain_feature,
    3 resource, 4 road, 5 river, 6 is_city_center, 7 owner_slot (0 none, 1..N civ, 255 self),
    8 improvement, 9 unit_present, 10 unit_owner_slot, 11 unit_type_cat (0-4), 12 unit_health_bucket.
    `+1` then `coerceAtMost(255)` on categorical channels. **No (x,y) today** → D1.1 adds 2 coord channels.
- `SampleSchema.kt` — `VERSION=2` (`:22`); `OnnxContract` (`:36-62`); `SPATIAL_CHANNELS` (`:86-100`);
  `NUM_SPATIAL_CHANNELS` (`:101`).
- `Vocab.kt` — indexers return **-1 on miss** (`tech/building/unit/policy/resource/terrain/improvement/
  religion`); counts `terrainCount/resourceCount/improvementCount/buildingCount/unitCount/religionCount/
  eraCount`; canonical sections drive both indexing and fingerprint (`:80-100`). **Confirm `maxCivTokens`
  during build** (plan assumes slot table size `maxCivTokens+2 = 42` ⇒ maxCivTokens=40).
- `FairOpponentModel.kt` — `encode`/`tokenWidth` (`:80-163`): fair per-civ token (84 dims, GnK),
  unmet→all-zero+masks-0; tech/spaceship COUNT only; demographics rank+bucket only.

## JVM ONNX bridge

- `OnnxPolicy.kt` — provenance gate (`:52-63`); `rich = mContract==CONTRACT_VERSION_RICH` (`:59`);
  rich-input inventory check (`:66-74`); `forwardRich` (`:127-138`); `buildRichTensors` (`:153-166`);
  `richTensorsFromArrays` (`:171-189`); `tokenTensors` (`:196-204`, empty→N=1 zero-token, mask 0).
  All tensors closed in `finally` (`:136,186,201`).

## Tests + benchmark + hex primitives

- Parity: `python/tests/test_parity.py` — `test_jvm_python_logits_match` (v1 blind, `:37-66`) and
  `test_jvm_python_rich_logits_match` (v2 rich multi-tensor, `:77-131`, atol **1e-4**, incl. empty
  entity-set path). JVM side via `./gradlew selfPlay --args="parity-run-rich <model> <obs> <out>"`.
  **D10 extends the rich parity test** to the wider input + logits.
- Dataplane tests: `tests/src/com/unciv/logic/simulation/dataplane/OnnxPolicyLegalityTest.kt`
  (3000-trial legality fuzz), `FairnessAndDeterminismTests.kt`. **D1.4 construction-bug unit test
  lands alongside these.**
- `Timers.kt` — `timeThis(name){block}` instance (`:85`) + companion (`:107`). Used by D8 to wrap
  `OnnxPolicy.forwardRich`.
- `HexMath.kt` — `clockPositionToHexcoordMap` (`:277-285`): 6 offsets 12=(+1,+1) 2=(0,+1) 4=(-1,0)
  6=(-1,-1) 8=(0,-1) 10=(+1,0); offset-grid note (`:101-106`). `TileMap.getIfTileExistsOrNull`
  (`:376-398`): world-wrap retries `getOrNull(x±radius, y∓radius)`; hex radius = `mapSize.radius`,
  rectangular radius = `width/2`. **D3's Python adjacency builder replicates this.**

---

## ⚠️ Corrections to the plan's prerequisite self-check

1. **`SimBenchmark` ALREADY EXISTS** (plan says "does NOT exist — v4 creates it"). It is
   `desktop/src/com/unciv/app/desktop/SimBenchmark.kt` (10.7 KB), gradle task `:desktop:simBench`
   (`desktop/build.gradle.kts:60`, `mainClass=com.unciv.app.desktop.SimBenchmark`). It already
   measures the **heuristic-baseline turns/s** (A* off/on, single + multi-thread, Medium, 6 majors +
   6 CS + barbs; `BENCH|`-prefixed output) — i.e. D8's required heuristic baseline is already there.
   It does **not** yet exercise the ONNX-policy path (ms/decision, ONNX-driven turns/s, the 70% gate).
   ⇒ **D8 EXTENDS the existing benchmark** (add an ONNX-policy mode + the 70%-of-heuristic rejection
   gate), it does not build it from scratch. Strict reuse-over-rebuild.

## Risks / surprises for the v4 extension

1. **Spatial-channel count is referenced in many places** — must move in lockstep: `buildSpatial`
   loop + `SPATIAL_CHANNELS` + `NUM_SPATIAL_CHANNELS` + Vocab fingerprint section + `contract.py`
   fallback + `OnnxPolicy` fallback + parity-test fixtures. D1's signed (x,y) channels need a
   decision (i16 side-tensor vs biased-u8) — `coerceAtMost(255)` would corrupt negative coords.
2. **Construction-namespace collision** (`Featurizer.kt:205`, `writeCityToken`): unit branch lacks
   the `+ buildingCount` offset → building#k and unit#k collide. D1.4 fixes; D2's single
   `buildingCount+unitCount+1` construction embedding depends on the fix.
3. **Fingerprint coupling** — any spatial change invalidates old fingerprints (datasets perishable;
   regenerate — expected).
4. **`token_specs_from_schema` silently falls back** to hardcoded widths if `perItem` missing — a
   schema-emit regression would not alarm. Keep schema emission authoritative.
5. **Strict ONNX tensor-name ordering** — extending widths is fine (coords ride inside `spatial`,
   map dims inside `global` → no new tensor names per D9), but order must stay fixed.
6. **No structured-encoder scaffold exists** — current encoders are uniform MLP+pool. D3/D4/D5
   (GNN, self-attention, cross-attention) are net-new modules; ONNX-exportability of GNN
   gather/scatter under opset 17 is the pre-registered export risk to validate early.
7. **Slot table (=42) assumes maxCivTokens=40** — verify in `Vocab.kt`/`SampleSchema.kt` before D2.
