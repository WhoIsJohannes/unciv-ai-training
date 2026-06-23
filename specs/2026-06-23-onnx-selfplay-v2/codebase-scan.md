# Phase 2 — Deep Codebase Scan (v2 design-critical findings)

## A. Spatial grid reconstruction — VERDICT: NOT recoverable without new emitted data
- `Tile.zeroBasedIndex` = **tile-insertion order** (`TileMap.kt:610-617` assigns `index` from `values.withIndex()`), NOT a row-major function of position.
- `HexMath` HAS clean hex→offset formulas: `getColumn = y - x`, `getRow = (x+y)//2` (`HexMath.kt:152-156`), and `TileMap` stores `leftX/bottomY/width` transients. BUT these require each tile's `position(x,y)`, which is **NOT in the shard** — only the `spatial[nTiles×13]` block indexed by zeroBasedIndex is emitted. `schema.json` carries `nTiles` only (`DataPlaneHooks.kt:167`), no per-tile coords, no W/H.
- Maps are **hexagonal** for Tiny/Medium (radius 10 → 331 tiles, radius 20 → 1261 tiles) ⇒ nTiles ≠ W×H; a bounding-box grid would be sparse with many invalid cells.
- **Bottom line (from the scan's own Top-Surprises + Actionable Summary):** 2D (col,row) is derivable ONLY if tile positions are emitted (forbidden) or the generator's insertion order is replayed in Python (fragile). → **Use per-tile-token + positional-feature + masked-pool fallback.** (Council 🔴 "2D grid reconstruction mathematically impossible" concurs.)

## B. Multi-tensor input on the JVM (OnnxPolicy growth) — mechanically straightforward
- `Observation.block(name): FloatArray` returns any block's raw values; VARIABLE block count = `values.size / perItem`. Accessors exist for entity token lists.
- onnxruntime-java: `OnnxTensor.createTensor(env, buffer, longArrayOf(...))` for each named tensor; `session.run(Map<String,OnnxTensor>)` already used for the single "obs" — multi-tensor is the same call with more entries. Feed u8 `spatial` as **float32** (simplest; matches training dtype) rather than ByteBuffer/int8 to avoid dtype friction.

## C. Map-size parametrization — easy, dynamic
- `SelfPlayRunner.mapParameters()` hardcodes `mapSize = MapSize.Tiny` (`SelfPlayRunner.kt:111-120`). Add a CLI arg threaded `gen`/`eval` arg array → `mapParameters(seed, mapSizeName)` → `GameSetupInfo` → `Simulation`. `MapSize("Medium")` constructor resolves the predefined (radius20, 44×29, 1261 tiles).
- `buildSpatial` sizes `FloatArray(tiles.size * channels)` dynamically and `nTiles` is read live ⇒ Medium shards "just work"; schema.json records the actual nTiles.

## D. Collision / drift map for the contract bump (1→2)
- Must change in lockstep: Python `contract.py` (`CONTRACT_VERSION`, `INPUT_NAME`→named-tensor set, per-tensor dyn axes, metadata props) ↔ Kotlin `SampleSchema.OnnxContract` (`CONTRACT_VERSION`, input names, META_* keys). `OnnxPolicy.init` provenance gate reads META_SCHEMA_VERSION / META_CONTRACT_VERSION / META_RULESET_FINGERPRINT — keep all three, add the new contract version.
- Parity: `SelfPlayRunner.parityDump` writes `block("global")+block("acting_civ")` CSV; `parityRun` reads CSV → single "obs" tensor; `test_parity.py` mirrors. v2: dump/run a **multi-tensor obs** (JSON of named arrays) on both sides.
- **No other consumers** of `INPUT_NAME="obs"` / the concat beyond OnnxPolicy, export_onnx, parityDump/Run, test_parity. `dataset.py` action decode (`actions[0]`,`actions[1]`) is unaffected by the value-critic change.

## E. Run reality
- gradle `selfPlay` JavaExec, `mainClass=SelfPlayRunner`, `jvmArgs=["-Xmx8G"]` (`desktop/build.gradle.kts:69-76`). `./gradlew selfPlay --args="..."`.
- python deps declared (torch/onnx/onnxruntime/numpy/matplotlib) but **no venv present** — must `pip install -e ./python` (or equivalent) before training. No scipy (binomial lives in Kotlin SimStats).

## Top surprises
1. **Spatial positions are NOT in the shard** — only `nTiles×13` by insertion-order index. Confirms grid-CNN is infeasible without forbidden new data → token-pool fallback is the correct, sanctioned path.
2. **`zeroBasedIndex` ≠ row-major** — it's insertion order, so even knowing W/H wouldn't place tiles correctly without positions. Settles the fork decisively.
3. **Medium is trivial to enable** (CLI arg; nTiles dynamic) — the only blocker for the ceiling test is encoder choice, now resolved.
4. **Contract 1→2 is a hard break** — old .onnx rejected by the new gate; perishable-artifact regeneration is expected. v1-REINFORCE baseline must therefore run via a preserved `--variant`, not the v2 model path.
5. **u8 spatial → feed as f32** on the JVM to avoid int-tensor typing; PyTorch input dtype must match.
6. **No python venv yet** — environment setup is a prerequisite Phase-3 step before any training.
