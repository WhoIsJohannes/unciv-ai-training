# Design — C1-kotlin-emit-contract

## Summary
Cluster C1 widens the v3 rich contract to v3 entirely by extending existing tensor WIDTHS (no new token-set names): spatial 13->15 (append plain float32 tile_x/tile_y outside the fog gates), global head 5->8 (radius, worldWrap bit, shape ordinal — the data Python needs to replicate TileMap.getIfTileExistsOrNull), unit token 8->9 and city token 16->17 (zeroBasedIndex). The construction-namespace collision at Featurizer.kt:205 is fixed with a buildingCount offset. The neighbor graph rides as TWO NEW model-input tensors spatial_adj [1,N,6] int64 + spatial_adj_mask [1,N,6] float32, derived in Python at train time and built by OnnxPolicy from the live TileMap at inference (real getIfTileExistsOrNull / tile.neighbors, no replication on the JVM); they are NOT folded into spatial because spatial is a [B,maxN,15] padded float token set whose ragged-N and dtype cannot carry per-row int64 neighbor indices without corrupting the masked pool and the parity layout. VERSION 2->3 and CONTRACT_VERSION_RICH 2->3 move in lockstep across all seven width/version sites; all fallbacks become fail-loud against the schema.

## Detailed design
## C1 design — Kotlin emit side + contract + inference graph build

Base worktree: `/Users/j/Unciv-onnx-selfplay-loop` (branch onnx-selfplay-loop = master 5ac0a3cf6).
All line anchors below are against the verbatim source I read in this session.

---

### Decision 0 (load-bearing): neighbor graph = two NEW input tensors, NOT folded into spatial

`spatial_adj` `[1, N, 6]` **int64** + `spatial_adj_mask` `[1, N, 6]` **float32**.

Justification (settles the prompt's mandated fork):
1. **Dtype mismatch.** The spatial token set is a `[B, maxN, 15]` **float32** tensor consumed by `_pad_token_set`/`masked_pool` (features.py:17-30, model.py:50-63). Neighbor indices are integers in `[0, N)` used as ONNX `Gather` indices; Gather indices must be int32/int64. Packing 6 indices as float32 channels 15..20 would (a) force a float->int cast inside the graph (lossy past 2^24, fine for N<1.3M but adds an op and a silent-precision footgun) and (b) collide semantically with the masked mean/max pool, which would average raw neighbor indices into the tile embedding — garbage. Keeping them a separate int64 tensor keeps the existing float pool untouched.
2. **Ragged-N axis is shared, not the payload.** `spatial`, `spatial_adj`, `spatial_adj_mask` all share the same N (tile count) and the same dynamic axis `n_spatial`. That is exactly what `export_onnx.py`'s ragged `{1:"n_<name>"}` machinery already expresses — adding two tensors that reuse the `n_spatial` symbolic dim is zero new export risk (research-notes Q3). Folding into spatial would instead require N to be both the token axis AND embed two different dtypes in one tensor — impossible at opset 17 without a Split+Cast dance.
3. **Mask semantics differ.** `spatial`'s existing presence mask is `[B, N]` (is-this-a-real-tile). The neighbor mask is `[B, N, 6]` (does-this-tile-have-a-neighbor-in-direction-d, after world-wrap). Different rank, different meaning; cannot be the spatial token mask.
4. **Parity + contract clarity.** A distinct name makes the OnnxPolicy input-inventory check, the export `names` list, and the parity fixture self-documenting; the GNN consumer in model.py reads `inputs["spatial_adj"]`/`inputs["spatial_adj_mask"]` by name. This is the v4 StructuredPolicyValueNet's only new contract surface.

Consequence: `RICH_TOKEN_NAMES` stays the 6 existing names (these are pooled token SETS with a `_mask` partner each — the adj tensors are NOT token sets and must NOT get the automatic `+"_mask"` pairing in OnnxPolicy's loop). The two adj tensors are registered as a SEPARATE inventory list. See D1.5 + D3-inference below.

Degree axis = 6 fixed (HexMath.clockPositionToHexcoordMap has exactly 6 directions, HexMath.kt:277-285). Direction order is FROZEN as the clock order `[12/0,2,4,6,8,10] = [(+1,+1),(0,+1),(-1,0),(-1,-1),(0,-1),(+1,0)]` and must be identical in the Python builder, the JVM inference builder, and the adjacency-parity test.

---

### D1.1 — buildSpatial: append tile_x/tile_y (13->15), outside fog gates

`Featurizer.kt:26` `channels` derives from `SampleSchema.NUM_SPATIAL_CHANNELS` (auto-widens once SPATIAL_CHANNELS grows — no edit needed here, it follows D1.5).

`buildSpatial` (Featurizer.kt:233-262): the per-tile loop early-`continue`s on `!explored` (line 243) and `!visible` (line 250). Position is static/always-known, so write x/y BEFORE the first gate, right after computing `base`:

```kotlin
val base = tile.zeroBasedIndex * channels
if (base < 0 || base + channels > out.size) continue
// D1.1: plain float32 coords, written for EVERY tile (static, fog-independent),
// BEFORE the explored/visible gates. Channels 13,14. Negative-safe, no +1/coerce.
out[base + 13] = tile.position.x
out[base + 14] = tile.position.y
val visible = omni || x.viewableTiles.contains(tile)
...
```
(`tile.position.x`/`.y` are Float already — HexCoord; no `.toFloat()` needed but harmless. They are NOT clamped: `coerceAtMost(255)` only bounds categorical vocab indices, codebase-scan §Resolved.) The Python adjacency builder reads exactly ch13/ch14 of each tile row to recover (x,y) and build the coords->zeroBasedIndex map.

---

### D1.2 — buildGlobal: append radius, worldWrap, shape ordinal (head 5->8, global 26->29)

Current head = 5 floats; global width = 5 + DEMOGRAPHIC_CATEGORIES(7)*3 = 26. After: head = 8, width = 29.

`shape` is a String constant (MapParameters.kt:10-15: rectangular/hexagonal/flatEarth), NOT an enum — must map to a stable ordinal explicitly. Use the canonical list order so it never drifts.

```kotlin
private fun buildGlobal(x: Civilization, demoCtx: DemographicsContext): FloatArray {
    val agg = FloatArray(demographics.size * 3)
    demographics.forEachIndexed { i, cat ->
        val s = demoCtx.perCategory.getValue(cat)
        agg[i * 3] = s.best; agg[i * 3 + 1] = s.avg; agg[i * 3 + 2] = s.worst
    }
    val mp = gameInfo.tileMap.mapParameters
    val shapeOrdinal = when (mp.shape) {
        MapShape.rectangular -> 0f
        MapShape.hexagonal   -> 1f
        MapShape.flatEarth   -> 2f
        else -> 1f   // unknown/custom -> hexagonal default (matches MapParameters default)
    }
    val head = floatArrayOf(
        gameInfo.turns.toFloat(), x.tech.era.eraNumber.toFloat(),
        gameInfo.tileMap.tileList.size.toFloat(),
        x.getKnownCivs().count { it.isMajorCiv() }.toFloat(),
        gameInfo.civilizations.count { it.isMajorCiv() && it.isAlive() }.toFloat(),
        // D1.2: map dims so Python can replicate getIfTileExistsOrNull (worldWrap retry).
        mp.mapSize.radius.toFloat(),
        if (mp.worldWrap) 1f else 0f,
        shapeOrdinal,
    )
    return head + agg
}
```
Add import `com.unciv.logic.map.MapShape` to Featurizer.kt (it currently imports only Civilization/GameInfo/etc.; MapShape is `com.unciv.logic.map.MapShape`).

CRITICAL replication note for the Python builder (D3-train, not this cluster but the contract it consumes): the world-wrap radius in `getIfTileExistsOrNull` (TileMap.kt:383-385) is `mapSize.radius` for hexagonal/flatEarth but `mapSize.width/2` for rectangular. radius alone is therefore NOT sufficient for rectangular maps. To keep the Python builder a pure function of emitted globals, I emit `radius` = the ALREADY-RESOLVED wrap radius (not the raw mapSize.radius) when worldWrap is on. Refine D1.2 to compute the effective wrap radius so Python never needs mapSize.width:

```kotlin
val wrapRadius = if (mp.shape == MapShape.rectangular) mp.mapSize.width / 2 else mp.mapSize.radius
... mp.mapSize.radius is replaced in the head by wrapRadius.toFloat() ...
```
Emit the channel as `effective_wrap_radius` (the exact integer the Kotlin world-wrap code uses). This removes the rectangular ambiguity and is the single number Python adds/subtracts. Document this in the global layout doc string. (Open question O1 flags whether to ALSO emit raw radius for the encoder's own use — see open_questions.)

---

### D1.3 — entity tile-index (unit 8->9, city 16->17)

`unitTokenWidth` (Featurizer.kt:30) `8 -> 9`; `cityTokenWidth` (Featurizer.kt:29) `16 -> 17`.

writeUnitToken (Featurizer.kt:210-223) — append as the 9th field:
```kotlin
w.put(u.promotions.getAvailablePromotions().count())
w.put(u.currentTile.zeroBasedIndex)   // D1.3: tile index (units already sorted by this)
```
writeCityToken (Featurizer.kt:185-208) — city carries no position today; append after the construction block. Because the construction fields are only written under `if (isOwn || hasSpy)`, the tile index must be written UNCONDITIONALLY (every city has a center tile and the token width is fixed at 17). Put it FIRST-after-the-conditional is wrong (slot drift when !isOwn && !hasSpy leaves slots 13,14 zero). Use TokenSlice's running cursor: the safe pattern is to write it at a FIXED slot regardless of branch. The cleanest: write tile index BEFORE the conditional construction block so its slot (index 12) is deterministic, and the construction fields become slots 13,14:

```kotlin
w.put(if (hasSpy) 1f else 0f)                 // slot 11
w.put(city.getCenterTile().zeroBasedIndex)    // slot 12 (D1.3, UNCONDITIONAL)
if (isOwn || hasSpy) {
    val cur = city.cityConstructions.currentConstructionName()
    w.put(... fixed namespace ...)            // slot 13
    w.put(city.cityConstructions.getBuiltBuildings().count())  // slot 14
}
```
Wait: current code writes hasSpy at slot 11, then construction at 12/13 — 14 slots used of 16 (slots 14,15 unused padding). Moving tile-index to slot 12 pushes construction to 13/14, total 15 of 17 (slots 15,16 padding). The TokenSlice cursor auto-advances, so no manual index math; just insert the `w.put(zeroBasedIndex)` line between the hasSpy put and the `if`. NOTE the perItem schema width (17) is what the loader reshapes by — slots beyond what's written stay 0 (FloatArray default), matching today's behavior for the 2 trailing pads.

Update the perItem emission: it is automatic — `varF32("own_units", unitTokenWidth, ownUnits)` and the city equivalents read the (now-9/17) field; the FloatArray allocations at Featurizer.kt:62/71/90/97 use `* cityTokenWidth`/`* unitTokenWidth` and auto-resize. No other Kotlin edit. The schema.json perItem is emitted from these widths (the Observation.Block.perItem), so contract.py picks them up via token_specs_from_schema with no fallback.

---

### D1.4 — fix construction-namespace collision (Featurizer.kt:205)

Current (BUG): `w.put((vocab.building(cur).takeIf { it >= 0 } ?: vocab.unit(cur)) + 1)`.
When `cur` is a building#k -> emits k+1. When it's a unit#k -> emits k+1 TOO (collision: building#3 and unit#3 both -> 4). Vocab indexers return -1 on miss (codebase-scan-light). Fix puts units into a disjoint range above all buildings:

```kotlin
val cur = city.cityConstructions.currentConstructionName()
w.put(
    vocab.building(cur).takeIf { it >= 0 }?.plus(1)
        ?: (vocab.unit(cur).takeIf { it >= 0 }?.plus(vocab.buildingCount + 1) ?: 0)
)
```
Namespace: 0 = none/unknown; building#k -> k+1 (1..buildingCount); unit#k -> buildingCount+1+k. No overlap. `vocab.buildingCount` exists (codebase-scan §Vocab counts). AC7 unit test lands beside OnnxPolicyLegalityTest. D2's single `buildingCount+unitCount+1` construction embedding depends on this fix.

---

### D1.5 — schema/contract bumps + SPATIAL_CHANNELS

SampleSchema.kt edits:
- `VERSION = 2` -> `3` (line 22). Update the doc comment: VERSION 3 adds tile_x/tile_y spatial channels, map-dim globals, per-entity tile index, and the structured (GNN) contract.
- `OnnxContract.CONTRACT_VERSION_RICH = 2` -> `3` (line 42). Update doc: "Contract v3 = structured rich input: contract-v2 multi-tensor + spatial neighbor graph (spatial_adj/spatial_adj_mask) for the GNN encoder."
- `SPATIAL_CHANNELS` (lines 86-100): append two entries (auto-bumps NUM_SPATIAL_CHANNELS to 15):
```kotlin
    "unit_health_bucket", // ...
    "tile_x",             // PERSISTENT (static): hex coord x as plain float (negative-safe)
    "tile_y",             // PERSISTENT (static): hex coord y as plain float
)
```
- `RICH_TOKEN_NAMES` stays the same 6 names. Add NEW constants for the graph tensors so both OnnxPolicy and the parity harness reference one SSOT:
```kotlin
    const val INPUT_SPATIAL_ADJ = "spatial_adj"
    const val INPUT_SPATIAL_ADJ_MASK = "spatial_adj_mask"
    /** Degree of the fixed hex neighbor axis (6). FROZEN clock order in HexMath. */
    const val SPATIAL_DEGREE = 6
```
- RulesetFingerprint coupling kept: `Vocab.canonicalSections` already hashes `"schema:spatialChannels" to SampleSchema.SPATIAL_CHANNELS` (Vocab.kt:97). Appending tile_x/tile_y AUTO-changes the fingerprint -> old shards/models refused (fail-loud, intended). Bumping VERSION ALSO changes the fingerprint (RulesetFingerprint hashes SampleSchema.VERSION). Both move together; no extra coupling edit needed. Do NOT add the adj tensor names to canonicalSections (they are derived, not emitted into shards — the fingerprint should track emitted layout only; the contract version gates the structured path).

contract.py edits (lockstep mirror):
- `CONTRACT_VERSION_RICH = 2` -> `3` (line 18).
- `RICH_TOKEN_NAMES` (line 29): unchanged.
- Add `INPUT_SPATIAL_ADJ = "spatial_adj"`, `INPUT_SPATIAL_ADJ_MASK = "spatial_adj_mask"`, `SPATIAL_DEGREE = 6` constants.
- `_TOKEN_WIDTH_FALLBACK` (lines 80-82): update spatial 13->15 AND make `token_specs_from_schema` FAIL-LOUD instead of silently falling back (council FND-0007/0011). Replace the silent `.get`/fallback with an assert that the schema actually carries the field, and that the spatial channel count matches the expected 15 for a v3 schema:
```python
_TOKEN_WIDTH_FALLBACK = {"spatial": 15, "own_units": 9, "opp_units": 9,
                         "own_cities": 17, "opp_cities": 17, "civ_tokens": 84}

def token_specs_from_schema(schema_path):
    sch = _schema(schema_path)
    layout = {b["name"]: b for b in sch.get("layout", [])}
    chans = sch.get("spatial_channels") or sch.get("spatialChannels")
    assert chans, f"schema {schema_path} missing spatial_channels (fail-loud, no silent fallback)"
    specs = {"spatial": len(chans)}
    for name in ("own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens"):
        entry = layout.get(name)
        assert entry and entry.get("perItem"), f"schema missing perItem for {name} (fail-loud)"
        specs[name] = int(entry["perItem"])
    return specs
```
The `_TOKEN_WIDTH_FALLBACK` constant is kept ONLY as a documented expectation / for the parity-dim consistency assert (cross-check schema-derived widths against it and raise on drift), not as a silent runtime path.

---

### D3-inference — OnnxPolicy builds spatial_adj + spatial_adj_mask from the live TileMap

Key advantage at inference: the JVM has the REAL TileMap, so it uses `getIfTileExistsOrNull` / `tile.neighbors` directly — NO replication of the world-wrap math (only the Python TRAIN builder replicates it; the adjacency-parity test guards Python-vs-Kotlin). This is the asymmetry the constraint demands.

The contract gate at OnnxPolicy.kt:59-63 must accept 3. The cleanest: introduce a `structured` flag for contract-v3 and keep `rich` true for {2,3} (v3 is a superset of v2 inputs + the two adj tensors):
```kotlin
val structured = mContract == SampleSchema.OnnxContract.CONTRACT_VERSION_RICH  // ==3 now
rich = structured || mContract == 2   // keep the v2 multi-tensor build path for legacy
check(mContract == SampleSchema.OnnxContract.CONTRACT_VERSION || mContract == 2 || structured) { ... {1,2,3} ... }
```
(Decision: accept {1,2,3} so already-shipped v2 rich-critic models keep loading. The v3 model additionally requires the two adj inputs; gate on `structured` for those.)

Rich-input inventory check (OnnxPolicy.kt:66-74): when `structured`, ALSO require the two graph tensors (and do NOT append `_mask` to them in the existing loop):
```kotlin
if (rich) {
    val want = mutableListOf(INPUT_GLOBAL, INPUT_ACTING)
    for (n in RICH_TOKEN_NAMES) { want += n; want += n + MASK_SUFFIX }
    if (structured) { want += INPUT_SPATIAL_ADJ; want += INPUT_SPATIAL_ADJ_MASK }
    val have = session.inputNames
    val missing = want.filter { it !in have }
    check(missing.isEmpty()) { "OnnxPolicy: model missing expected inputs $missing (model has $have)" }
}
```

forwardRich (OnnxPolicy.kt:127-138): when `structured`, build and add the two adj tensors to `inputs` (built from the SAME Observation whose spatial block defines tile order == zeroBasedIndex). They must be closed in the existing `finally` (they live in the same `inputs` map, so the existing `for (t in inputs.values) t.close()` already covers them — no new close site, satisfies "all tensors closed in finally"):
```kotlin
private fun forwardRich(obs: Observation): Pair<FloatArray, FloatArray> {
    val inputs = buildRichTensors(env, obs)
    try {
        if (structured) addSpatialGraphTensors(env, inputs, /* tileMap */ memoTileMap)
        session.run(inputs).use { res -> ... }
    } finally { for (t in inputs.values) try { t.close() } catch (_: Exception) {} }
}
```
PROBLEM: `buildRichTensors` is a companion (static, no civ context); the TileMap comes from the civ being decided. Pass it in. `forwardRich` is called from `logitsFor` which has `civ` -> thread `civ.gameInfo.tileMap` through. Cleanest: make `addSpatialGraphTensors` an instance method (needs `structured`) taking the TileMap:

```kotlin
/** Build spatial_adj [1,N,6] int64 + spatial_adj_mask [1,N,6] f32 from the LIVE TileMap.
 *  N = tileList.size (== spatial token count). Row i corresponds to tile.zeroBasedIndex==i.
 *  Direction order is the FROZEN clock order; missing/off-map neighbor -> index 0, mask 0
 *  (index 0 is a valid in-bounds Gather target; the mask zeroes its contribution — mirrors the
 *  masked_pool discipline). World-wrap is handled by getIfTileExistsOrNull (REAL, no replication). */
private fun addSpatialGraphTensors(env: OrtEnvironment, inputs: LinkedHashMap<String, OnnxTensor>, tileMap: TileMap) {
    val tiles = tileMap.tileList
    val n = tiles.size
    val deg = SampleSchema.OnnxContract.SPATIAL_DEGREE   // 6
    val idx = LongArray(n * deg)
    val msk = FloatArray(n * deg)
    // FROZEN clock order (HexMath.clockPositionToHexcoordMap): (+1,+1)(0,+1)(-1,0)(-1,-1)(0,-1)(+1,0)
    val dirs = arrayOf(intArrayOf(1,1), intArrayOf(0,1), intArrayOf(-1,0), intArrayOf(-1,-1), intArrayOf(0,-1), intArrayOf(1,0))
    for (tile in tiles) {
        val i = tile.zeroBasedIndex
        val px = tile.position.x.toInt(); val py = tile.position.y.toInt()
        for (d in 0 until deg) {
            val nb = tileMap.getIfTileExistsOrNull(px + dirs[d][0], py + dirs[d][1])
            if (nb != null) { idx[i * deg + d] = nb.zeroBasedIndex.toLong(); msk[i * deg + d] = 1f }
            // else leave idx=0L (in-bounds), msk=0f
        }
    }
    inputs[SampleSchema.OnnxContract.INPUT_SPATIAL_ADJ] =
        OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(idx), longArrayOf(1, n.toLong(), deg.toLong()))
    inputs[SampleSchema.OnnxContract.INPUT_SPATIAL_ADJ_MASK] =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(msk), longArrayOf(1, n.toLong(), deg.toLong()))
}
```
Notes:
- int64 (LongBuffer) chosen because ORT Gather indices are commonly int64 and the Python export will emit int64 indices (torch long). Must match the dtype the export declares for spatial_adj (lockstep: export_onnx dummy + dynamic_axes).
- Index 0 + mask 0 for off-map: index must be a VALID in-bounds value so ORT Gather never reads OOB (council FND-0025); mask 0 zeroes it. N>=1 always (a game has tiles), so 0 is always in bounds.
- Self-edge: NOT added here (the 6 dirs are strict neighbors). The GNN's own-node skip/residual is the model's job (D3), out of this cluster.
- `tile.neighbors` exists but returns a Sequence in clock-discovery order; using explicit `getIfTileExistsOrNull(px+dx,py+dy)` per FROZEN direction guarantees the same direction->column mapping as the Python builder (parity-critical). Do NOT use `tile.neighbors` for ordering.
- Add imports: `com.unciv.logic.map.TileMap`, `java.nio.LongBuffer`.

The blind `infer(input)` path and `forward` (contract v1) are untouched. `buildRichTensors`/`richTensorsFromArrays`/`tokenTensors` are untouched (the adj tensors are added in forwardRich, not in the companion token builder) — this keeps the parity harness's `richTensorsFromArrays` reusable; the parity harness adds the adj tensors via the same `addSpatialGraphTensors` logic (parityRunRich must construct a TileMap-equivalent or read precomputed adj — see D10/open questions).

---

### Lockstep ledger (every width/version site that moves together)
See `lockstep_sites`. Spatial 13->15: 4 sites. unit 8->9 / city 16->17: 3 sites. VERSION 2->3 + CONTRACT_VERSION_RICH 2->3: 4 sites. Global 26->29: derived, no hardcode but parity fixture + Dims read it from schema. Two new tensors: 5 sites (Kotlin const, contract.py const, OnnxPolicy build+inventory, export, parity).

## Exact edits
- **core/src/com/unciv/logic/simulation/dataplane/SampleSchema.kt** [line 22]: const val VERSION = 2 -> const val VERSION = 3; update doc comment to describe v3 (tile_x/tile_y channels, map-dim globals, per-entity tile index, structured GNN contract).
  _why:_ D1.5 schema layout bump; Python reader refuses mismatch; also changes RulesetFingerprint (perishable shards regenerate).
- **core/src/com/unciv/logic/simulation/dataplane/SampleSchema.kt** [line 42 (OnnxContract.CONTRACT_VERSION_RICH)]: = 2 -> = 3; doc: v3 = v2 multi-tensor + spatial neighbor graph for GNN.
  _why:_ D9 contract bump in lockstep with contract.py:18.
- **core/src/com/unciv/logic/simulation/dataplane/SampleSchema.kt** [after line 49 (MASK_SUFFIX), inside OnnxContract]: add const val INPUT_SPATIAL_ADJ="spatial_adj"; const val INPUT_SPATIAL_ADJ_MASK="spatial_adj_mask"; const val SPATIAL_DEGREE=6
  _why:_ SSOT for the two new graph input names + the fixed hex degree, referenced by OnnxPolicy, export, parity.
- **core/src/com/unciv/logic/simulation/dataplane/SampleSchema.kt** [lines 99-100 (end of SPATIAL_CHANNELS list)]: append "tile_x", "tile_y" entries (13->15). NUM_SPATIAL_CHANNELS auto-follows (.size).
  _why:_ D1.1/D1.5; also flows into Vocab fingerprint section automatically.
- **core/src/com/unciv/logic/simulation/dataplane/Featurizer.kt** [buildSpatial, after line 239 (base bounds check), before line 240]: out[base+13]=tile.position.x; out[base+14]=tile.position.y  (BEFORE the explored/visible gates, plain float, no coerce).
  _why:_ D1.1 static coords for every tile; Python reads ch13/14 to build adjacency.
- **core/src/com/unciv/logic/simulation/dataplane/Featurizer.kt** [buildGlobal, lines 159-164 (head floatArrayOf)]: append effectiveWrapRadius (=width/2 for rectangular else mapSize.radius), worldWrap bit, shapeOrdinal (rectangular=0,hexagonal=1,flatEarth=2,else 1). head 5->8.
  _why:_ D1.2 map dims Python needs to replicate getIfTileExistsOrNull world-wrap exactly.
- **core/src/com/unciv/logic/simulation/dataplane/Featurizer.kt** [top imports]: add import com.unciv.logic.map.MapShape
  _why:_ D1.2 references MapShape.rectangular/hexagonal/flatEarth constants.
- **core/src/com/unciv/logic/simulation/dataplane/Featurizer.kt** [line 30]: private val unitTokenWidth = 8 -> 9
  _why:_ D1.3 unit token gains zeroBasedIndex; auto-resizes ownUnits/oppUnits FloatArrays + schema perItem.
- **core/src/com/unciv/logic/simulation/dataplane/Featurizer.kt** [line 29]: private val cityTokenWidth = 16 -> 17
  _why:_ D1.3 city token gains center-tile zeroBasedIndex.
- **core/src/com/unciv/logic/simulation/dataplane/Featurizer.kt** [writeUnitToken, after line 222 (last w.put)]: w.put(u.currentTile.zeroBasedIndex)
  _why:_ D1.3 emit unit tile index (units already sorted by it).
- **core/src/com/unciv/logic/simulation/dataplane/Featurizer.kt** [writeCityToken, between line 202 (hasSpy put) and line 203 (if isOwn||hasSpy)]: w.put(city.getCenterTile().zeroBasedIndex)  // UNCONDITIONAL, fixed slot 12 before the conditional construction block
  _why:_ D1.3 city tile index must be at a deterministic slot regardless of isOwn/hasSpy branch.
- **core/src/com/unciv/logic/simulation/dataplane/Featurizer.kt** [line 205 (writeCityToken construction put)]: w.put(vocab.building(cur).takeIf{it>=0}?.plus(1) ?: (vocab.unit(cur).takeIf{it>=0}?.plus(vocab.buildingCount+1) ?: 0))
  _why:_ D1.4 fix construction-namespace collision (building#k vs unit#k); disjoint ranges.
- **python/unciv_train/contract.py** [line 18]: CONTRACT_VERSION_RICH = 2 -> 3
  _why:_ D9 lockstep mirror of SampleSchema.OnnxContract.CONTRACT_VERSION_RICH.
- **python/unciv_train/contract.py** [after line 29 (RICH_TOKEN_NAMES)]: add INPUT_SPATIAL_ADJ="spatial_adj", INPUT_SPATIAL_ADJ_MASK="spatial_adj_mask", SPATIAL_DEGREE=6
  _why:_ Python SSOT for the two new graph tensors consumed by StructuredPolicyValueNet + export.
- **python/unciv_train/contract.py** [lines 80-82 (_TOKEN_WIDTH_FALLBACK)]: spatial 13->15, own/opp_units 8->9, own/opp_cities 16->17 (keep civ_tokens 84). Retain only as a documented expectation for the drift-assert, not a silent runtime path.
  _why:_ Lockstep widths; council FND-0007.
- **python/unciv_train/contract.py** [token_specs_from_schema lines 85-102]: replace silent fallback with fail-loud asserts: require spatial_channels present and each entity perItem present; raise on absence.
  _why:_ council FND-0007/0011 fail-loud vs schema; a schema-emit regression must alarm, not silently revert to hardcoded widths.
- **desktop/src/com/unciv/app/desktop/OnnxPolicy.kt** [init, lines 59-63]: add `val structured = mContract == CONTRACT_VERSION_RICH(==3)`; set `rich = structured || mContract == 2`; gate accepts {1,2,3}.
  _why:_ D3-inference contract gate must load v3 and keep v2 legacy models.
- **desktop/src/com/unciv/app/desktop/OnnxPolicy.kt** [rich-input inventory check, lines 66-74]: when structured, also require INPUT_SPATIAL_ADJ + INPUT_SPATIAL_ADJ_MASK (NOT via the _mask loop); single check.
  _why:_ provenance: a v3 model must expose the graph inputs.
- **desktop/src/com/unciv/app/desktop/OnnxPolicy.kt** [forwardRich lines 127-138 + new instance method]: thread civ.gameInfo.tileMap into forwardRich; when structured call addSpatialGraphTensors(env, inputs, tileMap) which builds spatial_adj [1,N,6] int64 + spatial_adj_mask [1,N,6] f32 from getIfTileExistsOrNull over the FROZEN 6 clock dirs; tensors added into `inputs` so the existing finally closes them.
  _why:_ D3-inference: JVM builds the graph from the REAL TileMap, no world-wrap replication; OOB-safe (idx 0 + mask 0); all tensors closed in finally.
- **desktop/src/com/unciv/app/desktop/OnnxPolicy.kt** [logitsFor line 107]: pass obs together with civ.gameInfo.tileMap to forwardRich (signature change forwardRich(obs, tileMap)).
  _why:_ forwardRich needs the live TileMap for the graph build.
- **desktop/src/com/unciv/app/desktop/OnnxPolicy.kt** [imports + companion FALLBACK_WIDTH line 150-151]: add import com.unciv.logic.map.TileMap, java.nio.LongBuffer; update FALLBACK_WIDTH widths (own/opp_units 9, own/opp_cities 17) to stay in lockstep with the schema; keep as defensive-only.
  _why:_ lockstep width drift guard; council notes fallbacks must track schema.

## New inputs/tensors
- spatial_adj : int64 [1, N, 6] (N = tile count, dynamic axis n_spatial; degree axis 6 fixed). Per-tile neighbor zeroBasedIndex in FROZEN clock order; off-map/missing neighbor -> 0 (in-bounds dummy). Consumed by the GNN's Gather. NEW because neighbor indices are integer Gather indices that cannot live in the float32 spatial token tensor without a lossy cast and would poison the masked mean/max pool; a separate int64 tensor keeps the existing pool path byte-identical and reuses the existing ragged-N export machinery.
- spatial_adj_mask : float32 [1, N, 6] (same shape/axes as spatial_adj). 1 = real neighbor in that direction, 0 = off-map/world-wrap-missing. NEW because the neighbor mask is rank-3 [B,N,6] (per-direction) whereas the spatial token presence mask is rank-2 [B,N]; different semantics and rank, cannot be folded. The GNN multiplies messages by this mask before ReduceSum/ReduceMean over the degree axis (mirrors masked_pool NaN discipline).

## Lockstep sites
- Spatial channel count 13->15: SampleSchema.SPATIAL_CHANNELS (SampleSchema.kt:86-100, append tile_x/tile_y); Featurizer.channels (auto via NUM_SPATIAL_CHANNELS, :26); contract.py _TOKEN_WIDTH_FALLBACK['spatial'] (:81); dataset.py _rich_step_blocks reshape(-1,13) (:113) MUST read channel count from schema not literal; OnnxPolicy buildRichTensors spatial width uses NUM_SPATIAL_CHANNELS (auto, :157); parity-test fixture spatial:N dim (test_parity.py).
- unitTokenWidth 8->9: Featurizer.kt:30; contract.py _TOKEN_WIDTH_FALLBACK own/opp_units (:81); OnnxPolicy FALLBACK_WIDTH own/opp_units (:150); schema perItem auto-emitted; parity dims.
- cityTokenWidth 16->17: Featurizer.kt:29; contract.py _TOKEN_WIDTH_FALLBACK own/opp_cities (:82); OnnxPolicy FALLBACK_WIDTH own/opp_cities (:151); schema perItem auto; parity dims.
- Global width 26->29 (head 5->8): Featurizer.buildGlobal (:159-164) is the SSOT; Dims.global_w read from schema layout['global']['len'] (contract.py:65) — no hardcode but parity fixture global vector length + export dummy global shape must match.
- VERSION 2->3: SampleSchema.kt:22; RulesetFingerprint hashes it (auto fingerprint change); Python reader refusal; schema.json schemaVersion; OnnxPolicy expectedSchemaVersion check (:58).
- CONTRACT_VERSION_RICH 2->3: SampleSchema.kt:42; contract.py:18; OnnxPolicy gate (:59-63); export_onnx metadata stamp; parity test contract assertion.
- New tensors spatial_adj/spatial_adj_mask + SPATIAL_DEGREE=6: SampleSchema.OnnxContract consts; contract.py consts; OnnxPolicy addSpatialGraphTensors + inventory check; export_onnx names/dummies/dynamic_axes; model.py StructuredPolicyValueNet input reads; test_parity.py fixture + adjacency-parity test; FROZEN clock direction order shared by JVM builder + Python builder + parity reference.

## Export safety
All new ops are core ONNX <= opset 13, so opset-17 export is safe (research-notes Q1/Q3). The GNN consuming spatial_adj uses Gather (indices int64, data [B,N,15] -> [B,N,6,15]), Mul by spatial_adj_mask, ReduceSum/ReduceMean over the fixed degree axis (4) — NO scatter_add/scatter_reduce/index_add anywhere (the broken/silently-wrong opset-17 paths, pytorch#111159/#65138). Degree axis is a fixed constant 6, not dynamic, so the reduce axis is static and exporter-friendly. spatial_adj/spatial_adj_mask get the SAME dynamic-axes treatment the existing v2 token sets already use: batch {0} + ragged {1:\"n_spatial\"} shared with the spatial tensor (they MUST share the exact symbolic name so ORT ties the three N dims). Index safety: the JVM builder and the Python builder both clamp off-map neighbors to index 0 (always in-bounds since N>=1) with mask 0, so ORT Gather can never read OOB (council FND-0025) — Python must additionally assert 0<=idx<N before export per the spec gate. int64 dtype for spatial_adj matches torch long indices on export; the JVM LongBuffer tensor matches. Per the constraint: validate the export on the SMALL rung FIRST (export smoke test) before any scaling — gate before scaling (FND-0008/0023). Hand-rolled attention (D4/D5) is out of this cluster but the same no-MultiheadAttention/no-SDPA discipline applies downstream.

## Determinism/provenance
Determinism: spatial_adj/spatial_adj_mask are a pure function of (tile coords, effective wrap radius, worldWrap, shape) — all emitted into the shard (ch13/14 + global) — so the Python train-time builder is deterministic and reproducible from the shard alone (no edge list stored). At inference the JVM builds them from the live TileMap via getIfTileExistsOrNull, which is the authoritative source; the adjacency-parity test (D10) pins Python==Kotlin including world-wrap and ragged-edge tiles. The FROZEN clock direction order (HexMath.clockPositionToHexcoordMap, 6 dirs) is the shared invariant; encoding it as an explicit array on both sides (not tile.neighbors discovery order) removes nondeterminism. Provenance: VERSION 2->3 and the appended SPATIAL_CHANNELS both flow into RulesetFingerprint.compute (hashes SampleSchema.VERSION + canonicalSections incl. schema:spatialChannels, Vocab.kt:97), so any v2 shard/model is fail-loud-refused by OnnxPolicy's schema_version + ruleset_fingerprint checks (OnnxPolicy.kt:58,63) — datasets perishable by design. The adj tensor names are intentionally NOT added to canonicalSections (they are derived, not emitted), so the fingerprint tracks emitted layout only; the structured path is gated by contract_version=3, keeping the fingerprint's meaning crisp.

## Open questions
- O1: buildGlobal emits the EFFECTIVE wrap radius (width/2 for rectangular) for the Python builder. Should we ALSO emit the raw mapSize.radius as a separate global channel for the encoder's own use (e.g. map-scale feature), making head 5->9? Current design folds them into one channel to keep width minimal; confirm the encoder doesn't need both.
- O2: Self-edge handling. The 6-neighbor graph excludes the node itself. Does the D3 GNN add a self-loop (degree 7) or rely on a residual/own-node-concat? If self-loop, the degree axis becomes 7 and SPATIAL_DEGREE + both builders must change in lockstep. Needs the C-model cluster to confirm before freezing SPATIAL_DEGREE=6.
- O3: Parity-harness adj construction. parityRunRich (SelfPlayRunner.kt:351-396) currently builds tensors from a fixture Observation via richTensorsFromArrays. For v3 it must also produce spatial_adj/mask. Does the fixture carry a real (mini) TileMap to call getIfTileExistsOrNull, or does the parity test pass precomputed adj arrays in? This C1 design assumes a shared direction-array helper usable without a full TileMap; the exact D10 wiring (and whether to factor addSpatialGraphTensors to accept either a TileMap or precomputed coords+wrapRadius) is unresolved.
- O4: spatial_adj_mask dtype — float32 (chosen, matches masked_pool Mul) vs bool/int. float32 is simplest for the GNN Mul; confirm the GNN multiplies before reduce (so 0 contributions vanish) rather than using it as a softmax additive mask (which would need -inf, a different encoding).
- O5: Should the two adj tensors be added to META_INPUT_NAMES (the comma-joined ordered input list stamped in ONNX metadata)? Yes for completeness/self-description, but confirm the export_onnx names ordering places them deterministically (after the six token sets + masks) so the JVM inventory check ordering-agnostic membership test still holds.

## Risks
- Direction-order drift between the JVM inference builder, the Python train builder, and the adjacency-parity reference would silently mis-wire the GNN (wrong neighbor in each slot) without crashing. -> Define the 6 clock-order offsets as a single FROZEN constant array in one doc'd place per language, cite HexMath.clockPositionToHexcoordMap, and make the adjacency-parity test (D10) assert exact index-by-direction equality on a world-wrap map with ragged edges.
- Rectangular-map world-wrap radius is mapSize.width/2, not mapSize.radius (TileMap.kt:383-385) — emitting raw radius would make the Python builder wrong on rectangular maps. -> Emit the ALREADY-RESOLVED effective wrap radius (width/2 for rectangular, mapSize.radius otherwise) in buildGlobal so Python needs no shape-conditional and never reads mapSize.width.
- int64 dtype mismatch: if export declares spatial_adj as int32 but the JVM feeds int64 (or vice versa), ORT run fails at inference. -> Pin int64 on both sides (torch long on export, LongBuffer on JVM); add the dtype to the lockstep ledger and assert it in the parity test.
- dataset.py _rich_step_blocks hardcodes reshape(-1,13) (:113) — left unchanged it mis-decodes the 15-channel spatial block, corrupting every training row silently. -> This cluster flags it as a lockstep site; the C-Python cluster must change it to read the channel count from schema (FND-0007). Add a schema-vs-literal assert.
- city tile-index slot placement: if written inside the if(isOwn||hasSpy) block, !isOwn&&!hasSpy cities leave the index unwritten (0), making 'tile 0' look like every unknown city's location. -> Write zeroBasedIndex UNCONDITIONALLY before the conditional construction block (fixed slot 12), as specified in D1.3.
- OnnxPolicy companion buildRichTensors is static and has no TileMap, but the parity harness reuses it; adding adj there would break that reuse. -> Build adj in the instance forwardRich (not the companion), threading civ.gameInfo.tileMap; the parity harness constructs adj via the same shared direction-array logic against its fixture TileMap (resolve exact parity wiring in D10).
- Accepting contract {1,2,3} could let a stale v2 rich model load against a v3-fingerprint ruleset. -> The ruleset_fingerprint check (OnnxPolicy.kt:63) already changes with VERSION/SPATIAL_CHANNELS, so a v2 model built pre-bump fails the fingerprint gate regardless of the contract-version accept-set; the {1,2,3} accept-set only matters for models rebuilt against the same ruleset.

## VERDICT
```json
{
  "cluster": "C1 \u2014 Kotlin emit side + contract (D1, D9-Kotlin, D3-inference)",
  "export_safe": true,
  "lockstep_complete": false,
  "seam_preserved": true,
  "parity_feasible": false,
  "determinism_ok": true,
  "issues": [
    {
      "severity": "critical",
      "issue": "The spatial block is emitted as fixedU8(\"spatial\", ...) (Featurizer.kt:129, DT_U8). Observation.writeBlock (Observation.kt:69) serializes DT_U8 blocks as `b.u8(v.toInt().coerceIn(0,255))` \u2014 i.e. EVERY spatial channel is truncated to int and clamped to [0,255] at STORAGE time. The design's load-bearing premise (Decision 0, D1.1, codebase-scan \u00a7Resolved) that tile_x/tile_y ride as 'plain float32, negative-safe, no coerce' is FALSE for the spatial block: coords on any map wider than 255 in x or y collapse to 255, and there are no negative coords preserved (Unciv hex coords are signed and routinely negative). This silently destroys the (x,y) data the entire Python adjacency builder depends on \u2014 adjacency parity would be unreconstructable for real maps, defeating the GNN. The 'shards are float32 (resolved)' constraint is contradicted by the actual u8 storage path.",
      "fix": "Coords CANNOT live in the existing u8 spatial block. Pick one: (a) change spatial to fixedF32 (DT_F32) so all 15 channels store as float32 \u2014 but this changes the spatial block byte layout/size for ALL channels, bumps shard size ~4x for spatial, and must be reflected in dataset.py dtype read (already float32 in numpy) + parity + the leakage/byte-equality test; OR (b) emit a SEPARATE fixedF32 block (e.g. \"spatial_coords\", nTiles*2) carrying x,y as true float32 and read THAT in the Python adjacency builder, leaving the u8 spatial token set at 13 channels. Option (b) is cleaner and avoids re-typing the established u8 spatial plane. Either way the 13->15 'append to SPATIAL_CHANNELS' edit as written stores clamped garbage and must be redesigned. Re-verify against Observation.writeBlock dtype branch."
    },
    {
      "severity": "critical",
      "issue": "HexCoord is `data class HexCoord(val x: Int = 0, val y: Int = 0)` (HexMath.kt:391) \u2014 position.x/.y are Int, NOT Float. The design explicitly asserts 'tile.position.x/.y are Float already \u2014 HexCoord; no .toFloat() needed but harmless' (D1.1) and writes `out[base+13] = tile.position.x` into a FloatArray. Kotlin will NOT auto-widen Int->Float on array element assignment; this fails to compile. The asserted justification for omitting .toFloat() is factually wrong.",
      "fix": "Use `out[base+13] = tile.position.x.toFloat()` and `out[base+14] = tile.position.y.toFloat()`. Correct the design note: HexCoord coords are Int and .toFloat() is REQUIRED, not optional. (Same Int reality means the D3 inference builder's `tile.position.x.toInt()` is a redundant no-op, harmless.)"
    },
    {
      "severity": "major",
      "issue": "export_onnx.py has NO mechanism to add the two adj tensors as model inputs. export_rich (export_onnx.py:109-115) loops over token_specs and for EVERY entry auto-creates a `name+\"_mask\"` companion input plus dynamic_axes {0:batch,1:n_name}. `sample_inputs` (line 116-117) only OVERRIDES dummy values for already-declared names; it cannot add new input names, dynamic_axes, or positional args. So there is currently no code path to emit spatial_adj/spatial_adj_mask as inputs with int64 dtype and a shared n_spatial axis. The design's lockstep ledger lists 'export' as a site but exact_edits contains ZERO export_onnx.py edit. Without it the v3 ONNX simply won't have the inputs the JVM inventory check requires, and the int64 dtype (torch long) for spatial_adj must be explicitly created (torch.zeros default float) or the exported graph types the Gather indices as float.",
      "fix": "Add an explicit export_onnx.py edit: extend export_rich (or add export_structured) to append spatial_adj (dummy torch.zeros(1,n0,6,dtype=torch.long)) and spatial_adj_mask (torch.ones(1,n0,6)) to names/dummy/args, with dynamic_axes {0:'batch',1:'n_spatial'} on BOTH so the symbolic dim ties to the spatial tensor's n_spatial. Crucially the degree axis stays static (6). Validate export on the SMALL rung first per the constraint. This belongs IN this cluster's edit list because C1 owns the contract surface the export must mirror."
    },
    {
      "severity": "major",
      "issue": "Parity is infeasible for a v3 model as designed. parityRunRich (SelfPlayRunner.kt:351-396) only parses 'vec' and 'set' fixture lines and builds tensors via OnnxPolicy.richTensorsFromArrays, which has no knowledge of spatial_adj/spatial_adj_mask. A v3 ONNX declares those two inputs as REQUIRED; session.run will throw 'missing input' so the parity test cannot run at all. The design's O3 acknowledges this but leaves it unresolved, and the HARD CONSTRAINT mandates an adjacency-parity test guarding the Python-vs-Kotlin replication. Additionally addSpatialGraphTensors is designed as an instance method needing a live TileMap, but the parity harness has no TileMap.",
      "fix": "Resolve O3 concretely in-plan: add a new fixture line type (e.g. 'adj <N> <6> <ints...>' and 'adjmask ...') OR factor the neighbor-builder to accept precomputed coords+effectiveWrapRadius+worldWrap (no TileMap) so both parityRunRich and the Python builder call the SAME pure function. parityRunRich must inject the two adj tensors into the inputs map before session.run. The adjacency-parity test should drive a small synthetic world-wrap map with ragged edges and assert index-by-direction equality. Until this is specified, the mandated adjacency-parity guard does not exist."
    },
    {
      "severity": "major",
      "issue": "Spatial 13->15 lockstep is incomplete within this cluster's exact_edits. dataset.py line 113 hardcodes `reshape(-1, 13)` and RICH_TOKEN_BLOCKS docstring/dataset.py:36 says 'spatial is FIXED nTiles*13'. Left unchanged, every training row mis-decodes a 15-channel spatial block into 13-wide rows, silently corrupting all rich training. The design lists it as a risk and defers it to 'the C-Python cluster', but it is a direct consequence of the C1 width bump and the prompt requires the lockstep to move across dataset.py reshape. It is in the lockstep_sites list but absent from exact_edits, so a reader applying only C1 edits ships a broken pipeline.",
      "fix": "Either own the dataset.py:113 reshape edit here (read channel count from schema/token_specs, e.g. reshape(-1, token_specs['spatial'])) or make the cluster boundary explicit and add a hard cross-cluster dependency note that C1 MUST NOT merge without the dataset.py edit. Also bump the dataset.py:36 docstring 13->dynamic."
    },
    {
      "severity": "major",
      "issue": "Off-map neighbor world-wrap reconstruction in Python is harder than 'a single number Python adds/subtracts' (D1.2/O1). getIfTileExistsOrNull (TileMap.kt:383-395) does NOT add the effective radius to one coordinate; it tries getOrNull(x+radius, y-radius) THEN getOrNull(x-radius, y+radius) \u2014 a two-direction wrap on BOTH coords simultaneously, then a real lookup against the actual tile set (which is arbitrary, not a dense rectangle). Emitting only effective_wrap_radius + worldWrap + shape is NOT sufficient for Python to replicate this unless Python also has the full coord->index map (which it reconstructs from ch13/14 \u2014 but see the u8 clamping critical issue, which currently makes that map wrong). The design under-states the replication complexity and the per-coord wrap formula.",
      "fix": "Specify the Python builder to (1) build a dict {(x,y)->zeroBasedIndex} from the recovered coords, (2) for each tile+direction compute (nx,ny), look up directly; if miss and worldWrap, try (nx+R, ny-R) then (nx-R, ny+R) against the dict, mirroring TileMap.kt exactly; R = effective_wrap_radius from globals. The adjacency-parity test must cover wrapped edges. Note the coord recovery prerequisite is blocked by the u8 storage critical issue."
    },
    {
      "severity": "minor",
      "issue": "writeCityToken construction-collision fix (D1.4) uses vocab.buildingCount, confirmed to exist (Vocab.kt:52) \u2014 OK. But the design's writeCityToken slot-renumbering prose is self-contradictory (it first says 'slot 12', then 'Wait: ... pushes construction to 13/14'). The actual TokenSlice cursor auto-advances so any insertion order is positionally consistent, but the muddled prose risks a misimplementation that writes the index inside the if-block. The real requirement is simply: insert one unconditional w.put(city.getCenterTile().zeroBasedIndex.toFloat()) between the hasSpy put (line 202) and the `if (isOwn||hasSpy)` (line 203), and widen cityTokenWidth 16->17.",
      "fix": "Drop the slot-number narrative; state the edit as the single unconditional insert before the conditional block. Also note zeroBasedIndex is Int so .toFloat() (or TokenSlice.put(Int) overload, which exists at Featurizer.kt:267) handles it \u2014 use w.put(Int) overload for clarity. Same for the unit token: w.put(u.currentTile.zeroBasedIndex) uses the Int overload, fine."
    },
    {
      "severity": "minor",
      "issue": "Imports: design adds `com.unciv.logic.map.MapShape` to Featurizer.kt and references MapShape.rectangular/hexagonal/flatEarth. Confirm the enum/constants path: TileMap.kt:384 uses `mapParameters.shape == MapShape.rectangular`, so MapShape exists, but the design earlier calls shape 'a String constant (MapParameters.kt:10-15)' then uses MapShape enum members \u2014 inconsistent. If shape is actually a MapShape enum the `when` is fine; if it is a String the `when (mp.shape) { MapShape.rectangular -> }` won't compile. This was not verified against MapParameters.kt in the reviewed sources.",
      "fix": "Read MapParameters.kt to confirm whether `shape` is a String or MapShape enum before freezing the buildGlobal `when`. TileMap.kt:384 comparing to MapShape.rectangular implies enum (or a typed constant); align the buildGlobal code and the 'String constant' claim. Low risk but must be pinned to avoid a compile break."
    },
    {
      "severity": "minor",
      "issue": "META_INPUT_NAMES (O5): export stamps input_names as comma-joined names order. The JVM inventory check (OnnxPolicy.kt:71-73) is a membership test (filter !in have), order-agnostic, so ordering is not load-bearing for loading. But if the two adj tensors are appended to `names` in export, they WILL appear in META_INPUT_NAMES automatically \u2014 fine. No action needed beyond ensuring export appends them; flagging that O5 is a non-issue for the membership check.",
      "fix": "No change required; the inventory check is order-agnostic. Just ensure export appends adj names so META_INPUT_NAMES stays self-describing."
    }
  ],
  "verdict": "REVISE",
  "corrected_notes": "Two CRITICAL corrections invalidate the design's central premise and must be fixed before planning: (1) The spatial block is DT_U8 (Featurizer.kt:129) and Observation.writeBlock (Observation.kt:69) stores u8 blocks as `v.toInt().coerceIn(0,255)` \u2014 so tile_x/tile_y appended to SPATIAL_CHANNELS are truncated-and-clamped to [0,255] at storage, destroying signed/large coords. The 'coords ride as plain float32, negative-safe' resolution is FALSE for the u8 spatial block. Coords must go in a DT_F32 block \u2014 either retype spatial to fixedF32 (heavy: changes spatial byte layout for all channels + parity + byte-equality test) or add a separate fixedF32 'spatial_coords' [nTiles*2] block that the Python adjacency builder reads. (2) HexCoord.x/.y are Int (HexMath.kt:391), not Float; `out[base+13]=tile.position.x` won't compile \u2014 require .toFloat(). The design's note that .toFloat() is unnecessary is wrong.\\n\\nLockstep gaps to close: export_onnx.py has NO edit and NO mechanism to add the two adj inputs (export_rich auto-pairs _mask per token_spec; sample_inputs only overrides values) \u2014 add an explicit export edit emitting spatial_adj (int64) + spatial_adj_mask with shared n_spatial dynamic axis and static degree-6; validate on the small rung first. dataset.py:113 reshape(-1,13) and :36 docstring must move with the spatial width (listed in lockstep_sites but missing from exact_edits) or be a hard cross-cluster gate. Parity is currently infeasible for v3: parityRunRich (SelfPlayRunner.kt:351-396) only knows vec/set lines and builds via richTensorsFromArrays with no adj support, and the mandated adjacency-parity test does not yet exist \u2014 resolve O3 with a pure coords+wrapRadius neighbor-builder shared by JVM-inference, parity, and the Python train builder.\\n\\nWhat IS correct and verified: frozen 6-dir clock order matches HexMath.clockPositionToHexcoordMap exactly [(1,1)(0,1)(-1,0)(-1,-1)(0,-1)(1,0)]; vocab.buildingCount exists (Vocab.kt:52) so the D1.4 namespace fix is sound and disjoint; the seam forward(inputs:dict)->(tech,policy,tanh(value)) is untouched (new nn.Module only); the two-new-tensor decision (separate int64 adj + f32 mask, not folded into the float pool) is well-justified and export-safe (Gather+Mul+ReduceSum/Mean, no scatter, static degree axis); contract.py fail-loud rewrite of token_specs_from_schema is correct and the schema key is 'spatialChannels' (DataPlaneHooks.kt:171, matched by the `or sch.get('spatialChannels')` branch); Dims.global_w reads layout['global']['len'] so head 5->8 auto-follows; OnnxPolicy {1,2,3} accept-set + structured gate is sound and the fingerprint gate (VERSION + SPATIAL_CHANNELS via Vocab.kt:97) correctly fail-loud-refuses stale v2 models; adj tensors added into the existing `inputs` map are closed by the existing finally. The rich tensors are fed as float32 even from u8 blocks at inference (tensor-level), so the 'spatial is float32 at the tensor' framing is fine for INFERENCE \u2014 the bug is purely on the STORAGE/training side."
}
```
