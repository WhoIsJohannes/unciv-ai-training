# Deep Codebase Scan — v4 Structured Encoder (exact edit-site source)

Verbatim current source for every edit site, from the deep-scan branch (base master `5ac0a3cf6`).
Use these for exact-match edits. Companion to `codebase-scan-light.md` (the map) and `research-notes.md`.

## Resolved: shard dtype → coords as plain float32 channels
`ShardFormat.LeBuffer.f32s(FloatArray)` writes blocks as **float32 LE**; Python reads `<f4`. The
`coerceAtMost(255)` in `buildSpatial` bounds *categorical vocab indices* (embedding-table size), NOT
storage. ⇒ **(x,y) coords = 2 plain float32 spatial channels (13→15), no +1/clamp, negative-safe, no
side-tensor.** (Resolves council FND-0002.) The Python adjacency builder reads ch13/ch14.

---

## Kotlin — Featurizer.kt

`channels = SampleSchema.NUM_SPATIAL_CHANNELS` (line 26). `constrW = vocab.buildingCount + vocab.unitCount` (line 106).
`ownerSlot` lambda (line 43): `if (civId == x.civID) 255 else (presentCivIndex[civId]?.plus(1) ?: 0)` → 0 none, 1..N civ, **255 self**.

### buildSpatial (233–262)
```kotlin
private fun buildSpatial(x: Civilization, ownerSlot: (String) -> Int): FloatArray {
    val tiles = gameInfo.tileMap.tileList
    val out = FloatArray(tiles.size * channels)
    val omni = config.omniscientOpponents
    for (tile in tiles) {
        val base = tile.zeroBasedIndex * channels
        if (base < 0 || base + channels > out.size) continue
        val visible = omni || x.viewableTiles.contains(tile)
        val explored = visible || x.hasExplored(tile)
        out[base] = if (visible) 2f else if (explored) 1f else 0f
        if (!explored) continue
        out[base + 1] = ((vocab.terrain(tile.baseTerrain) + 1).coerceAtMost(255)).toFloat()
        out[base + 2] = ((tile.terrainFeatures.firstOrNull()?.let { vocab.terrain(it) + 1 } ?: 0).coerceAtMost(255)).toFloat()
        out[base + 3] = ((tile.resource?.let { vocab.resource(it) + 1 } ?: 0).coerceAtMost(255)).toFloat()
        out[base + 4] = tile.roadStatus.ordinal.toFloat()
        out[base + 5] = if (tile.hasBottomRightRiver || tile.hasBottomRiver || tile.hasBottomLeftRiver) 1f else 0f
        out[base + 6] = if (tile.isCityCenter()) 1f else 0f
        if (!visible) continue
        out[base + 7] = (tile.getOwner()?.civID?.let { ownerSlot(it) } ?: 0).toFloat()
        out[base + 8] = ((tile.improvement?.let { vocab.improvement(it) + 1 } ?: 0).coerceAtMost(255)).toFloat()
        val unit = tile.getUnits().firstOrNull()
        if (unit != null) {
            out[base + 9] = 1f
            out[base + 10] = (ownerSlot(unit.civ.civID)).toFloat()
            out[base + 11] = unitTypeCat(unit).toFloat()
            out[base + 12] = (unit.health.coerceIn(0, 100) / 100f * 4f).toFloat()
        }
    }
    return out
}
```
**D1.1 edit:** widen `channels` by 2; after the per-tile block, write `out[base+13]=tile.position.x` and
`out[base+14]=tile.position.y` (plain float, OUTSIDE the `if(!explored)`/`if(!visible)` gates — position is
always known/static). Append `tile_x`,`tile_y` to `SPATIAL_CHANNELS`.

### buildGlobal (153–166)
```kotlin
private fun buildGlobal(x: Civilization, demoCtx: DemographicsContext): FloatArray {
    val agg = FloatArray(demographics.size * 3)
    demographics.forEachIndexed { i, cat ->
        val s = demoCtx.perCategory.getValue(cat)
        agg[i * 3] = s.best; agg[i * 3 + 1] = s.avg; agg[i * 3 + 2] = s.worst
    }
    val head = floatArrayOf(
        gameInfo.turns.toFloat(), x.tech.era.eraNumber.toFloat(),
        gameInfo.tileMap.tileList.size.toFloat(),
        x.getKnownCivs().count { it.isMajorCiv() }.toFloat(),
        gameInfo.civilizations.count { it.isMajorCiv() && it.isAlive() }.toFloat(),
    )
    return head + agg
}
```
**D1.2 edit:** append `mapSize.radius`, `worldWrap` bit, `shape` ordinal to `head` (and to the global layout
width). Python adjacency builder needs radius+worldWrap to replicate `TileMap.getIfTileExistsOrNull`.

### writeUnitToken (210–223, width 8) — D1.3 adds tile index
```kotlin
private fun writeUnitToken(arr, off, u, x, isOwn, ownerSlot) {
    val w = TokenSlice(arr, off, unitTokenWidth)
    val capital = x.getCapital()?.getCenterTile(); val t = u.currentTile
    w.put(1f); w.put(if (isOwn) 1f else 0f); w.put(ownerSlot(u.civ.civID)); w.put(unitTypeCat(u))
    w.put((u.health.coerceIn(0,100)/100f*4f).toInt())
    w.put(if (capital!=null) t.position.x.toInt()-capital.position.x.toInt() else 0)
    w.put(if (capital!=null) t.position.y.toInt()-capital.position.y.toInt() else 0)
    w.put(u.promotions.getAvailablePromotions().count())
}
```
**D1.3 edit:** add `w.put(u.currentTile.zeroBasedIndex)` (width 8→9). `unitTokenWidth` is hardcoded at line 29.

### writeCityToken (185–208, width 16) — D1.3 + D1.4
```kotlin
private fun writeCityToken(arr, off, city, x, isOwn, ownerSlot) {
    val w = TokenSlice(arr, off, cityTokenWidth)
    w.put(1f); w.put(if (isOwn) 1f else 0f); w.put(ownerSlot(city.civ.civID))
    w.put(city.population.population); w.put(CityCombatant(city).getDefendingStrength(null))
    w.put(((city.health.coerceIn(0,200))/200f*4f).toInt()); w.put(city.getCenterTile().airUnits.size)
    val rel = city.religion.getMajorityReligionName(); w.put((rel?.let { vocab.religion(it)+1 }) ?: 0)
    w.put(if (city.isInResistance()) 1f else 0f); w.put(if (city.isPuppet) 1f else 0f)
    w.put(if (city.isBeingRazed) 1f else 0f)
    val hasSpy = config.omniscientOpponents || (!isOwn && x.espionageManager.getSpiesInCity(city).any { it.isSetUp() })
    w.put(if (hasSpy) 1f else 0f)
    if (isOwn || hasSpy) {
        val cur = city.cityConstructions.currentConstructionName()
        w.put((vocab.building(cur).takeIf { it >= 0 } ?: vocab.unit(cur)) + 1)   // LINE 205 — BUG
        w.put(city.cityConstructions.getBuiltBuildings().count())
    }
}
```
**D1.4 fix (line 205):** unit branch must add the building-count offset:
`w.put(vocab.building(cur).takeIf{it>=0}?.plus(1) ?: (vocab.unit(cur).takeIf{it>=0}?.plus(vocab.buildingCount+1) ?: 0))`
so building#k (→k+1) and unit#k (→buildingCount+k+1) never collide. `cityTokenWidth` hardcoded line 30.
**D1.3 edit:** add `w.put(city.getCenterTile().zeroBasedIndex)` (city carries no position today).

### buildActingCiv (168–183) — unchanged by v4 (head + tech/policy/branch one-hots; D2 may sum-embed, optional)

---

## Kotlin — SampleSchema.kt
`VERSION = 2` (line 22) → bump to **3** (D1.5). `OnnxContract.CONTRACT_VERSION_RICH = 2` → bump to **3** (D9).
`RICH_TOKEN_NAMES = [spatial, own_units, opp_units, own_cities, opp_cities, civ_tokens]` (unchanged — coords ride
inside spatial, map dims inside global, tile-index inside unit/city tokens → **no new tensor names**, minimal bridge churn).
`SPATIAL_CHANNELS` (86–100, 13 entries) → append `tile_x`,`tile_y` (→15). `NUM_SPATIAL_CHANNELS` is `.size` (auto).
Metadata keys: schema_version, ruleset_fingerprint, contract_version, input_width, tech_width, policy_width, input_names.

## Kotlin — Vocab.kt + SampleConfig.kt + RulesetFingerprint.kt
- Indexers return **-1 on miss**; counts: techCount/buildingCount/unitCount/policyCount/policyBranchCount/resourceCount/promotionCount/nationCount (+ religion/era/terrain/improvement via `index()`).
- `maxCivTokens = maxMajorCivs(16) + maxCityStates(24) = 40` (SampleConfig.kt:17). **Slot table = maxCivTokens+2 = 42** (0 unknown, 1..40 civ, 255 self → reindex 255→41 before embedding). ✓ confirms plan.
- `RulesetFingerprint.compute` hashes `SampleSchema.VERSION` + `Vocab.canonicalSections(ruleset)`, which includes `"schema:spatialChannels" to SampleSchema.SPATIAL_CHANNELS` (Vocab.kt:97). ⇒ adding channels auto-changes the fingerprint (old shards/models refused — fail-loud, intended).

---

## Python — model.py (the seam: forward(inputs:dict)->(tech,policy,tanh(value)))
```python
class _TokenEncoder(nn.Module):           # 66-76  — per-token MLP → masked mean+max pool
    def __init__(self, in_dim, out_dim):
        self.mlp = nn.Sequential(nn.Linear(in_dim,out_dim),nn.ReLU(),nn.Linear(out_dim,out_dim),nn.ReLU())
        self.out_w = 2*out_dim
    def forward(self, tokens, mask): return masked_pool(self.mlp(tokens), mask)

def masked_pool(tokens, mask):            # 50-63 — NaN-guarded mean‖max; all-padding row → 0
    m = mask.unsqueeze(-1)
    safe_count = mask.sum(1,keepdim=True).clamp(min=1.0)
    mean = (tokens*m).sum(1)/safe_count
    masked = tokens.masked_fill(m==0, float("-inf"))
    mx = masked.max(1).values
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    return torch.cat([mean,mx],1)

class RichPolicyValueNet(nn.Module):      # 79-113
    def __init__(self, dims, token_specs, *, token_dim=32, hidden=256):
        self.token_names = list(token_specs.keys())
        self.encoders = nn.ModuleDict({n:_TokenEncoder(w,token_dim) for n,w in token_specs.items()})
        trunk_in = dims.global_w + dims.acting_w + sum(e.out_w for e in self.encoders.values())
        self.trunk = nn.Sequential(nn.Linear(trunk_in,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.tech_head = nn.Linear(hidden,dims.tech_w); self.policy_head = nn.Linear(hidden,dims.policy_w)
        self.value_head = nn.Linear(hidden,1); _small_init_value_head(self.value_head)
    def forward(self, inputs):
        parts=[inputs[self.INPUT_GLOBAL], inputs[self.INPUT_ACTING]]
        for name in self.token_names: parts.append(self.encoders[name](inputs[name], inputs[name+"_mask"]))
        h=self.trunk(torch.cat(parts,1))
        return self.tech_head(h), self.policy_head(h), torch.tanh(self.value_head(h))
```
`_small_init_value_head` (20-26): uniform(-1e-3,1e-3) weight, zero bias.
**v4:** add a NEW module `StructuredPolicyValueNet` with the SAME `forward(inputs:dict)->(tech,policy,tanh(value))` and the SAME `INPUT_GLOBAL/INPUT_ACTING` attrs, consuming the new neighbor-index/mask + embeddings. `_optimize_actor_critic`/`compute_gae`/train_actor_critic_rich core UNTOUCHED — only the nn.Module swapped.

## Python — features.py, contract.py, export_onnx.py, dataset.py, run_loop.py
(full bodies captured — key edit points)
- `features.py`: `build_rich_batch` (33-48), `build_rich_single` (51-69, parity ref — spatial `reshape(-1,width)`), `_pad_token_set` (17-30). v4 adds neighbor-index/mask construction here (or a `hexgraph.py` called here).
- `contract.py`: `CONTRACT_VERSION_RICH=2`→3; `RICH_TOKEN_NAMES` (29); `token_specs_from_schema` (85-102) reads `spatial_channels` len; `_TOKEN_WIDTH_FALLBACK` (80-82) — **make fail-loud vs schema** (FND-0007/0011). `Dims` (45-56).
- `export_onnx.py`: `export_rich` (82-138) builds positional `names` in order, dummy shapes, `dynamic_axes` (batch{0}+ragged{1:"n_<name>"}), opset 17, stamps metadata via `_RichPolicyOnly` (32-44). v4: feed neighbor-index/mask as extra inputs with their own dynamic axes; **export smoke test on small rung FIRST** (FND-0008/0023).
- `dataset.py`: `_rich_step_blocks` (106-121) — **`reshape(-1,13)` hardcoded at line 113 → must read channel count from schema** (FND-0007). `RICH_TOKEN_BLOCKS` (37).
- `run_loop.py`: `train_round` dispatch (74-94) — add `--variant structured` (alias rich-v2) → new encoder + ladder/guard. Rich export block (211-222).

## JVM bridge — OnnxPolicy.kt
- Contract gate (59): `rich = mContract == CONTRACT_VERSION_RICH` → must accept **3**. Rich-input inventory check (66-74) — must include any new input names (none new if neighbor-index rides as a derived tensor... NOTE: neighbor-index is **derived in Python at train time** but the EXPORTED ONNX expects it as an input → the JVM must ALSO build it at inference. **Design decision: JVM builds neighbor-index/mask from spatial coords + global map-dims at inference**, OR bake adjacency as a constant. See plan §D3-inference.)
- `forwardRich` (127-138), `buildRichTensors` (153-166), `richTensorsFromArrays` (171-189), `tokenTensors` (196-204, empty→N=1 zero-token, mask 0). `FALLBACK_WIDTH` (150-151) — keep in lockstep with contract.py. All tensors closed in finally.

## SimBenchmark.kt (D8 EXTENDS this — already exists)
`internal object SimBenchmark` (36); `main` (51-77); `runBoundedTurns` (184-193, single game N turns wall-clock); `runFullGames` (196-203, MT batch via `Simulation`); `buildBaseGame` (157-181, Medium, 6 majors+6 CS+barbs). Config: NUM_MAJOR_CIVS=6, NUM_CITY_STATES=6, BARBARIANS=true, MAX_TURNS=500, MT_CAP_TURNS=300. Gradle `:desktop:simBench`. `BENCH|`-prefixed output. **D8: add an ONNX-policy mode (inject OnnxPolicy as the learner), measure turns/s + ms/decision (wrap forwardRich in Timers.timeThis), report vs the existing heuristic baseline, reject rung < 70%.**

## Tests — test_parity.py + HexMath/TileMap
- `test_jvm_python_rich_logits_match` (78-131): dims(4,4,5,4), token_specs spatial:3 + entities; builds fixture (vec/set lines), runs ORT (python) vs `./gradlew selfPlay --args="parity-run-rich <model> <obs> <out>"`, atol 1e-4, incl. empty set. JVM entry: `SelfPlayRunner.kt:73 "parity-run-rich"→parityRunRich (351-396)`. **D10: extend fixture + dims to the richer input incl. neighbor-index/mask + logits.**
- `HexMath.clockPositionToHexcoordMap` (277-289): 12/0=(1,1) 2=(0,1) 4=(-1,0) 6=(-1,-1) 8=(0,-1) 10=(1,0).
- `TileMap.getIfTileExistsOrNull` (376-398): try (x,y); if worldWrap, radius=mapSize.radius (rectangular: width/2); try (x+radius,y-radius) then (x-radius,y+radius). **Python adjacency builder replicates EXACTLY** (+ adjacency-parity test, FND-0036).

---

## Collision / refactor risks
1. Hardcoded widths: `unitTokenWidth=8`/`cityTokenWidth=16` (Featurizer 29-30), `_TOKEN_WIDTH_FALLBACK` (contract.py 80-82), `FALLBACK_WIDTH` (OnnxPolicy 150-151), `reshape(-1,13)` (dataset.py 113), parity dims. ALL must move in lockstep; make fallbacks fail-loud vs schema.
2. Construction-namespace bug (Featurizer 205) — live; D1.4 + AC7 test.
3. Fingerprint coupling — adding channels changes fingerprint (intended; regenerate shards).
4. Fallback-width drift Python↔Kotlin — parity uses synthetic dims so won't catch real-width drift; add a schema-consistency assert.
5. **Neighbor-index must be built on BOTH sides** (Python train + JVM inference) OR be a model-derived constant — resolve in plan §D3.

## Top surprises
1. civ_tokens width = **84** is per-civ feature width (FairOpponentModel computed), NOT the 42 slot-table — distinct concepts; both correct.
2. spatial stored FLAT `[nTiles*channels]` float32, reshaped at read — keep reshape sites in lockstep with channel count.
3. empty entity sets: JVM pads N=max(1,count) zero-token+mask0; Python masked_pool NaN-guards — replicate exactly for new neighbor-mask.
4. v1+v2 models coexist via the contract gate; v4 becomes contract **v3** (third path) — gate must accept {1,2,3} or {2,3}.
5. coords stored as float32 (no quantization) → 2 plain coord channels, no side-tensor (resolves FND-0002).
6. `overflow` flag exists but is diagnostic-only (not fed to policy) — out of scope.
7. 3-valued visibility (0/1/2) fed raw — preserve semantics.
