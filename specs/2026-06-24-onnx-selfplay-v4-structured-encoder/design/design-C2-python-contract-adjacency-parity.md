# Design — C2-python-contract-adjacency-parity

## Summary
C2 makes the Python side of v4 consume two new degree-6 hex-adjacency tensors (neighbor_index [N,6] int64, neighbor_mask [N,6] float32) derived purely from the emitted per-tile coords (spatial ch13/14) plus three map-dim scalars (effective wrap-radius, worldWrap bit, shape ordinal) the C1 cluster adds to the global head. A new hexgraph.py replicates TileMap.getTilesAtDistance(1)+getIfTileExistsOrNull EXACTLY (the 6 offsets, world-wrap retry, OOB-safe index bounding per FND-0025); features.py/dataset.py/contract.py/export_onnx.py thread these tensors through with the spatial channel count read from schema (fail-loud fallback, FND-0007/0011); the contract bumps to RICH v3 in lockstep. D10 extends rich parity to the multi-tensor adjacency input, adds an adjacency-parity test (Python builder vs hand-computed world-wrap+ragged reference, FND-0036) and a contract-version-mismatch refusal test (FND-0022). The neighbor tensors are NEW ONNX inputs (cannot be folded — they are data-dependent on live geometry and must be built at inference by the JVM too), with ragged dynamic axis n_spatial shared with the spatial token set.

## Detailed design
# C2 Design — Python contract / features / dataset / export + hexgraph adjacency builder + parity

All file:line anchors are against the worktree `/Users/j/Unciv-onnx-selfplay-loop`.

## 0. The load-bearing facts grounded from source

- Spatial is stored FLAT `[nTiles*channels]` float32 and reshaped on read (`dataset.py:113` `reshape(-1,13)`; `features.py:63` `arr.reshape(-1, width)`). Spatial row `r` == the tile with `zeroBasedIndex==r` (Featurizer writes `out[tile.zeroBasedIndex*channels]`; `TileMap.setTransients` sets `zeroBasedIndex = values.withIndex()` order, `TileMap.kt:610-617`). ⇒ after C1's D1.1, `spatial[r,13]==tile.position.x`, `spatial[r,14]==tile.position.y`. The `(x,y)->row` map is therefore exactly `{(int(spatial[r,13]),int(spatial[r,14])): r}`.
- The REAL neighbor enumeration is `Tile.neighbors = getTilesAtDistance(1)` (`Tile.kt:115`), implemented at `TileMap.kt:254-285`. For `distance==1` it yields 6 calls to `getIfTileExistsOrNull`, in this exact order of (dx,dy) offsets from the center:
  1. `(currentX,currentY)` start = `(cx-1,cy-1)` → **(-1,-1)**
  2. `(2cx-currentX,2cy-currentY)` → **(+1,+1)**
  3. after `currentX+=1`: `(cx,cy-1)` → **(0,-1)**
  4. mirror → **(0,+1)**
  5. after `currentX+=1,currentY+=1`: `(cx+1,cy)` → **(+1,0)**
  6. mirror → **(-1,0)**
  So the degree-6 offset list, in JVM emission order, is `OFFSETS = [(-1,-1),(+1,+1),(0,-1),(0,+1),(+1,0),(-1,0)]`. This is a permutation of `HexMath.clockPositionToHexcoordMap`'s set `{(1,1),(0,1),(-1,0),(-1,-1),(0,-1),(1,0)}` (the spec's 12/2/4/6/8/10 list) — SAME six neighbors, different order. Because the GNN reduce over the degree-6 axis is permutation-invariant the order is mathematically irrelevant to the GNN, BUT Python and the JVM inference builder MUST use the IDENTICAL order so the per-slot neighbor_index/neighbor_mask tensors are byte-identical (the parity test asserts this). **Decision: adopt the `getTilesAtDistance(1)` order above as the canonical `OFFSETS`** (it is the order the live JVM neighbors actually materialize, so the JVM inference builder can reuse `tile.neighbors` directly if desired — see C3 note).
- `getIfTileExistsOrNull(x,y)` (`TileMap.kt:376-398`): (A) if `(x,y)` is a present tile return it; (B) else if `!worldWrap` return null; (C) else `radius = mapSize.radius`, but `if shape==rectangular: radius = mapSize.width/2`; try `(x+radius, y-radius)`; (D) else try `(x-radius, y+radius)`; else null. `getOrNull(x,y)` (`TileMap.kt:222`) is purely "is (x,y) a present coordinate?" — no leftX/bottomY needed in Python because the present-coord set is the `(x,y)->row` keyset.
- ⚠️ **wrap-radius ambiguity:** the effective wrap radius is `mapSize.width/2` for rectangular and `mapSize.radius` otherwise. C1's decisions.md says it emits `radius, worldWrap, shape`. If C1 emits the raw `mapSize.radius`, Python CANNOT recover `width/2` for rectangular maps from radius alone. **Contract requirement on C1 (lockstep): emit the ALREADY-RESOLVED effective wrap radius as a single float `wrap_radius` (Kotlin computes `if (shape==rectangular) mapSize.width/2 else mapSize.radius`).** Then Python needs only `(wrap_radius, worldWrap)` and `shape` becomes optional/diagnostic. I design the builder to accept `wrap_radius` directly; I also keep a `shape`+`width`+`radius` fallback path documented as an open question in case C1 emits raw fields.

## 1. New module `python/unciv_train/hexgraph.py` (D3-Python)

```python
"""Degree-6 hex-adjacency builder — EXACT Python replica of Unciv's
TileMap.getTilesAtDistance(1) + getIfTileExistsOrNull world-wrap, for the export-safe
gather-GNN (research-notes D-gnn-export: no scatter_add). Produces neighbor_index [N,6]
(int64) + neighbor_mask [N,6] (float32). All indices are bounded 0<=idx<N so ORT Gather
can never read OOB (council FND-0025); missing neighbors get index 0 + mask 0.
"""
from __future__ import annotations
import numpy as np

# JVM getTilesAtDistance(1) emission order — see design notes. MUST stay byte-identical
# to the JVM inference neighbor builder (parity test guards this).
OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (1, 1), (0, -1), (0, 1), (1, 0), (-1, 0),
)
SENTINEL = 0  # bounded dummy row for a missing neighbor (paired with mask 0)


def build_neighbor_graph(
    coords: np.ndarray,          # [N,2] int — (x,y) per spatial row (ch13,ch14)
    *,
    wrap_radius: int,            # effective wrap radius: width/2 (rect) else mapSize.radius
    world_wrap: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (neighbor_index [N,6] int64, neighbor_mask [N,6] float32)."""
    n = int(coords.shape[0])
    # (x,y) -> row, replicating getOrNull's "is this a present coordinate" test.
    coord_to_row: dict[tuple[int, int], int] = {}
    for r in range(n):
        coord_to_row[(int(coords[r, 0]), int(coords[r, 1]))] = r

    def resolve(x: int, y: int) -> int | None:        # == getIfTileExistsOrNull
        row = coord_to_row.get((x, y))
        if row is not None:
            return row
        if not world_wrap:
            return None
        rs = wrap_radius
        row = coord_to_row.get((x + rs, y - rs))       # A: right->left wrap
        if row is not None:
            return row
        row = coord_to_row.get((x - rs, y + rs))       # B: left->right wrap
        return row

    nbr_idx = np.full((n, 6), SENTINEL, dtype=np.int64)
    nbr_mask = np.zeros((n, 6), dtype=np.float32)
    for r in range(n):
        cx, cy = int(coords[r, 0]), int(coords[r, 1])
        for k, (dx, dy) in enumerate(OFFSETS):
            row = resolve(cx + dx, cy + dy)
            if row is not None:
                nbr_idx[r, k] = row
                nbr_mask[r, k] = 1.0
    # FND-0025 hard bound: indices must be valid Gather rows even for the sentinel.
    assert nbr_idx.min() >= 0 and nbr_idx.max() < max(n, 1), "neighbor index OOB"
    return nbr_idx, nbr_mask


def build_neighbor_graph_from_spatial(
    spatial: np.ndarray,         # [N, n_channels] float32 (post reshape)
    global_vec: np.ndarray,      # [global_w] float32 — map dims live at fixed head slots
    *,
    x_channel: int,              # index of tile_x channel (== n_channels-2)
    y_channel: int,              # index of tile_y channel (== n_channels-1)
    wrap_radius_slot: int,       # global head slot of wrap_radius (see GLOBAL_MAPDIM_OFFSET)
    world_wrap_slot: int,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.stack([spatial[:, x_channel], spatial[:, y_channel]], axis=1).astype(np.int64)
    wrap_radius = int(round(float(global_vec[wrap_radius_slot])))
    world_wrap = bool(round(float(global_vec[world_wrap_slot])))
    return build_neighbor_graph(coords, wrap_radius=wrap_radius, world_wrap=world_wrap)
```

Notes:
- `resolve` is a 1:1 port of `getIfTileExistsOrNull`: (A) direct, (B) `!worldWrap` short-circuit, (C) `(x+r,y-r)`, (D) `(x-r,y+r)`. Order of A/B/C/D preserved exactly (FND-0036 hinges on this).
- Missing neighbor → `(SENTINEL=0, mask 0)`. SENTINEL must be a valid row (0) not -1, because ORT `Gather` will read it before the mask zeroes it; mask multiply makes its contribution exactly 0. This is the same discipline as `masked_pool` (model.py:57-62) and the JVM empty-token pad (N=1 zero-token + mask 0).
- O(N·6) with N≈1261 (Medium) — trivially fast. No scatter anywhere (research-notes Q1).

## 2. `contract.py` (D9-Python)

- Bump `CONTRACT_VERSION_RICH = 2` → `3` (contract.py:18), lockstep with `SampleSchema.OnnxContract.CONTRACT_VERSION_RICH` and `OnnxPolicy` gate.
- **Decision: neighbor_index/neighbor_mask are NEW named ONNX inputs, NOT folded.** They are data-dependent on the live map geometry, cannot be constant-baked (map differs per game/rung), and the ONNX graph's `Gather` needs them as runtime tensors. They ride alongside the `spatial` token set sharing its ragged axis `n_spatial`. Add explicit names so the positional export order and the JVM inventory check are unambiguous:
```python
INPUT_NEIGHBOR_INDEX = "neighbor_index"   # [batch, n_spatial, 6] int64
INPUT_NEIGHBOR_MASK = "neighbor_mask"     # [batch, n_spatial, 6] float32
NEIGHBOR_DEGREE = 6
```
  They are appended to the ordered input list IMMEDIATELY after the spatial pair (`spatial`, `spatial_mask`) so the positional order is `global, acting_civ, spatial, spatial_mask, neighbor_index, neighbor_mask, own_units, own_units_mask, ...`. Keeping them adjacent to spatial documents the coupling. (Alternative — append at the very end — also works; adjacency chosen for readability. Either way the order is frozen by the parity test.)
- `token_specs_from_schema` (contract.py:85-102): unchanged in shape (spatial width now reads as 15 automatically from `len(spatial_channels)`), but **make the fallback FAIL-LOUD** (FND-0007/0011): if the schema omits `spatial_channels`/`spatialChannels` OR an entity `perItem`, raise instead of silently using `_TOKEN_WIDTH_FALLBACK`:
```python
def token_specs_from_schema(schema_path):
    sch = _schema(schema_path)
    layout = {b["name"]: b for b in sch.get("layout", [])}
    chans = sch.get("spatial_channels") or sch.get("spatialChannels")
    if not chans:
        raise ValueError(f"{schema_path}: schema omits spatial_channels — refusing silent fallback "
                         f"(spatial width is the v4 god-constant; SSOT is Kotlin SampleSchema)")
    specs = {"spatial": len(chans)}
    for name in ("own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens"):
        entry = layout.get(name)
        if not (entry and entry.get("perItem")):
            raise ValueError(f"{schema_path}: layout[{name!r}] missing perItem — refusing silent fallback")
        specs[name] = int(entry["perItem"])
    return specs
```
  `_TOKEN_WIDTH_FALLBACK` (contract.py:80-82) is then DELETED (no silent revert path). Update its docstring/usages; nothing else references it (grep-confirmed only used inside `token_specs_from_schema`).
- Add `GLOBAL_MAPDIM_OFFSET = 5` and `MAPDIM_SLOTS = ("wrap_radius","world_wrap","shape")` constants documenting that C1 appends the 3 map-dim scalars at global head slots 5,6,7 (right after the 5 existing head scalars `turns,era,tileCount,knownMajors,aliveMajors`, BEFORE the demographics agg block). The builder reads `wrap_radius_slot=GLOBAL_MAPDIM_OFFSET`, `world_wrap_slot=GLOBAL_MAPDIM_OFFSET+1`. (This is a hard lockstep contract with C1's buildGlobal edit — see lockstep_sites.)
- `Dims` (contract.py:45-56): no change (global_w auto-grows by 3 via schema; neighbor tensors carry no learned width). Confirm `dims_from_schema` still reads `layout["global"]["len"]` which now includes the 3 map-dims — correct, no edit.

## 3. `features.py` (D3-Python wiring)

`build_rich_single` (features.py:51-69) and `build_rich_batch` (features.py:33-48) must emit the two neighbor tensors. The spatial width is now 15 and the last two channels are coords.

`build_rich_single` edit (after the spatial reshape, before/with the token loop):
```python
from .hexgraph import build_neighbor_graph
from .contract import GLOBAL_MAPDIM_OFFSET

def build_rich_single(step_blocks, token_specs):
    g = np.asarray(step_blocks["global"], dtype=np.float32).reshape(1, -1)
    a = np.asarray(step_blocks["acting_civ"], dtype=np.float32).reshape(1, -1)
    out = {"global": torch.tensor(g), "acting_civ": torch.tensor(a)}
    spatial_w = token_specs["spatial"]
    for name, width in token_specs.items():
        arr = np.asarray(step_blocks[name], dtype=np.float32)
        if name == "spatial":
            arr = arr.reshape(-1, width)
            nbr_idx, nbr_mask = build_neighbor_graph(
                np.stack([arr[:, width-2], arr[:, width-1]], 1).astype(np.int64),
                wrap_radius=int(round(float(g[0, GLOBAL_MAPDIM_OFFSET]))),
                world_wrap=bool(round(float(g[0, GLOBAL_MAPDIM_OFFSET+1]))))
            out["neighbor_index"] = torch.tensor(nbr_idx[None, ...])     # [1,N,6]
            out["neighbor_mask"] = torch.tensor(nbr_mask[None, ...])     # [1,N,6]
        elif arr.ndim == 1:
            arr = arr.reshape(0, width) if arr.size == 0 else arr.reshape(-1, width)
        padded, mask = _pad_token_set([arr], width)
        out[name] = torch.tensor(padded)
        out[name + "_mask"] = torch.tensor(mask)
    return out
```

`build_rich_batch` edit — neighbor graph is per-step and N varies per step; spatial is padded to batch-max N (the `_pad_token_set` over `spatial`). The neighbor tensors must be padded to the SAME max N and the SAME slot order. Build per-step graphs, then pad to `[B, maxN, 6]`:
```python
def build_rich_batch(trajectories, dims, token_specs):
    steps = [blk for t in trajectories for blk in (t.rich or [])]
    n = len(steps); assert n > 0, "build_rich_batch called with no steps"
    out = {"global": torch.tensor(np.stack([s["global"] for s in steps]).astype(np.float32)),
           "acting_civ": torch.tensor(np.stack([s["acting_civ"] for s in steps]).astype(np.float32))}
    spatial_w = token_specs["spatial"]
    # per-step spatial arrays already [Ni, spatial_w] from _rich_step_blocks
    spatials = [np.asarray(s["spatial"], np.float32) for s in steps]
    maxn = max(1, max(a.shape[0] for a in spatials))
    nbr_idx = np.zeros((n, maxn, 6), np.int64)
    nbr_mask = np.zeros((n, maxn, 6), np.float32)
    for i, sp in enumerate(spatials):
        ni = sp.shape[0]
        if ni:
            gi, gm = build_neighbor_graph(
                np.stack([sp[:, spatial_w-2], sp[:, spatial_w-1]], 1).astype(np.int64),
                wrap_radius=int(round(float(out["global"][i, GLOBAL_MAPDIM_OFFSET]))),
                world_wrap=bool(round(float(out["global"][i, GLOBAL_MAPDIM_OFFSET+1]))))
            nbr_idx[i, :ni] = gi; nbr_mask[i, :ni] = gm
    out["neighbor_index"] = torch.tensor(nbr_idx)
    out["neighbor_mask"] = torch.tensor(nbr_mask)
    for name, width in token_specs.items():
        padded, mask = _pad_token_set([s[name] for s in steps], width)
        out[name] = torch.tensor(padded); out[name + "_mask"] = torch.tensor(mask)
    return out
```
  ⚠️ **Padding-consistency invariant:** the neighbor tensors are padded to `maxn` computed from the spatial arrays exactly as `_pad_token_set` computes its `maxn` for `spatial` — they MUST agree. Compute `maxn` once and reuse (or assert `out["spatial"].shape[1] == nbr_idx.shape[1]`). Per-step indices into a step's own spatial rows remain < that step's Ni ≤ maxn, so padding rows (mask 0) keep index 0 (valid). Cross-step contamination impossible because each step's graph is built from its own coords only.

## 4. `dataset.py` (D9-Python)

`_rich_step_blocks` (dataset.py:106-121) line 113 `reshape(-1, 13)` is the hardcoded god-constant. Replace with the schema channel count, threaded in. Two options:
- (chosen) pass `n_channels` into `_rich_step_blocks` from `load_trajectories` (which already has `expected_version`; add `expected_spatial_channels: int`). `load_trajectories` is called from run_loop with the schema available.
```python
def _rich_step_blocks(blocks, n_channels):
    out = {"global": np.asarray(blocks["global"], np.float32),
           "acting_civ": np.asarray(blocks["acting_civ"], np.float32)}
    spatial = np.asarray(blocks["spatial"], np.float32)
    assert spatial.size % n_channels == 0, (
        f"spatial block size {spatial.size} not divisible by n_channels {n_channels} "
        "(schema/shard channel-count drift)")
    out["spatial"] = spatial.reshape(-1, n_channels)
    for name in ("own_units","opp_units","own_cities","opp_cities","civ_tokens"):
        b = blocks.get(name)
        arr = np.asarray(b, np.float32) if b is not None else np.zeros((0,0), np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(0, arr.shape[0] if arr.size else 0)
        out[name] = arr
    return out
```
  `load_trajectories` signature gains `expected_spatial_channels: int` and passes it at dataset.py:163. `RICH_TOKEN_BLOCKS` (dataset.py:37) unchanged (neighbor tensors are NOT shard blocks — they are derived, never stored; the HARD CONSTRAINT "no edge list in shard"). The docstring at dataset.py:36 and line 107 that says "FIXED nTiles*13" must be updated to "nTiles*n_channels (schema-driven)".
- The caller (run_loop.py:204) must pass `expected_spatial_channels=len(schema["spatial_channels"])`. run_loop is C4's file; I specify the contract: `load_trajectories(..., expected_spatial_channels=...)`.

## 5. `export_onnx.export_rich` (D9-Python)

Add the two neighbor inputs with correct dtypes + dynamic axes, sharing the `n_spatial` ragged axis. Critical: `neighbor_index` is **int64** (ORT Gather index dtype) — must export as int64, not float32. The current dummy-building loop is float32-only; neighbor tensors need their own typed dummies inserted in positional order right after spatial.

```python
def export_rich(net, dims, token_specs, out_path, *, schema_version,
                ruleset_fingerprint, sample_inputs=None, opset=17):
    import numpy as np
    out_path = str(out_path); net.eval()
    names = [contract.INPUT_GLOBAL, contract.INPUT_ACTING]
    dummy = {contract.INPUT_GLOBAL: torch.zeros(1, dims.global_w),
             contract.INPUT_ACTING: torch.zeros(1, dims.acting_w)}
    dyn = {contract.INPUT_GLOBAL: {0: "batch"}, contract.INPUT_ACTING: {0: "batch"},
           contract.OUTPUT_TECH: {0: "batch"}, contract.OUTPUT_POLICY: {0: "batch"}}
    n0 = 2
    for name, width in token_specs.items():
        dummy[name] = torch.zeros(1, n0, width)
        dummy[name + "_mask"] = torch.ones(1, n0)
        names += [name, name + "_mask"]
        dyn[name] = {0: "batch", 1: "n_" + name}
        dyn[name + "_mask"] = {0: "batch", 1: "n_" + name}
        if name == "spatial":   # neighbor tensors share spatial's ragged axis, sit right after it
            ni = contract.INPUT_NEIGHBOR_INDEX; nm = contract.INPUT_NEIGHBOR_MASK
            dummy[ni] = torch.zeros(1, n0, contract.NEIGHBOR_DEGREE, dtype=torch.int64)
            dummy[nm] = torch.ones(1, n0, contract.NEIGHBOR_DEGREE)
            names += [ni, nm]
            dyn[ni] = {0: "batch", 1: "n_spatial"}
            dyn[nm] = {0: "batch", 1: "n_spatial"}
    if sample_inputs is not None:   # keep int64 for neighbor_index when overriding the dummy
        for k, v in sample_inputs.items():
            t = torch.as_tensor(np.asarray(v))
            dummy[k] = t.long() if k == contract.INPUT_NEIGHBOR_INDEX else t.float()
    wrapped = _RichPolicyOnly(net, names)
    args = tuple(dummy[n] for n in names)
    torch.onnx.export(wrapped, args, out_path, input_names=names,
                      output_names=[contract.OUTPUT_TECH, contract.OUTPUT_POLICY],
                      dynamic_axes=dyn, opset_version=opset)
    model = onnx.load(out_path)
    onnx.helper.set_model_props(model, { ... META_INPUT_NAMES: ",".join(names), ...,
        contract.META_CONTRACT_VERSION: str(contract.CONTRACT_VERSION_RICH)})  # now "3"
    onnx.save(model, out_path)
```
  `_RichPolicyOnly.forward` (export_onnx.py:41-44) is unchanged (it already reassembles the dict from positional `names`). The dummy with `n0=2` spatial rows and a `neighbor_index` referencing rows {0,1} (zeros are valid) traces fine.
  **Export smoke test (FND-0008/0023, gate before scaling):** a unit test that builds the StructuredPolicyValueNet on the SMALL rung dims, calls `export_rich`, reloads with `onnx.checker.check_model`, runs an ORT `InferenceSession`, and asserts the input names == `names` (incl. neighbor_index/mask) and neighbor_index dtype == int64, with a tiny synthetic feed (N=4 tiles, a hand-built degree-6 graph). This runs in the Python-only test suite (no JVM), so it gates the export op-coverage of the gather-GNN early.

## 6. D10 parity + new tests

### 6a. Extend `test_jvm_python_rich_logits_match` (test_parity.py:77-131)
- Switch `net = StructuredPolicyValueNet(...)` (the C2/C4 module) and `schema_version=3`, `ruleset_fingerprint` unchanged.
- The fixture `blocks["spatial"]` becomes `[7,5]` (3 base feature channels for the synthetic dims + 2 coord channels) with hand-chosen integer coords forming a small connected hex patch (so some neighbor slots are real, some ragged-edge → mask 0). `dims`/`token_specs` `spatial: 5`.
- `build_rich_single` now also emits `neighbor_index`/`neighbor_mask` into `feed` automatically — no extra Python wiring; the ORT `sess.run(..., feed)` consumes them.
- The JVM fixture writer must ALSO emit the neighbor tensors so the JVM `parity-run-rich` consumes the SAME graph (the JVM side is C3's `parityRunRich`; the fixture format gains `neighbor_index set N 6 ...` and `neighbor_mask set N 6 ...` lines via `_fixture_line_set`). Append:
```python
nbr_idx = feed["neighbor_index"][0]; nbr_mask = feed["neighbor_mask"][0]
lines.append(_fixture_line_set("neighbor_index", nbr_idx.astype(np.float32), 6))
lines.append(_fixture_line_set("neighbor_mask", nbr_mask, 6))
```
  (int64 indices are written as floats in the text fixture and parsed back; values are small exact integers so the round-trip is lossless — document this; alternatively add an int fixture line type. Keep atol 1e-4 on the logits.)

### 6b. NEW adjacency-parity test (FND-0036) — Python-only, no JVM
Hand-compute a reference for: (i) an interior tile (all 6 neighbors present), (ii) a ragged map-edge tile (some neighbors missing → mask 0), (iii) a world-wrap tile where the wrapped neighbor resolves via branch (C) `(x+r,y-r)` and (iv) branch (D) `(x-r,y+r)`. Assert `build_neighbor_graph` reproduces the hand-computed `neighbor_index`/`neighbor_mask` exactly, including the OFFSETS order and the OOB index bound. This pins the EXACT replication of `getIfTileExistsOrNull` independent of the JVM (cheap, always runs). Add an optional JVM cross-check variant later (a `parity-adjacency` JVM entry dumping `tile.neighbors` zeroBasedIndex per tile) — flagged as open question whether C3 adds that entry.

### 6c. NEW contract-version-mismatch refusal test (FND-0022) — Python-only
- Export a rich model with `schema_version=2` (or stamp `META_CONTRACT_VERSION="2"`), then assert the v4 load/consumer path REFUSES it (raises) rather than silently mis-decoding. Mirror the existing `ProvenanceError` discipline (dataset.py:21,76-82): construct a fake shard provenance with `schema_version != 3` and assert `load_trajectories(..., expected_version=3)` raises `ProvenanceError`. Also assert `token_specs_from_schema` raises on a schema lacking `spatial_channels` (the fail-loud fallback from §2). This covers both the migration refusal and the silent-fallback refusal in one test module.

## 7. Complexity / export-safety recap (for the plan)
- Spatial uses the gather-GNN over the degree-6 axis: O(N·6) memory/compute, N≈1261 — NOT O(N²). The neighbor tensors add `N*6` int64 + `N*6` float32 per step (~60KB/step on Medium) — negligible, derived at batch-build time, never stored in shards (HARD CONSTRAINT honored).
- All adjacency ops on the ONNX side are `Gather` + `Mul` + `ReduceSum/Mean` (opset ≤13 core) — the export-safe realization from research-notes Q1; no scatter_add/index_add/scatter_reduce anywhere in C2.

## Exact edits
- **python/unciv_train/hexgraph.py** [new file]: Create module with OFFSETS=((-1,-1),(1,1),(0,-1),(0,1),(1,0),(-1,0)), SENTINEL=0, build_neighbor_graph(coords,*,wrap_radius,world_wrap)->(neighbor_index[N,6] int64, neighbor_mask[N,6] f32) replicating getTilesAtDistance(1)+getIfTileExistsOrNull, plus build_neighbor_graph_from_spatial helper. Asserts 0<=idx<N (FND-0025).
  _why:_ D3-Python: the export-safe gather-GNN adjacency, exact Kotlin replica.
- **python/unciv_train/contract.py** [line 18]: CONTRACT_VERSION_RICH = 2 -> 3
  _why:_ D9 lockstep contract bump for the v4 multi-tensor input.
- **python/unciv_train/contract.py** [after line 29 (RICH_TOKEN_NAMES)]: Add INPUT_NEIGHBOR_INDEX='neighbor_index', INPUT_NEIGHBOR_MASK='neighbor_mask', NEIGHBOR_DEGREE=6; add GLOBAL_MAPDIM_OFFSET=5 and MAPDIM_SLOTS doc constants.
  _why:_ New ONNX input names + the global-head map-dim slot contract with C1.
- **python/unciv_train/contract.py** [lines 80-102]: Delete _TOKEN_WIDTH_FALLBACK; rewrite token_specs_from_schema to RAISE ValueError when spatial_channels or an entity perItem is missing (no silent fallback).
  _why:_ FND-0007/0011 fail-loud god-constant; Kotlin SampleSchema is SSOT.
- **python/unciv_train/features.py** [lines 33-48 (build_rich_batch) and 51-69 (build_rich_single)]: Import hexgraph + GLOBAL_MAPDIM_OFFSET; when name=='spatial' compute neighbor_index/neighbor_mask from coords (last 2 channels) + global map-dim slots and add to the output dict; in batch path pad neighbor tensors to the same max-N as the spatial token set.
  _why:_ D3-Python: thread the derived adjacency tensors into both batch + parity-reference paths.
- **python/unciv_train/dataset.py** [line 113 (and signature of _rich_step_blocks + load_trajectories call at 163)]: Replace reshape(-1,13) with reshape(-1,n_channels); add n_channels param to _rich_step_blocks; add expected_spatial_channels param to load_trajectories; assert spatial.size % n_channels == 0. Update docstrings (lines 36,107) from 'nTiles*13' to schema-driven.
  _why:_ FND-0007 remove hardcoded channel count; fail-loud on drift.
- **python/unciv_train/export_onnx.py** [lines 109-117 (token loop) + 128-138 (metadata)]: Inside the token loop, when name=='spatial' also add neighbor_index (int64 dummy [1,n0,6]) and neighbor_mask (f32 [1,n0,6]) to dummy/names/dyn with shared 'n_spatial' axis; keep int64 dtype when sample_inputs overrides neighbor_index. Metadata META_CONTRACT_VERSION now stamps CONTRACT_VERSION_RICH (=3) and META_INPUT_NAMES includes the new names.
  _why:_ D9 export with correct dtype + dynamic axes; positional order frozen.
- **python/tests/test_parity.py** [lines 85-131]: Switch to StructuredPolicyValueNet, schema_version=3, spatial width 5 with 2 coord channels forming a small hex patch; build_rich_single now emits neighbor tensors into feed; write neighbor_index/neighbor_mask fixture lines for the JVM side.
  _why:_ D10 extend rich parity to the multi-tensor adjacency input.
- **python/tests/test_hexgraph.py** [new file]: Adjacency-parity test (FND-0036): interior/ragged-edge/world-wrap branch-C/branch-D cases vs hand-computed neighbor_index/mask; assert OFFSETS order + index bounds.
  _why:_ D10 guard the Python-vs-Kotlin replication.
- **python/tests/test_contract_mismatch.py** [new file]: Contract/version-mismatch refusal (FND-0022): assert load_trajectories raises ProvenanceError on schema_version!=3; assert token_specs_from_schema raises when spatial_channels missing; export smoke test (export_rich on small rung -> onnx.checker + ORT run, neighbor_index dtype int64, input names match).
  _why:_ D10 + FND-0008/0023 export gate + fail-loud refusal.

## New inputs/tensors
- neighbor_index [batch, n_spatial, 6] int64 — degree-6 neighbor row indices; NEW ONNX input (data-dependent on live geometry, cannot be folded/baked; ORT Gather needs it at runtime; JVM rebuilds at inference). Sentinel 0 + mask 0 for missing.
- neighbor_mask [batch, n_spatial, 6] float32 — 1 real / 0 missing-or-padding; multiplies gathered messages before the degree-axis reduce; shares n_spatial ragged axis with spatial. NEW ONNX input.
- Not stored in shards (HARD CONSTRAINT honored): derived in Python (hexgraph) from emitted coords ch13/14 + global map-dim slots, and rebuilt by the JVM from the live TileMap at inference. Only NEW stored data is the 2 coord channels (tile_x,tile_y) added by C1 to spatial + 3 map-dim scalars in the existing global tensor.

## Lockstep sites
- python/unciv_train/contract.py:18 CONTRACT_VERSION_RICH 2->3 (mirror SampleSchema.OnnxContract.CONTRACT_VERSION_RICH and OnnxPolicy gate at OnnxPolicy.kt:59)
- SampleSchema.kt:22 VERSION 2->3 (C1) — schema_version stamped by export_onnx must be 3; dataset expected_version=3
- Spatial channel count: Kotlin SampleSchema.SPATIAL_CHANNELS (->15, C1) is SSOT; Python reads len(spatial_channels) in contract.token_specs_from_schema; dataset._rich_step_blocks n_channels; features coord-channel index (width-2,width-1); export dummy width — all derive from the schema, NO hardcoded 13/15
- contract.GLOBAL_MAPDIM_OFFSET=5 must match C1 buildGlobal appending wrap_radius/world_wrap/shape at head slots 5,6,7 (after the 5 existing head scalars, before demographics agg)
- ONNX positional input order: global, acting_civ, spatial, spatial_mask, neighbor_index, neighbor_mask, own_units, ... — frozen across export_onnx names list, _RichPolicyOnly zip, parity fixture write order, and the JVM OnnxPolicy build order (C3)
- neighbor degree = 6 (contract.NEIGHBOR_DEGREE) must match OFFSETS length and the JVM inference builder degree
- hexgraph.OFFSETS order must equal the JVM getTilesAtDistance(1) neighbor order used by the C3 inference builder (parity test enforces byte-identical neighbor_index)
- neighbor_index dtype int64 in export dummy AND in JVM OnnxTensor build (C3) AND in features output
- _TOKEN_WIDTH_FALLBACK deleted in Python ⇒ FALLBACK_WIDTH in OnnxPolicy.kt:150-151 (C3) must likewise be made fail-loud vs schema for consistency

## Export safety
opset-17 safe. The adjacency tensors are consumed in the StructuredPolicyValueNet via Gather (neighbor rows -> [N,6,C]) + Mul (by neighbor_mask) + ReduceSum/ReduceMean over the fixed degree-6 axis — all core ONNX <= opset 13 (research-notes Q1). NO scatter_add/index_add/scatter_reduce anywhere in C2 (those are the broken/silently-wrong paths). neighbor_index MUST export as int64 (ORT Gather index dtype) — handled by typed dummy in export_rich and an int64 cast on the sample_inputs override; a float32 neighbor_index would make ORT reject the model. SENTINEL index 0 is a valid Gather row so even missing neighbors (mask 0) never read OOB (FND-0025; also asserted in build_neighbor_graph). Dynamic axes: neighbor_index/mask share the spatial ragged axis n_spatial (axis 1) + batch (axis 0) — identical treatment to the existing v2 spatial token set, no new dynamic-axis op-support risk (research-notes Q3). Export smoke test on the SMALL rung (test_contract_mismatch.py) runs onnx.checker.check_model + an ORT InferenceSession with a hand-built degree-6 feed BEFORE any full-scale run (FND-0008/0023 gate). The StructuredPolicyValueNet keeps the FROZEN seam forward(inputs:dict)->(tech,policy,tanh(value)) and the same INPUT_GLOBAL/INPUT_ACTING attrs, so _RichPolicyOnly and the export plumbing are unchanged in structure.

## Determinism/provenance
Determinism: build_neighbor_graph is a pure function of (coords, wrap_radius, world_wrap); coord_to_row is built by iterating rows 0..N-1 so the last writer wins on duplicate coords (coords are unique per tile, so no ambiguity). OFFSETS is a fixed tuple; output is fully deterministic and order-stable, satisfying the parity 1e-4 contract by construction (indices are exact integers). Provenance: schema_version bumps 2->3 (C1 SampleSchema.VERSION) and CONTRACT_VERSION_RICH 2->3; adding tile_x/tile_y to SampleSchema.SPATIAL_CHANNELS auto-changes the ruleset fingerprint (Vocab.canonicalSections hashes 'schema:spatialChannels', codebase-scan.md:128), so old v2/v3-mismatched shards are REFUSED fail-loud by dataset.load_trajectories' ProvenanceError (dataset.py:76-82) and by the new contract-mismatch test. The fail-loud token_specs_from_schema additionally refuses a schema that omits spatial_channels (no silent revert to a stale width). export_onnx stamps META_CONTRACT_VERSION=3 + META_INPUT_NAMES including neighbor_index/mask so the JVM provenance/inventory gate (OnnxPolicy.kt:52-74) refuses a model whose input set or contract version disagrees.

## Open questions
- WRAP-RADIUS CONTRACT WITH C1 (highest priority): getIfTileExistsOrNull uses effective wrap radius = mapSize.width/2 for rectangular, else mapSize.radius. C1's decisions.md says it emits 'radius, worldWrap, shape'. If C1 emits RAW mapSize.radius, Python cannot recover width/2 for rectangular maps. RESOLUTION I assume: C1 emits a single pre-resolved 'wrap_radius' float at global head slot 5. If C1 instead emits raw radius+width+shape, hexgraph.build_neighbor_graph_from_spatial must branch on shape==rectangular(ordinal) to pick width/2 — needs the shape-ordinal->name mapping and width also emitted. Must confirm with C1 which fields land and at which head slots.
- GLOBAL head slot layout: I assume map-dims land at slots 5,6,7 (after the 5 fixed head scalars, before the demographics agg). If C1 appends them AFTER the agg block instead, GLOBAL_MAPDIM_OFFSET must be global_w-3 (computable from dims) — confirm placement with C1 so the constant is robust (prefer a fixed pre-agg offset, or surface map-dims as a named schema field rather than positional).
- Fixture dtype for neighbor_index in the JVM parity text format: I write int64 indices as floats (lossless for small exact integers) reusing _fixture_line_set. Cleaner is an explicit int fixture line type parsed on the JVM as long — decide with C3 whether parityRunRich's fixture parser gains an int set type.
- JVM-side adjacency cross-check: the always-on adjacency-parity test (test_hexgraph.py) is Python-only vs hand-computed reference. An optional stronger guard dumps tile.neighbors zeroBasedIndex from a real JVM game (new 'parity-adjacency' selfPlay entry) and diffs vs hexgraph — decide with C3 whether to add that JVM entry now or defer.
- StructuredPolicyValueNet exact forward consumption of neighbor_index/mask (the Gather+masked-reduce) is C4/D3-model's deliverable; C2 only guarantees the input contract (names, dtypes, shapes, axes). Confirm C4 reads inputs['neighbor_index']/['neighbor_mask'] with these exact names.

## Risks
- neighbor_index exported as float32 (current export loop is float-only) -> ORT rejects Gather indices at load. -> Typed int64 dummy in export_rich for neighbor_index; .long() cast on sample_inputs override; export smoke test asserts input dtype int64 (FND-0008).
- OFFSETS order diverges from the JVM inference neighbor order -> per-slot neighbor_index differs -> GNN messages mismatch (parity fails or, worse, silently different at inference where there is no parity check). -> Adopt the exact getTilesAtDistance(1) emission order as canonical OFFSETS; adjacency-parity test + extended rich parity test (neighbor tensors in fixture) lock it; lockstep_site documents C3 must use the same order.
- Batch padding mismatch: neighbor tensors padded to a different max-N than the spatial token set -> shape mismatch / wrong rows in the GNN. -> Compute maxn once from spatial arrays and reuse for both; assert out['spatial'].shape[1]==neighbor_index.shape[1].
- Wrap-radius wrong for rectangular maps (width/2 vs radius) -> wrong wrapped neighbors at the east/west seam, silently degraded GNN. -> Require C1 to emit pre-resolved wrap_radius (open question); adjacency-parity test includes both world-wrap branches (C and D) on a rectangular-shaped fixture.
- Removing _TOKEN_WIDTH_FALLBACK breaks callers that relied on the silent default (e.g. a schema mid-migration). -> Grep-confirmed only token_specs_from_schema referenced it; fail-loud is the intended behavior (FND-0007/0011); contract-mismatch test asserts the raise; schema emission is authoritative (C1).
- Map-dim head-slot offset drift between C1 emission and Python read -> reads demographics as wrap_radius -> nonsense graph but no crash. -> Fix GLOBAL_MAPDIM_OFFSET as a single shared constant + open question to confirm placement; consider surfacing map-dims as a named schema field instead of positional (flagged).
- Coords stored as float32 then int()-cast for the coord_to_row keys could mismatch on negative coords / rounding. -> Coords are exact small integers stored as float32 (lossless); use int(round(.)) on reads; adjacency-parity test includes negative-coordinate tiles.

## VERDICT
```json
{
  "cluster": "C2 \u2014 Python contract/features/dataset/export + Python adjacency builder + parity (D9-Python, D3-Python, D10)",
  "export_safe": true,
  "lockstep_complete": false,
  "seam_preserved": true,
  "parity_feasible": true,
  "determinism_ok": true,
  "issues": [
    {
      "severity": "critical",
      "issue": "FALSE PREMISE \u2014 the spatial block is u8, not float32. Featurizer.kt:129 emits `fixedU8(\"spatial\", buildSpatial(...))`; SampleSchema.kt:67 DT_U8=\"<u1\"; Featurizer:16 comment \"the spatial plane use u8\"; reader.py:43/115/129/138 decodes spatial as np.uint8. The design (and the HARD CONSTRAINT it quotes: 'Coords ride as 2 plain float32 spatial channels (13->15)... (shards are float32; resolved)') is wrong: global/acting_civ are fixedF32 but SPATIAL is u8. If C1 appends tile_x/tile_y as spatial channels 13/14 they are stored as UNSIGNED BYTES (0..255). Unciv hex coords are centered and routinely NEGATIVE (a u8 maps -1 -> 255) and on rectangular/large maps the raw x can exceed 255 (wraps mod 256). Every such coordinate is corrupted on storage, so hexgraph.build_neighbor_graph (which does int(round(float(coord)))) computes the WRONG neighbor_index/neighbor_mask for exactly the edge/world-wrap tiles the GNN and the parity test target. decisions.md FND-0002/0035 EXPLICITLY left this unresolved: 'If shards are u8 -> documented bias offset.' The design never resolves it and asserts it away.",
      "fix": "Resolve the coord-storage dtype as a HARD lockstep contract with C1 before C2 is built. Option A (preferred for fidelity): store tile_x/tile_y in a NEW fixedF32 (or DT_I32) spatial-coord block separate from the u8 spatial planes, OR change the spatial block dtype to i32 (4x shard bloat \u2014 the comment at Featurizer:16 chose u8 specifically to avoid this). Option B (FND-0035 'documented bias offset'): C1 emits x+BIAS, y+BIAS clamped into 0..255 with BIAS=wrap_radius, and hexgraph subtracts BIAS \u2014 but this FAILS for any map whose coord span exceeds 256, so it must be guarded by a fail-loud range assert at emission AND in hexgraph. Whichever is chosen, hexgraph.build_neighbor_graph and features.build_rich_single/batch must decode coords through that contract (subtract bias / read the f32 coord block), NOT read raw u8 floats as the design currently does. Add an explicit decode step + a fail-loud assert that every (x,y) is integer-exact and in-range."
    },
    {
      "severity": "critical",
      "issue": "TRAIN/INFERENCE ADJACENCY ASYMMETRY MASKED BY THE PARITY TEST. At INFERENCE the JVM builds neighbor-index from the live TileMap (OnnxPolicy buildRichTensors feeds spatial as float32 losslessly, OnnxPolicy.kt:146) \u2014 correct. At TRAINING, hexgraph reads coords back from the u8-quantized shard \u2014 corrupted (see above). The proposed parity test (test_parity.py:94-103 extended) hand-builds `spatial` as float `rng.standard_normal((7,5))` and feeds it straight into build_rich_single \u2014 it NEVER round-trips through the u8 shard encoder/reader. So the parity assertion (atol 1e-4) stays GREEN while real training shards carry corrupted adjacency. The design claims parity 'guards' the adjacency; it only guards Python-vs-JVM tensor-build symmetry on un-quantized coords, not the storage-decode that training actually uses.",
      "fix": "Add a SHARD-ROUNDTRIP test that writes a real (or reader-emulated) u8 spatial block with negative + >255 coords, reloads via unciv_dataplane.reader, runs _rich_step_blocks + hexgraph, and asserts neighbor_index/neighbor_mask equal the live-TileMap reference. This is the only test that exercises the coord-storage contract. The hexgraph adjacency-parity test (test_hexgraph.py, \u00a76b) and the extended logits parity (\u00a76a) must additionally feed coords through the chosen decode path (bias-subtract or f32 coord block) so they actually cover the storage representation, not idealized floats."
    },
    {
      "severity": "major",
      "issue": "WRONG SOURCE CLAIM about reusing tile.neighbors. The design note in \u00a70 says 'the JVM inference builder can reuse tile.neighbors directly if desired (see C3 note)' and lists OFFSETS as 'the order the live JVM neighbors actually materialize.' But Tile.neighbors = getTilesAtDistance(1).toList() and getTilesAtDistance ends with `.filterNotNull()` (TileMap.kt:285; Tile.kt:115). tile.neighbors is NULL-FILTERED: a 4-neighbor edge tile yields a length-4 list with NO slot/offset information, so it CANNOT produce the fixed degree-6 per-slot neighbor_index/neighbor_mask. C3 reusing tile.neighbors would silently misalign slots and produce a different neighbor_index than Python (or a ragged tensor).",
      "fix": "Correct the design note: the C3 JVM inference builder must replicate the UNFILTERED degree-6 enumeration (call getIfTileExistsOrNull for each of the 6 OFFSETS in order, emitting SENTINEL+mask0 for null), NOT tile.neighbors. The OFFSETS order derivation itself is correct; only the 'reuse tile.neighbors' shortcut is wrong. Flag as a binding constraint on C3."
    },
    {
      "severity": "major",
      "issue": "wrap_radius cross-cluster contract is unresolved, not just an open question. getIfTileExistsOrNull (TileMap.kt:383-385) uses radius = (shape==rectangular) ? mapSize.width/2 : mapSize.radius. The design ASSUMES C1 emits a single pre-resolved wrap_radius float at global head slot 5, but decisions.md only says C1 emits 'radius, worldWrap, shape'. If C1 emits raw mapSize.radius, Python cannot recover width/2 for rectangular maps and every east/west world-wrap neighbor is wrong (silently \u2014 no crash). The design correctly raises this as open-question #1 but then builds the whole hexgraph API around the unconfirmed assumption.",
      "fix": "Make 'C1 emits pre-resolved effective wrap_radius (Kotlin computes if(shape==rectangular) mapSize.width/2 else mapSize.radius) as one fixedF32 global-head scalar' a BLOCKING contract item, not an open question. hexgraph must take wrap_radius directly. Add a rectangular-shape world-wrap case (both branch C and branch D) to the adjacency-parity test with an actual width/2 != radius map to prove it."
    },
    {
      "severity": "major",
      "issue": "GLOBAL_MAPDIM_OFFSET=5 is a positional guess into a fixedF32 block whose layout is owned by C1. buildGlobal (Featurizer.kt:159-165) currently emits exactly 5 head scalars (turns, era, tileCount, knownMajors, aliveMajors) then `agg` (demographics, size*3). If C1 appends the map-dims AFTER agg (or reorders), reading slot 5 returns a demographics value as wrap_radius -> nonsense graph, no crash. The constant is fragile and duplicated as a Python literal with no schema-driven cross-check.",
      "fix": "Prefer surfacing the map-dim slots as a NAMED schema field (e.g. add 'mapDims' indices to schema.json from DataPlaneHooks) and read them by name, instead of a positional GLOBAL_MAPDIM_OFFSET literal. If positional is kept, pin it as a hard lockstep_site with C1's exact buildGlobal edit (insert at index 5, before agg) AND add a runtime assert (e.g. wrap_radius is a small non-negative integer-valued float, world_wrap in {0,1}) to fail loud on slot drift."
    },
    {
      "severity": "minor",
      "issue": "Schema key casing: the design's fail-loud rewrite of token_specs_from_schema references spatial_channels in one raise message but the actual schema key is camelCase 'spatialChannels' (DataPlaneHooks.kt:171; emitted as \"spatialChannels\":[...]). The original code's `sch.get(\"spatial_channels\") or sch.get(\"spatialChannels\")` handles both; the proposed rewrite keeps the `or`, so functionally fine, but the error string and any new readers must not assume snake_case.",
      "fix": "Keep the `or sch.get(\"spatialChannels\")` in the rewritten check (the design does). Make the raise message name both keys so a future maintainer isn't misled."
    },
    {
      "severity": "minor",
      "issue": "Fixture int64 round-trip for neighbor_index via _fixture_line_set (float text). Writing int64 indices as floats is lossless for small exact integers, but the design itself flags this as fragile. With the larger fix (shard-roundtrip), neighbor_index may exceed float32 exact-integer range only above 2^24 (N never reaches that), so it is safe; just document and assert exactness on parse.",
      "fix": "Acceptable as-is for N<<2^24; add an assert in the JVM fixture parser that the parsed float equals its rounded long. Decide with C3 whether to add an explicit int fixture line type (design open-question #3)."
    },
    {
      "severity": "minor",
      "issue": "Padding-consistency invariant relies on recomputing maxn identically to _pad_token_set. _pad_token_set uses maxn=max(1, max(counts)) (features.py:21). The design's build_rich_batch computes maxn=max(1, max(a.shape[0] for a in spatials)) separately, then later calls _pad_token_set on spatial again. If a step has zero spatial rows (shouldn't happen but is possible for a defensive/empty obs) the two maxn agree (both clamp to 1), but the neighbor tensors would be [.,1,6] of zeros vs spatial [.,1,spatial_w] padding \u2014 consistent. Risk is only if the two code paths drift.",
      "fix": "Compute maxn ONCE from the spatial arrays and pass it into both the neighbor padding and (a maxn-parameterized) _pad_token_set for spatial; or keep the design's assert out['spatial'].shape[1] == neighbor_index.shape[1] as a hard post-condition. The design already lists this assert \u2014 keep it."
    }
  ],
  "verdict": "REVISE",
  "corrected_notes": "Anchors verified and CORRECT: contract.py:18 (CONTRACT_VERSION_RICH=2), :29 (RICH_TOKEN_NAMES), :80-82 (_TOKEN_WIDTH_FALLBACK \u2014 grep-confirmed only used at contract.py:95,101 inside token_specs_from_schema, safe to delete), :85-102 (token_specs_from_schema); dataset.py:113 (reshape(-1,13) is the ONLY hardcoded-13 site), :106-121 (_rich_step_blocks), :163 (load_trajectories rich call), :37 RICH_TOKEN_BLOCKS; features.py:33-48/51-69; export_onnx.py:109-117 token loop + 128-138 metadata; OnnxPolicy.kt:150-151 FALLBACK_WIDTH (real lockstep site), :66-74 rich-input inventory gate. run_loop caller is at run_loop.py:204 with schema available at :195-198, so threading expected_spatial_channels is feasible (design correctly defers to C4).\n\nOFFSETS DERIVATION IS CORRECT: traced getTilesAtDistance(origin,1) at TileMap.kt:254-285 by hand for distance=1: order is [(-1,-1),(+1,+1),(0,-1),(0,+1),(+1,0),(-1,0)] exactly as the design states. getIfTileExistsOrNull (TileMap.kt:376-398) branch order A(direct)/B(!worldWrap null)/C(x+r,y-r)/D(x-r,y+r) is replicated correctly by hexgraph.resolve.\n\nROW MAPPING IS CORRECT: buildSpatial (Featurizer.kt:233-262) iterates tileList writing out[tile.zeroBasedIndex*channels]; zeroBasedIndex is dense over present tiles (TileMap.kt:610-617 values.withIndex()), so spatial row r == tile with zeroBasedIndex r, and coord_to_row keyset == present-tile set == getOrNull's present-coord test. Valid.\n\nEXPORT SAFETY IS SOUND: opset-17 Gather+Mul+ReduceSum over fixed degree-6 axis, no scatter; neighbor_index int64 dummy + dynamic axes sharing n_spatial are correct; the int64 cast fix in the sample_inputs override is needed and correct. seam forward(dict)->(tech,policy) untouched, _RichPolicyOnly unchanged. Determinism/provenance fine: build_neighbor_graph is pure; adding tile_x/tile_y to SPATIAL_CHANNELS changes the fingerprint via Vocab.kt:97 'schema:spatialChannels', so stale shards are refused (dataset ProvenanceError). The global block IS fixedF32 (Featurizer.kt:120), so wrap_radius/world_wrap/shape store losslessly \u2014 only the SPATIAL coords have the u8 problem.\n\nTHE BLOCKER: the entire design rests on coords being float32-faithful in the spatial block. They are NOT \u2014 spatial is fixedU8 / DT_U8 '<u1'. This is the unresolved FND-0002/0035 fork in decisions.md:45-48 ('If shards are u8 -> documented bias offset'), which the design wrongly treats as already resolved. C2 cannot be built until C1's coord-storage contract (separate f32/i32 coord block, or documented+range-guarded bias offset) is locked, and hexgraph/features/dataset must decode coords through that contract rather than reading raw u8 floats. A shard-roundtrip test (not just the float-fed parity test) is mandatory to prove it."
}
```
