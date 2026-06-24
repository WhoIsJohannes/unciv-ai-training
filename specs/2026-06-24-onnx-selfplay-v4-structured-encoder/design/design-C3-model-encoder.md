# Design — C3-model-encoder

## Summary
A new nn.Module StructuredPolicyValueNet replaces RichPolicyValueNet behind the FROZEN seam forward(inputs:dict)->(tech_logits, policy_logits, tanh(value)), keeping INPUT_GLOBAL/INPUT_ACTING and the same construction site in train_actor_critic_rich. Per-token integer fields are split into categorical (embedded) vs numeric (scalar) slices via a per-token-set FieldSpec derived from schema; categorical embedding counts come from a new emitted vocabCounts schema block (never hardcoded, row-0 sentinel). Spatial tiles run a degree-6 gather-GNN (Gather neighbor rows into [B,N,6,C], message MLP, mask-multiply, masked ReduceSum/Mean) using two NEW model input tensors neighbor_index [B,N,6] (int64) and neighbor_mask [B,N,6] (float32) built in Python from emitted coords+map-dims and at inference by OnnxPolicy from the live TileMap; entity sets run hand-rolled pre-LN self-attention; a single-query cross-attention pools GNN nodes + entity tokens into a fixed context; a split trunk feeds separate policy-late {tech,policy} and value-late {value} heads. Every op is opset-17 core (Gather/Mul/ReduceSum/MatMul/Softmax/LayerNorm/Add), all masked softmaxes/pools mirror masked_pool's NaN guards (clamp(count,1), -inf fill + isfinite fallback) so an all-padding row can never produce NaN. Three constructor-arg rungs (small/medium/large) implement the D7 ladder.

## Detailed design
# StructuredPolicyValueNet — full design (C3)

All anchors are the worktree `/Users/j/Unciv-onnx-selfplay-loop`. The new module lives in
`python/unciv_train/model.py` alongside `RichPolicyValueNet` (model.py:79-113) and reuses
`masked_pool` (model.py:50-63), `_small_init_value_head` (model.py:20-26).

## 0. The frozen seam + construction site (what must NOT change)
- Seam: `forward(inputs: dict) -> (tech_logits, policy_logits, torch.tanh(value))`, attrs
  `INPUT_GLOBAL="global"`, `INPUT_ACTING="acting_civ"` (model.py:88-89,108-113).
- Construction site: `train.py:262` `net = RichPolicyValueNet(dims, token_specs)` inside
  `train_actor_critic_rich` (train.py:240-275). v4 swaps this to
  `StructuredPolicyValueNet(dims, token_specs, vocab_counts, field_specs, **rung)`; `forward_fn`
  (train.py:268) and `_optimize_actor_critic` (train.py:115-194) are UNTOUCHED.
- Export wrapper `_RichPolicyOnly` (export_onnx.py:32-44) already reassembles the dict from
  positional tensors in `names` order and drops value — works unchanged for the new module as long
  as the two new tensors are appended to `names` (see §3.4 / Cluster handoff).

## 1. D2 — Embeddings (counts NEVER hardcoded; row-0 sentinel)

### 1.1 Where the counts come from (NEW schema emission required)
Today `schema.json` (DataPlaneHooks.kt:144-174) emits `spatialChannels` + `layout` + caps, but
**not vocab cardinalities**. `contract.token_specs_from_schema` (contract.py:85-102) only yields
per-token *widths*, not vocab counts. Embedding `num_embeddings` therefore needs a new source. The
counts already exist deterministically in `Vocab` (Vocab.kt:51-58 + `size(category)` 25) and drive
the fingerprint via `canonicalSections` (Vocab.kt:80-100), so adding them to the header is free and
auto-fingerprinted-safe (no new fingerprint coupling beyond what D1 already triggers).

**Edit DataPlaneHooks.buildHeaderJson (DataPlaneHooks.kt:171-172):** insert a `vocabCounts` object:
```kotlin
""""vocabCounts":{""" +
    """"terrain":${vocab.size(Vocab.TERRAINS)},"resource":${vocab.size(Vocab.RESOURCES)},""" +
    """"improvement":${vocab.size(Vocab.IMPROVEMENTS)},"religion":${vocab.size(Vocab.RELIGIONS)},""" +
    """"era":${vocab.size(Vocab.ERAS)},"building":${vocab.buildingCount},"unit":${vocab.unitCount},""" +
    """"nation":${vocab.nationCount},"promotion":${vocab.promotionCount}},""" +
```
(Featurizer holds the `Vocab`; thread `vocab` into `buildHeaderJson` — it is already constructed in
the recorder path; if not in scope, pass `vocab.size(...)` results in via a small data class. Confirm
call site during build — see open questions.)

**Python side — new `contract.vocab_counts_from_schema(schema_path) -> dict[str,int]`** (add near
contract.py:85), reading `sch["vocabCounts"]`; **fail-loud** (no silent fallback) per FND-0007/0011:
```python
def vocab_counts_from_schema(schema_path):
    sch = _schema(schema_path)
    vc = sch.get("vocabCounts")
    if not vc:
        raise ValueError("schema.json missing 'vocabCounts' (regenerate shards on contract v3)")
    return {k: int(v) for k, v in vc.items()}
```

### 1.2 Embedding tables (StructuredPolicyValueNet.__init__)
`num_embeddings = count + 1` (row 0 = the `+1`/0-unknown sentinel the Featurizer already encodes:
SPATIAL_CHANNELS comments, e.g. terrain `+1; 0=unknown`). Shared tables per the constraint:
```python
class _Embeds(nn.Module):
    def __init__(self, vc: dict[str,int], embed_dim: int):
        super().__init__()
        E = embed_dim
        # terrain base + feature SHARE one table (both index vocab.terrain): count+1
        self.terrain = nn.Embedding(vc["terrain"] + 1, E)
        self.resource = nn.Embedding(vc["resource"] + 1, E)
        self.improvement = nn.Embedding(vc["improvement"] + 1, E)
        # owner_slot & unit_owner_slot SHARE a slot table size 42 (maxCivTokens 40 + 2):
        #   0 none/unknown, 1..40 civ, 41 self (Featurizer emits 255 for self -> reindex 255->41)
        self.slot = nn.Embedding(42, E)
        self.unit_type = nn.Embedding(5 + 1, E)        # unitTypeCat 0..4 -> +1 to keep 0=sentinel
        self.road = nn.Embedding(3 + 1, E)             # roadStatus.ordinal 0..2 (None/Road/Railroad)
        self.religion = nn.Embedding(vc["religion"] + 1, E)
        self.era = nn.Embedding(vc["era"] + 1, E)
        # current_construction single table: buildingCount + unitCount + 1 (consistent w/ D1.4 fix)
        self.construction = nn.Embedding(vc["building"] + vc["unit"] + 1, E)
```
Notes:
- **Reindex 255->41 for slot fields** must happen in `forward` before lookup (the raw float channel
  carries 255 for self). Do it with a `torch.where(x == 255, 41, x)` then `.clamp(0, 41).long()`.
  This clamp also makes the embedding lookup OOB-safe (mirrors the Gather bounds discipline,
  FND-0025). `unit_type` raw is 0..4, store as-is (no +1) — but then `num_embeddings = 5` and there
  is no sentinel collision because `unit_present` channel gates it; keep 5 (NOT 5+1) to match the
  Kotlin 0..4 (channel 11 doc) — **decision: unit_type table = 5, no sentinel** (0 == "none" is a
  real category there, see open questions). `road` table = 3 (ordinals 0..2), no sentinel.
- `embed_dim` is a constructor arg (D7 ladder).

### 1.3 Per-token-set FieldSpec (which slice is categorical vs numeric)
A token set is a `[B, N, width]` float tensor. Each set declares, per column, either `("num",)` or
`("emb", table_name)`. Numeric columns are concatenated as raw scalars; categorical columns are
looked up and concatenated. Built from the schema layout + the fixed channel/field order (these
orders are the SSOT in Kotlin and known to Python):

- **spatial (15 ch after D1.1):** col0 visibility=num(raw 0/1/2 preserved, FND surprise 7),
  1 terrain_base=emb terrain, 2 terrain_feature=emb terrain (SHARED), 3 resource=emb resource,
  4 road=emb road, 5 river=num, 6 is_city_center=num, 7 owner_slot=emb slot, 8 improvement=emb
  improvement, 9 unit_present=num, 10 unit_owner_slot=emb slot, 11 unit_type=emb unit_type,
  12 unit_health_bucket=num, 13 tile_x=num, 14 tile_y=num.
- **own_units/opp_units (9 after D1.3):** 0 presence=num, 1 isOwn=num, 2 ownerSlot=emb slot,
  3 unitTypeCat=emb unit_type, 4 health=num, 5 capital_dx=num, 6 capital_dy=num, 7 promo_count=num,
  8 tile_index=num (tile_index is fed to the GNN cross-link as a *numeric* feature; it is NOT
  embedded — it is a pointer, see §3.5).
- **own_cities/opp_cities (17 after D1.3):** 0 presence=num,1 isOwn=num,2 ownerSlot=emb slot,
  3 pop=num,4 defense=num,5 health=num,6 air=num,7 majority_religion=emb religion,
  8 resistance=num,9 puppet=num,10 razed=num,11 hasSpy=num,12 current_construction=emb
  construction,13 builtBuildings=num,14 tile_index=num (D1.3 add at end). (Width 16->17: the
  D1.3 city tile-index is appended; map exact column order from writeCityToken order in
  Featurizer.kt:185-208 during build.)
- **civ_tokens (84):** all numeric (FairOpponentModel GnK token is already rank/bucket/count floats,
  light scan line 79-81) — no embeddings; passed as scalars. (Surprise 1: 84 is the FairOpponent
  feature width, distinct from the 42 slot table.)

`field_specs: dict[str, list[tuple]]` is constructed once (a helper `build_field_specs(token_specs)`
in contract.py keyed by the fixed orders above; widths cross-checked against `token_specs` and
asserted equal — fail-loud if drift). The per-token "projected" width after embedding =
`n_numeric*1 + n_categorical*embed_dim`, fed into a per-token-set input projection `Linear(proj_w, C)`.

### 1.4 Embed-then-project (per token set)
```python
def _embed_tokens(self, name, x):            # x: [B, N, width]
    spec = self.field_specs[name]
    cols = []
    for j, s in enumerate(spec):
        v = x[..., j]                        # [B, N]
        if s[0] == "num":
            cols.append(v.unsqueeze(-1))     # [B, N, 1]
        else:
            idx = v
            if s[1] == "slot":
                idx = torch.where(idx == 255, torch.full_like(idx, 41.0), idx)
            idx = idx.clamp(0, self.embeds.tables[s[1]].num_embeddings - 1).long()
            cols.append(self.embeds.tables[s[1]](idx))   # [B, N, E]
    return torch.cat(cols, dim=-1)           # [B, N, proj_w]
```
(`self.embeds.tables` = a `nn.ModuleDict` aliasing the shared tables: `terrain`,`resource`,
`improvement`,`slot`,`unit_type`,`road`,`religion`,`era`,`construction`.)

## 2. D3 — Gather-GNN over the spatial tile graph (opset-17 safe)

### 2.1 Inputs (two NEW model input tensors — justified)
The GNN needs, per node, its 6 hex-neighbor row indices + a presence mask. These CANNOT be folded
into `spatial` because (a) they are int indices into the *node axis* (a different shape `[B,N,6]`),
(b) they are derived from coords+map-dims (Python at train, JVM at inference), not stored per-channel.
**Decision: two NEW ONNX input tensors**:
- `neighbor_index` : `int64 [B, N, 6]` — for each node, the row index (0..N-1) of its 6 neighbors;
  self/own-row used as a safe filler for missing edges (clamped 0..N-1 so ORT Gather never reads OOB,
  FND-0025), with the corresponding mask 0.
- `neighbor_mask`  : `float32 [B, N, 6]` — 1 if that neighbor edge exists, 0 otherwise (map edge,
  unexplored handled by the mask only — adjacency is geometric, see §3.5).

They are NOT folded (rationale above) and they ARE new ONNX inputs. This is the ONE place v4 adds
tensor names; it is unavoidable given the gather realization (research-notes Q1/Q3). The contract
RICH_TOKEN_NAMES stays unchanged (these are graph tensors, not token sets), but they get their own
dynamic axes `{0:"batch", 1:"n_spatial"}` (the second axis must equal spatial's N — see export §3.4).

### 2.2 Layer (Gather + masked reduce; ONLY Gather/Mul/ReduceSum/MatMul/Add)
```python
class _GatherGNNLayer(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2*C, C), nn.ReLU(), nn.Linear(C, C))
        self.upd = nn.Sequential(nn.Linear(2*C, C), nn.ReLU())
        self.norm = nn.LayerNorm(C)
    def forward(self, h, nbr_idx, nbr_mask):    # h:[B,N,C] idx:[B,N,6] mask:[B,N,6]
        B, N, C = h.shape
        # gather neighbor features: batched gather along node axis
        idx = nbr_idx.reshape(B, N*6)                     # [B, N*6]
        gathered = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, C))  # [B, N*6, C]
        nbr = gathered.reshape(B, N, 6, C)                # [B,N,6,C]
        self_exp = h.unsqueeze(2).expand(-1, -1, 6, -1)   # [B,N,6,C]
        m = self.msg(torch.cat([self_exp, nbr], dim=-1))  # [B,N,6,C]
        m = m * nbr_mask.unsqueeze(-1)                     # mask padding edges (Mul)
        deg = nbr_mask.sum(dim=2, keepdim=True).clamp(min=1.0)   # [B,N,1] NaN-guard (==masked_pool)
        agg = m.sum(dim=2) / deg                           # ReduceSum / safe mean -> [B,N,C]
        out = self.upd(torch.cat([h, agg], dim=-1))        # [B,N,C]
        return self.norm(h + out)                          # residual + LN (pre/post per rung)
```
- `torch.gather` exports to ONNX `GatherElements`/`Gather` at opset 13 — safe. NO scatter anywhere
  (research-notes Q1). Degree axis is a FIXED 6 so the reduce is a plain `ReduceSum` over a static
  axis (research Q1/Q3). `deg.clamp(min=1)` guarantees no /0 even for an all-isolated node (mirrors
  masked_pool safe_count, model.py:58).
- `L` layers x `C` channels (D7). Input projection `Linear(proj_w_spatial, C)` first.
- **Output is per-node `[B,N,C]` (NOT pooled)** — fed as cross-attention keys/values (§4).

### 2.3 Complexity (FND-0012/0026): O(N*6) with N≈1261 tiles; no O(N^2) over tiles.

## 3. D4/D5 — Hand-rolled attention (no MHA/SDPA; opset-17 safe)

### 3.1 Primitive (Linear Q/K/V + matmul + scale + masked softmax + matmul + out-proj, pre-LN)
```python
def _masked_softmax(scores, mask):           # scores:[...,Lq,Lk], mask:[...,Lk] (1 keep,0 pad)
    neg = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(mask.unsqueeze(-2) == 0, neg)   # additive -inf-equivalent
    w = torch.softmax(scores, dim=-1)
    # NaN guard: an all-padding key row -> uniform-but-zeroed; re-zero via mask so it never NaNs
    w = w * mask.unsqueeze(-2)                                  # kill any residual on pad keys
    denom = w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return w / denom                                           # safe renorm, never NaN

class _MHA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h=heads; self.dk=dim//heads
        self.q=nn.Linear(dim,dim); self.k=nn.Linear(dim,dim); self.v=nn.Linear(dim,dim)
        self.o=nn.Linear(dim,dim)
    def _split(self, x): B,L,_=x.shape; return x.view(B,L,self.h,self.dk).transpose(1,2)
    def forward(self, q_in, kv_in, kv_mask):          # q:[B,Lq,D] kv:[B,Lk,D] mask:[B,Lk]
        q=self._split(self.q(q_in)); k=self._split(self.k(kv_in)); v=self._split(self.v(kv_in))
        scores=torch.matmul(q,k.transpose(-1,-2))/ (self.dk ** 0.5)        # [B,h,Lq,Lk]
        w=_masked_softmax(scores, kv_mask.unsqueeze(1))                    # mask broadcast over heads
        ctx=torch.matmul(w,v).transpose(1,2).reshape(q_in.shape[0], q_in.shape[1], -1)
        return self.o(ctx)
```
All ops: Linear(MatMul+Add), MatMul, Div, Softmax, Mul, Add — opset-17 core. NO `nn.MultiheadAttention`,
NO `F.scaled_dot_product_attention` (research Q2). Pre-LN wrapper: `x = x + mha(LN(x), LN(x), mask)`.

### 3.2 D4 — self-attention per entity set
For each of `own_units, opp_units, own_cities, opp_cities, civ_tokens`: project embedded tokens
(`_embed_tokens` -> `Linear(proj_w, attn_dim)`), run `A_e` pre-LN self-attention blocks with the
set's **presence mask** (the existing `<name>_mask` input, [B,N] from OnnxPolicy.tokenTensors /
features._pad_token_set). Self-attention is q=kv=the set, so `_masked_softmax` uses the same mask.
Output: per-set per-token embeddings `[B, M_e, attn_dim]` (NOT pooled here — kept for cross-attn).
Self-attention only over **small** entity sets (M up to caps: maxOwnUnits etc.), O(M^2) tiny.

### 3.3 D5 — single-query cross-attention -> fixed context vector
- Query: `q = Linear([global ‖ acting_civ] -> attn_dim)`, shape `[B, 1, attn_dim]` (ONE query =>
  O(N) not O(N^2), FND-0012).
- Keys/values: concat of GNN nodes `[B,N,attn_dim]` (after a `Linear(C->attn_dim)`) and ALL entity
  tokens `[B, sum M_e, attn_dim]` along the token axis -> `[B, N+sum M_e, attn_dim]`.
- Union mask: concat(spatial_mask, all entity masks) along the same axis. **Spatial nodes have a
  presence mask too**: a node is a valid key iff visibility/explored — but the simplest correct
  choice is mask=1 for all N tiles (they always exist) OR gate by `spatial_mask` if D1 emits one.
  Spatial currently has no `_mask` (it is a fixed full-board set). **Decision: spatial mask = all-ones
  [B,N]** (every tile is a real node; unexplored tiles carry zeroed channels already, model learns
  to down-weight) — confirm vs whether a visibility mask is wanted (open question).
- `A_x` pre-LN cross-attention blocks; the all-padding-row NaN guard in `_masked_softmax` covers the
  degenerate "no keys at all" case (cannot happen since spatial N>=1, but guarded anyway, mirroring
  masked_pool model.py:62).
- Output `cross_ctx = context[:, 0, :]` -> `[B, attn_dim]`.

### 3.4 Export (handoff to the export cluster; stated for completeness)
`export_rich` (export_onnx.py:82-138) must append `neighbor_index`/`neighbor_mask` to `names` AFTER
the token sets, with dummy `torch.zeros(1, n0_spatial, 6, dtype=torch.int64)` /
`torch.ones(1, n0_spatial, 6)` and `dyn` `{0:"batch", 1:"n_spatial"}` so axis-1 equals spatial's.
`_RichPolicyOnly.forward(*tensors)` already zips by `names` (export_onnx.py:41-43) — works as-is.
**Export smoke-test on the SMALL rung FIRST** (FND-0008/0023). The new module's `forward` reads
`inputs["neighbor_index"]`, `inputs["neighbor_mask"]` from the dict.

### 3.5 Adjacency build (handoff to data/inference cluster; constraints I impose on it)
- Python (train): from spatial coords ch13/ch14 (D1.1) + global radius/worldWrap/shape (D1.2), build
  `(x,y)->row` from the spatial rows (row r == tile.zeroBasedIndex == buildSpatial write order,
  Featurizer.kt:233 uses `tile.zeroBasedIndex*channels`), then for each of the 6 clock offsets
  {(1,1),(0,1),(-1,0),(-1,-1),(0,-1),(1,0)} (HexMath.kt:277-285) compute the neighbor (x',y'),
  replicate `TileMap.getIfTileExistsOrNull` world-wrap (TileMap.kt:376-398: try (x',y'); if worldWrap
  try (x'+R,y'-R) then (x'-R,y'+R), R = radius or width/2 for rectangular), map to row or mask 0.
  zeroBasedIndex is ASSIGNMENT-ORDER not a formula (TileMap.kt:610-617) — so the `(x,y)->row` dict
  MUST be built from emitted coords, never computed. This is why coords ride in spatial.
- JVM (inference): OnnxPolicy builds the same `[1,N,6]`/`[1,N,6]` tensors from the live TileMap via
  real `getIfTileExistsOrNull` + `zeroBasedIndex` (no replication of geometry; uses the engine).
- An **adjacency-parity test** (FND-0036) guards Python-vs-Kotlin equality incl. world-wrap + ragged
  edges. These belong to the data/inference clusters; C3 only fixes the tensor contract shape
  ([B,N,6] int64 idx + float mask, self-filler clamped, missing-edge mask 0).

## 4. D6 — Split trunk + heads
```python
shared = nn.Sequential(nn.Linear(cross_ctx_w + global_w + acting_w, trunk_w), nn.ReLU(),
                       nn.Linear(trunk_w, trunk_w), nn.ReLU())
policy_late = nn.Sequential(nn.Linear(trunk_w, trunk_w), nn.ReLU())   # separate from value
value_late  = nn.Sequential(nn.Linear(trunk_w, trunk_w), nn.ReLU())
self.tech_head=nn.Linear(trunk_w,dims.tech_w); self.policy_head=nn.Linear(trunk_w,dims.policy_w)
self.value_head=nn.Linear(trunk_w,1); _small_init_value_head(self.value_head)   # reuse 20-26
```
`forward` tail (matches seam exactly):
```python
body = shared(torch.cat([cross_ctx, g, a], dim=1))
ph = policy_late(body); vh = value_late(body)
return self.tech_head(ph), self.policy_head(ph), torch.tanh(self.value_head(vh))
```
Value head training-only (dropped by `_RichPolicyOnly`, export_onnx.py:43). `torch.tanh` preserved.

## 5. D7 — Ladder (constructor args; 3 rungs)
```python
class StructuredPolicyValueNet(nn.Module):
    INPUT_GLOBAL="global"; INPUT_ACTING="acting_civ"
    def __init__(self, dims, token_specs, vocab_counts, field_specs, *,
                 embed_dim=16, gnn_layers=2, gnn_channels=64, attn_layers=1,
                 attn_heads=2, attn_dim=64, trunk_w=256): ...
    def forward(self, inputs): ...   # signature == seam
RUNGS = {
 "small":  dict(embed_dim=8,  gnn_layers=1, gnn_channels=32, attn_layers=1, attn_heads=2, attn_dim=32,  trunk_w=128),
 "medium": dict(embed_dim=16, gnn_layers=2, gnn_channels=64, attn_layers=1, attn_heads=4, attn_dim=64,  trunk_w=256),
 "large":  dict(embed_dim=24, gnn_layers=3, gnn_channels=96, attn_layers=2, attn_heads=4, attn_dim=128, trunk_w=384),
}
```
Wire a `--variant structured` + `--rung {small,medium,large}` in run_loop.train_round (run_loop.py:74-94)
that loads `vocab_counts_from_schema` + `build_field_specs` and calls the new constructor (data/loop
cluster owns the dispatch; C3 owns the module + RUNGS). Export the SMALL rung first.

## 6. Forward (full skeleton, seam-exact)
```python
def forward(self, inputs):
    g = inputs[self.INPUT_GLOBAL]; a = inputs[self.INPUT_ACTING]
    # spatial -> embed -> GNN
    sp = self._embed_tokens("spatial", inputs["spatial"])         # [B,N,proj]
    h = self.spatial_in(sp)                                        # Linear -> [B,N,C]
    nbr_idx = inputs["neighbor_index"]; nbr_mask = inputs["neighbor_mask"]
    for layer in self.gnn: h = layer(h, nbr_idx, nbr_mask)        # [B,N,C]
    gnn_kv = self.gnn_to_attn(h)                                  # [B,N,attn_dim]
    sp_mask = torch.ones(h.shape[0], h.shape[1], device=h.device, dtype=g.dtype)  # all tiles real
    # entity self-attention
    ent_tok=[]; ent_mask=[]
    for name in self.entity_names:
        t = self.entity_in[name](self._embed_tokens(name, inputs[name]))   # [B,M,attn_dim]
        m = inputs[name + "_mask"]
        for blk in self.self_attn[name]: t = t + blk(self.ln1[name](t), self.ln1[name](t), m)
        ent_tok.append(t); ent_mask.append(m)
    keys = torch.cat([gnn_kv] + ent_tok, dim=1)
    kmask = torch.cat([sp_mask] + ent_mask, dim=1)
    q = self.q_proj(torch.cat([g, a], dim=1)).unsqueeze(1)        # [B,1,attn_dim]
    for blk in self.cross: q = q + blk(self.lnq(q), self.lnk(keys), kmask)
    cross_ctx = q[:, 0, :]
    body = self.shared(torch.cat([cross_ctx, g, a], dim=1))
    ph = self.policy_late(body); vh = self.value_late(body)
    return self.tech_head(ph), self.policy_head(ph), torch.tanh(self.value_head(vh))
```

## 7. NaN-safety audit (must mirror masked_pool, model.py:50-63)
- GNN aggregate: `deg.clamp(min=1.0)` -> never /0 (==safe_count, model.py:58).
- Masked softmax: `masked_fill(min)`, then `w*mask`, then `denom.clamp(min=1e-9)` -> an all-padding
  key row yields all-zero weights (not NaN), context=0 (==max-over-empty->0 guard, model.py:62).
- Embedding index: `clamp(0, num_embeddings-1).long()` -> no OOB Gather (FND-0025) -> no NaN.
- Spatial N>=1 always (board non-empty); entity sets padded to max(1,count) by
  OnnxPolicy.tokenTensors (OnnxPolicy.kt:196-204) / features._pad_token_set (features.py:17-30) so
  q/kv axes are never length-0.

## Exact edits
- **python/unciv_train/model.py** [after RichPolicyValueNet (model.py:113)]: Add new module StructuredPolicyValueNet (+ helpers _Embeds, _GatherGNNLayer, _MHA, _masked_softmax, RUNGS) with forward signature identical to the seam: forward(inputs:dict)->(tech_logits, policy_logits, tanh(value)); attrs INPUT_GLOBAL/INPUT_ACTING. Reuse masked_pool guards' style and _small_init_value_head (model.py:20-26).
  _why:_ v4 swaps only the nn.Module; seam frozen, RichPolicyValueNet left intact for coexistence.
- **python/unciv_train/train.py** [train_actor_critic_rich, net construction (train.py:262)]: Branch on variant: when structured, net = StructuredPolicyValueNet(dims, token_specs, vocab_counts, field_specs, **RUNGS[rung]); build_rich_batch must also yield neighbor_index/neighbor_mask in `inputs` (train.py:266). _optimize_actor_critic/forward_fn untouched.
  _why:_ Construction site is the only train.py change permitted; core optimizer frozen (AC6).
- **core/src/com/unciv/logic/simulation/dataplane/DataPlaneHooks.kt** [buildHeaderJson JSON assembly (DataPlaneHooks.kt:171-172)]: Emit a "vocabCounts":{terrain,resource,improvement,religion,era,building,unit,nation,promotion} object from Vocab.size(...)/count getters.
  _why:_ Embedding num_embeddings must be read from schema/metadata, never hardcoded (D2 constraint); counts already drive the fingerprint so it is correctness-safe.
- **python/unciv_train/contract.py** [after token_specs_from_schema (contract.py:102)]: Add vocab_counts_from_schema(schema_path) reading sch['vocabCounts'] FAIL-LOUD (raise if missing), and build_field_specs(token_specs) returning the per-token-set categorical/numeric column plan keyed by the fixed channel/field order; assert each plan length == token_specs width (fail-loud drift).
  _why:_ Python mirror of the new schema field + the SSOT field plan; FND-0007/0011 fail-loud fallbacks.
- **python/unciv_train/contract.py** [CONTRACT_VERSION_RICH (contract.py:18)]: Bump 2 -> 3 (lockstep with SampleSchema.OnnxContract.CONTRACT_VERSION_RICH and SampleSchema.VERSION 2->3).
  _why:_ New layout (coords, map dims, tile indices, vocabCounts, neighbor tensors) = new contract; refuse old artifacts.
- **python/unciv_train/run_loop.py** [train_round dispatch (run_loop.py:74-94)]: Add variant 'structured' (alias rich-v2): load token_specs + vocab_counts + field_specs from schema, pick rung, call StructuredPolicyValueNet trainer path; return ('rich', token_specs) exporter tag with neighbor tensors added to export names.
  _why:_ Demand-driven ladder staging (D7); reuses the rich export path.
- **python/unciv_train/features.py** [build_rich_batch (features.py:33-48) and build_rich_single (features.py:51-69)]: After assembling spatial, derive neighbor_index [B,N,6] int64 + neighbor_mask [B,N,6] float32 from spatial coords (ch13/14) + global map-dims, add to the returned dict (handoff to data cluster; C3 fixes the tensor shape/dtype contract).
  _why:_ GNN inputs must be built on the Python side for training, matching JVM inference build.

## New inputs/tensors
- neighbor_index : int64 [B, N, 6] — per-tile hex-neighbor ROW indices (0..N-1) into the spatial node axis; cannot be folded into spatial because it indexes the node axis (different shape) and is derived from coords+map-dims not a per-channel value. Clamped 0..N-1 (self-filler for missing edges) so ORT Gather is OOB-safe (FND-0025). Needed because the GNN gathers neighbor features instead of scatter (opset-17 safe realization, research Q1).
- neighbor_mask : float32 [B, N, 6] — 1 if the corresponding neighbor edge exists (within map / valid after world-wrap), 0 otherwise. Drives the masked ReduceSum/Mean over the fixed degree-6 axis and the deg.clamp(min=1) NaN guard. Shares the 'n_spatial' dynamic axis with spatial + neighbor_index. Separate tensor (not folded) for the same shape/derivation reasons as neighbor_index.

## Lockstep sites
- SampleSchema.VERSION 2->3 (SampleSchema.kt:22)
- SampleSchema.OnnxContract.CONTRACT_VERSION_RICH 2->3 (SampleSchema.kt:42)
- contract.CONTRACT_VERSION_RICH 2->3 (contract.py:18)
- OnnxPolicy contract gate accepts 3: rich = mContract == CONTRACT_VERSION_RICH (OnnxPolicy.kt:59-62)
- SPATIAL_CHANNELS 13->15 (SampleSchema.kt:86-100) and Featurizer channels (Featurizer.kt:26)
- contract _TOKEN_WIDTH_FALLBACK spatial 13->15 — make fail-loud vs schema (contract.py:81-82)
- dataset.py reshape(-1,13)->read channel count from schema (dataset.py:113)
- OnnxPolicy FALLBACK_WIDTH own_units/own_cities widths 8/16 -> 9/17 (OnnxPolicy.kt:150-151)
- Featurizer unitTokenWidth 8->9, cityTokenWidth 16->17 (Featurizer.kt:29-30)
- NEW schema vocabCounts emission (DataPlaneHooks.kt:171) <-> vocab_counts_from_schema (contract.py)
- NEW neighbor_index/neighbor_mask in export names (export_onnx.py:109-115), features build (features.py:33-69), JVM build (OnnxPolicy.kt buildRichTensors 153-166), parity dims (test_parity.py:85-118)
- field_specs column orders mirror Featurizer writeUnitToken/writeCityToken/buildSpatial field order (Featurizer.kt:185-262)

## Export safety
All ops used are opset-17 core. D2 embeddings = Gather (opset 13). D3 GNN = torch.gather (GatherElements/Gather, opset 13) + Mul + ReduceSum/ReduceMean (opset 13/18-but-13 form used) + Div + Linear (MatMul+Add) + LayerNorm (opset 17 has native LayerNormalization) + Add; NO scatter_add/scatter_reduce/index_add anywhere (research-notes Q1: broken/silently-wrong on duplicate indices — avoided entirely via fixed degree-6 gather). D4/D5 attention is hand-rolled: Linear Q/K/V, MatMul, Div (scale), masked_fill+Softmax, Mul, Add, out Linear; NO nn.MultiheadAttention / F.scaled_dot_product_attention (research-notes Q2: no native ONNX attention before opset 23). Dynamic axes: batch {0} on all, ragged {1:'n_spatial'} on spatial + neighbor_index + neighbor_mask (must share the axis label so N is consistent), {1:'n_<name>'} on each entity set + its mask (same scheme the v2 export already uses, export_onnx.py:114-115; research Q3 confirms Gather+ReduceSum+masked-softmax all support dynamic batch and N at opset 17). neighbor_index is int64 (ORT Gather index dtype); values clamped 0..N-1 in the Python/JVM builder so Gather can never read OOB. NaN-impossible: every reduce/softmax clamps the denominator (deg.clamp(min=1), denom.clamp(min=1e-9)) and re-zeros padded weights, mirroring masked_pool (model.py:58,62); embedding indices clamped to table range. MANDATORY: export the SMALL rung first as a smoke test (onnx.checker + an ORT run on the parity fixture) before scaling (FND-0008/0023) — the new gather/attention graph is the pre-registered export risk.

## Determinism/provenance
Embedding counts are read from the new schema.json 'vocabCounts' (sourced from Vocab.size/count, which already feed canonicalSections and thus the ruleset fingerprint, Vocab.kt:80-100) — never hardcoded; a count change re-fingerprints and the OnnxPolicy provenance gate (OnnxPolicy.kt:51-63) refuses mismatched artifacts. token_specs and field_specs widths are asserted equal (fail-loud) so Python/Kotlin width drift is caught (not silently truncated as features._pad_token_set min(width) would do, features.py:27). neighbor_index/neighbor_mask are deterministic functions of emitted coords+map-dims (Python) and the live TileMap (JVM); zeroBasedIndex is assignment-order (TileMap.kt:610-617) so the (x,y)->row map is built from emitted coords, guaranteeing train/inference symmetry, guarded by the adjacency-parity test (FND-0036). torch.manual_seed(seed) is set before net construction (train.py:260) so embedding init is reproducible per round.

## Open questions
- Does buildHeaderJson have the live Vocab in scope, or must the counts map be passed in from the Featurizer/recorder? (affects exact edit at DataPlaneHooks.kt:171)
- unit_type table size: 5 (0=real 'none') vs 6 (0=sentinel)? Design picks 5 (no +1, matches current Featurizer channel 11 semantics) — confirm no Featurizer +1 is planned for it.
- Spatial cross-attention key mask: all-ones vs a real visibility/explored mask? Design picks all-ones for v1; a visibility mask would need a new spatial_mask emission.
- Should the GNN consume tile coords (ch13/14) as input node features too, or are they only used by the adjacency builder? Design includes them as numeric node features (harmless, gives positional signal) — confirm acceptable vs redundant with graph structure.
- Cross-attention keys union: include GNN nodes for ALL N tiles, or only a top-k/visible subset to bound N keys? Design uses all N (O(N) single query, fine for N≈1261); revisit only if memory-bound.
- tile_index in unit/city tokens (D1.3): design treats it as a NUMERIC pointer feature, not embedded, and does NOT yet wire a unit->tile cross-edge into the GNN. Is an explicit entity-to-node edge wanted in v4, or is cross-attention sufficient? Design defers entity->node edges to keep the GNN purely geometric.
- Rung channel/layer sizes (RUNGS) are first-cut; the acceptance harness (AC1 Medium z-test) may demand retuning — values are starting points, not load-bearing.

## Risks
- vocabCounts not in scope of buildHeaderJson (no Vocab param there today; Featurizer holds it). -> Thread the live Vocab (or a precomputed counts map) into buildHeaderJson at its call site; verify the recorder/featurizer call chain during build. Fail-loud in Python if the field is absent.
- unit_type table sizing (5 vs 6): channel 11 emits 0..4 where 0=none is a REAL category, but the +1 sentinel convention elsewhere makes 0=unknown. -> Decision: unit_type = nn.Embedding(5) (no +1), value 0 is a real 'none' row; do NOT +1 in the Featurizer for unit_type (it isn't today). Document and add an assert in build_field_specs. road similarly = nn.Embedding(3) raw ordinal.
- GNN gather uses torch.gather along the node axis with [B,N*6] indices; some ORT/exporter versions are picky about GatherElements vs Gather index rank. -> Use index_select-style gather via reshape to [B, N*6] then reshape back (shown in sketch); validate on the SMALL rung export smoke test FIRST; if GatherElements misbehaves, fall back to torch.index_select per-batch loop only if batch is static (it is not) — prefer the reshape+gather form which is well-supported at opset 13.
- Two new ONNX input names break the strict positional export order / JVM inventory check (OnnxPolicy.kt:66-74 only checks RICH_TOKEN_NAMES). -> Append neighbor tensors AFTER token sets in export names; extend the JVM inventory want-list to include neighbor_index/neighbor_mask (data/inference cluster). Their dynamic axis label must equal spatial's 'n_spatial'.
- spatial mask = all-ones may let the model attend to never-explored (all-zero) tiles, diluting cross-attention. -> Acceptable v1 (zeroed channels are learnable down-weight); if it hurts, emit a real spatial presence/visibility mask in a later iteration. Flagged as open question, not blocking.
- Embedding-then-concat changes per-token projected width vs the v3 raw-float width, so any width-derived assert tied to token_specs could fire. -> token_specs stays the RAW per-token width (what shards store); the projected width is computed internally from field_specs+embed_dim. Keep token_specs as the on-wire contract; do not conflate with proj width.

## VERDICT
```json
{
  "cluster": "C3-model-encoder",
  "export_safe": false,
  "lockstep_complete": false,
  "seam_preserved": true,
  "parity_feasible": true,
  "determinism_ok": false,
  "verdict": "REVISE",
  "issues": [
    {
      "severity": "critical",
      "issue": "INFERENCE PATH CANNOT BUILD neighbor tensors as designed. The exported ONNX will declare neighbor_index/neighbor_mask as REQUIRED inputs, but the JVM rich-inference path that actually runs the session (OnnxPolicy.forwardRich -> buildRichTensors -> richTensorsFromArrays, OnnxPolicy.kt:127-204) is a STATIC companion that takes only an `Observation` (a flat list of named float blocks) and has NO access to the live TileMap. The companion builders iterate exactly RICH_TOKEN_NAMES and emit only token+mask tensors; there is no seam to inject TileMap-derived int64 neighbor tensors. ORT session.run will then throw 'missing required input neighbor_index'. The design's claim 'OnnxPolicy builds neighbor-index from the live TileMap' is not realizable against the current code without a structural change the design does not specify.",
      "fix": "Make forwardRich the build site (it has `obs` produced from `civ.gameInfo` at OnnxPolicy.kt:106, so the live TileMap IS reachable there via the Featurizer's gameInfo). Add a NON-static neighbor-tensor builder that takes gameInfo.tileMap + vocab map-dims, produces [1,N,6] int64 idx + [1,N,6] f32 mask via real getIfTileExistsOrNull + zeroBasedIndex, and inject them into the LinkedHashMap returned by buildRichTensors BEFORE session.run. richTensorsFromArrays must gain an int64 tensor path (createTensor over a LongBuffer / int[][][]) since today it only builds float32. Specify this in C3's tensor contract or explicitly hand it to the inference cluster as a hard requirement, not a one-line 'OnnxPolicy builds it' aside."
    },
    {
      "severity": "critical",
      "issue": "JVM PROVENANCE INVENTORY GATE WILL REJECT THE v4 MODEL. OnnxPolicy.kt:66-74 builds the required-input want-list strictly from RICH_TOKEN_NAMES (+ global/acting + _mask each) and asserts `missing.isEmpty()` against session.inputNames. The v4 ONNX adds neighbor_index/neighbor_mask to its inputs; these are NOT in RICH_TOKEN_NAMES (design \u00a72.1 explicitly keeps RICH_TOKEN_NAMES unchanged), so `have` (session.inputNames) will contain two names not covered by `want`. Note the check is one-directional (want subset of have), so EXTRA inputs do not fail the gate by themselves \u2014 BUT the real failure is the inverse: the model REQUIRES neighbor_* yet nothing in want forces their presence, so a malformed export that drops them passes the gate, AND session.run then fails at runtime with no provenance diagnostic. The gate no longer guards the true input inventory.",
      "fix": "Extend the want-list at OnnxPolicy.kt:67-70 to include neighbor_index/neighbor_mask (add a NEIGHBOR_INPUT_NAMES constant in SampleSchema.OnnxContract alongside RICH_TOKEN_NAMES, mirrored in contract.py). Add these two names to the lockstep_sites list. This is a missed lockstep site the design did not enumerate."
    },
    {
      "severity": "critical",
      "issue": "EXPORT DTYPE CORRUPTION of neighbor_index. export_onnx.export_rich line 117 force-casts every sample_inputs entry to float32 (`torch.as_tensor(np.asarray(v, dtype=np.float32))`), and run_loop.py:219-220 additionally rebuilds `sample` from build_rich_batch (all float). If neighbor_index is threaded through sample_inputs (the natural place, since the design says features.build_rich_batch yields it), it is coerced to float32, so the traced graph's Gather receives a float index -> either export failure or a silently wrong int cast. The design asserts neighbor_index is int64 but never reconciles this with the float32-only sample/dummy plumbing.",
      "fix": "In export_rich, build the neighbor dummies/sample explicitly with dtype=torch.int64 for neighbor_index and torch.float32 for neighbor_mask, OUTSIDE the float32 .update() at line 117 (exclude them from that dict-comprehension, or special-case by name). The dummy must be torch.zeros(1, n0_spatial, 6, dtype=torch.int64). Add neighbor_index/neighbor_mask to the export `names`/`dyn` after token sets. State this dtype carve-out as a hard constraint on the export cluster."
    },
    {
      "severity": "major",
      "issue": "neighbor dynamic-axis N MUST equal spatial's N but the design's dummy generation breaks the binding. export_rich uses n0=2 for every token-set dummy row count (export_onnx.py:110). The design says give neighbor tensors `dyn {0:batch,1:n_spatial}` to share spatial's axis label, but ONNX dynamic_axes only ties shapes by SYMBOLIC LABEL, not value, during tracing; if the spatial dummy has 2 rows and the neighbor dummy is built with a different n0 (the design says torch.zeros(1, n0_spatial, 6) without defining n0_spatial vs the 2 used for spatial), the traced concrete shapes differ and torch.gather over [B,N,...] with idx into a length-2 node axis can trace constants that pin N. Must build neighbor dummies with EXACTLY the same row count as the spatial dummy (n0=2) AND the same dynamic label 'n_spatial' that spatial already gets from the token_specs loop ({1:'n_'+name} = 'n_spatial').",
      "fix": "Reuse the spatial dummy's row count: after the token_specs loop, set n_sp = dummy['spatial'].shape[1]; dummy['neighbor_index']=torch.zeros(1,n_sp,6,dtype=int64) (values in 0..n_sp-1, e.g. zeros, which is in-range), dummy['neighbor_mask']=torch.zeros(1,n_sp,6); dyn for both = {0:'batch',1:'n_spatial'} (label must be byte-identical to spatial's 'n_'+'spatial'). Verify on the small-rung export smoke test that ORT accepts a different N at run time."
    },
    {
      "severity": "major",
      "issue": "FAIL-LOUD FALLBACKS NOT ACTUALLY MADE FAIL-LOUD. The design lists in lockstep_sites that _TOKEN_WIDTH_FALLBACK (contract.py:80-82) and dataset.py reshape(-1,13) and OnnxPolicy FALLBACK_WIDTH (8/16) must become fail-loud / schema-driven, but the exact_edits array contains NO edit for contract.py:81-82, dataset.py:113, or OnnxPolicy.kt:150-151. They appear only as prose in lockstep_sites. As written, after the width bump (13->15, 8->9, 16->17) these stale fallbacks will SILENTLY truncate/mis-shape v3 data exactly where the council FND-0007/0011 said to fail loud (features._pad_token_set min(width) at features.py:27 silently truncates to the smaller width). The vocab_counts fail-loud is specified; the WIDTH fail-louds are dropped from exact_edits.",
      "fix": "Add concrete exact_edits for: (a) dataset.py:113 reshape(-1,13) -> reshape(-1, n_spatial_channels_from_schema) with a raise if unknown; (b) contract.py:81-82 fallback -> raise when schema omits the field (or delete the fallback and require schema); (c) OnnxPolicy.kt:150-151 FALLBACK_WIDTH 8/16 -> 9/17 AND assert against schema perItem. All belong to C3-adjacent lockstep; enumerate them as edits, not prose."
    },
    {
      "severity": "major",
      "issue": "VOCAB NOT IN SCOPE at the buildHeaderJson edit site, and the proposed edit is mechanically wrong. DataPlaneHooks.buildHeaderJson (DataPlaneHooks.kt:144-174) signature is (gameInfo, fingerprint, gameId, seed, caps, blocks) \u2014 there is NO `vocab` parameter and no Vocab is constructed inside it. The design's inserted Kotlin literally references `vocab.size(...)`/`vocab.buildingCount` which will not compile. The design flags this only as an open question/risk, not as a blocking edit defect, yet the proposed exact_edit at DataPlaneHooks.kt:171-172 is non-compiling as given.",
      "fix": "Thread the Vocab (or a precomputed VocabCounts data class) into buildHeaderJson from its call site (find the recorder/header-emit caller; Featurizer holds `vocab` at Featurizer.kt:20). Update the buildHeaderJson signature and ALL call sites. Treat as a required signature change, not an open question. Also: vocab exposes size(category) and the *Count getters but NOT every category as a count getter \u2014 terrain/resource/improvement/religion/era have no *Count getter, only size(Vocab.TERRAINS) etc.; the design correctly uses size(...) for those, but double-check building/unit/nation/promotion use the *Count getters (they exist) and terrain/resource/improvement/religion/era use size(Vocab.X) (they do) \u2014 the mix is correct, keep it consistent."
    },
    {
      "severity": "minor",
      "issue": "PARITY FIXTURE / D10 cannot exercise the GNN without neighbor tensors, and the parity harness build_rich_single (features.py:51-69) + JVM parityRunRich path build tensors ONLY from token sets. The design defers neighbor-tensor construction in the parity fixture to the data/test cluster but the existing test_parity uses synthetic dims (spatial:3, N=7) with NO map geometry, so an adjacency-parity test cannot reuse this fixture \u2014 it needs a real (x,y)+radius+worldWrap fixture. C3 sets the [B,N,6] contract but the parity feasibility depends on a new fixture that does not exist yet.",
      "fix": "Make C3's contract output explicit that the adjacency-parity test needs a NEW geometry fixture (coords + radius + worldWrap + expected neighbor rows), separate from the synthetic-dims rich-logits parity. Confirm the JVM parityRunRich entry is extended to build+emit neighbor tensors so the int64 path is parity-covered. Otherwise determinism of the train/inference symmetry is unguarded."
    },
    {
      "severity": "minor",
      "issue": "unit_type embedding sizing is internally inconsistent with the slot reindex discipline. The design decides unit_type table = 5 (0='none' is a real category, no +1) which is fine, but the spatial channel 11 (unit_type_cat) is only WRITTEN when a unit is present (Featurizer buildSpatial base+11 inside `if(unit!=null)`, codebase-scan.md:45); for tiles with no unit the channel stays 0 \u2014 which collides with the 'none' category 0. This is benign (no unit -> category 'none' is semantically correct) but the design's clamp(0,4) is right only because raw values are guaranteed 0..4; the same channel for an UNEXPLORED tile is also 0 (never written) and will be embedded as 'none' too \u2014 acceptable but the design should note unexplored vs no-unit both map to row 0 and rely on visibility channel 0 + unit_present channel 9 (kept numeric) to disambiguate. Confirm unit_present (col 9) is numeric in field_specs (it is) so the model can gate.",
      "fix": "Document that unit_type row 0 is overloaded (no-unit AND unexplored) and that disambiguation rides on the numeric visibility (col 0) + unit_present (col 9) channels; no code change needed, but add the assertion in build_field_specs that channels 0 and 9 are 'num'."
    }
  ],
  "corrected_notes": "Seam is genuinely preserved: forward(inputs:dict)->(tech_logits, policy_logits, torch.tanh(value)) with INPUT_GLOBAL/INPUT_ACTING matches model.py:88-89,108-113; train.py:261 is the sole construction site and _optimize_actor_critic/forward_fn (train.py:267-275) stay untouched; _RichPolicyOnly zip-by-names (export_onnx.py:41-43) tolerates appended inputs. Opset-17 op choices are sound: torch.gather (Gather/GatherElements), masked ReduceSum + deg.clamp(min=1), hand-rolled attention (Linear/MatMul/Div/Softmax/Mul/Add), LayerNormalization native at 17 \u2014 no scatter, no MHA/SDPA. NaN guards mirror masked_pool correctly.\n\nThe design FAILS on the inference/contract realization, not the math:\n1) The neighbor tensors are declared model inputs but there is NO build site on the JVM inference path that has the TileMap (the static companion builders take only Observation). forwardRich IS the correct injection point (it has civ.gameInfo) \u2014 the design must say so concretely and add an int64 tensor builder to richTensorsFromArrays.\n2) Two new ONNX inputs are NOT added to the provenance inventory want-list (OnnxPolicy.kt:67) nor to a contract constant \u2014 a missed lockstep site.\n3) The float32-only sample_inputs/dummy plumbing (export_onnx.py:117, run_loop.py:219-220) will corrupt neighbor_index to float \u2014 needs an explicit int64 carve-out.\n4) neighbor dummy row count/axis-label must be bound byte-identically to spatial's existing 'n_'+'spatial' label and same n0, else dynamic-N tracing pins N.\n5) The width fail-loud edits (dataset.py:113, contract.py:81-82, OnnxPolicy.kt:150-151) are named in lockstep_sites but absent from exact_edits \u2014 fold them into the edit list with concrete raises.\n6) buildHeaderJson has no vocab in scope; the inserted Kotlin won't compile \u2014 make threading Vocab a required signature change at all call sites, not an open question.\n\nColumn maps and HexMath clock offsets {(1,1),(0,1),(-1,0),(-1,-1),(0,-1),(1,0)} match codebase-scan.md:183. Slot table 42 (255->41 reindex) matches Featurizer ownerSlot (Featurizer.kt:43, 255=self). vocab.size(Vocab.X) vs *Count getter mix is correct. ATOL/parity feasible once a real-geometry fixture is added."
}
```
