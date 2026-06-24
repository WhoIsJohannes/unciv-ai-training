# Plan — v4 Structured Encoder (hex-GNN + embeddings + attention)

**Mode** BUILD · **Size** L · **Base** master `5ac0a3cf6` · **Branch** `onnx-selfplay-loop` · **Contract** v2→**v3**

> Detailed per-cluster designs (with code) live in `design/design-C1..C4.md`; locked decisions in
> `decisions.md`; exact source in `codebase-scan.md`; export research in `research-notes.md`.

## TL;DR — how this solves the ask
The v3 rich-critic loses on Medium (14.7% vs blind 28.9%) because its spatial encoder is a masked
mean+max **pool** — permutation-invariant, zero positional signal — so on 1261 Medium tiles it discards
2D locality. v4 replaces the pool with a **structured encoder**: categorical **embeddings** + a hex-aware
**GNN** (gather neighbors over true hex adjacency, masked-reduce over the fixed degree-6 axis) + hand-rolled
**self-attention** over entity sets + **cross-attention** (a `[global⊕acting_civ]` query reads the board),
governed by a demand-driven **capacity ladder** under a hard **throughput guard**. The frozen training core
(`_optimize_actor_critic`/`compute_gae`/terminal-reward+critic) is untouched — only the `nn.Module` is swapped.

**Pre-registered, falsifiable success metric (AC1):** the structured encoder BEATS the v3 rich-pool on
**Medium** at **p<0.05** (two-proportion z, structured as p1, over the final 200-game eval), and **Tiny does
not regress** — defined as **non-inferiority**: one-sided two-proportion z gives p>0.05 for the hypothesis
"structured < v3" over Tiny's final eval (FND-0018/0020). **"Budget held constant"** := identical
(rounds=8, gen-games/round=16, eval-games/round=80, opponent=RandomPolicy, fixed seeds, + a 200-game final
eval) as the v3 rich-critic Medium run (FND-0030/0017). **A null/negative result is an accepted, reported
outcome** (the user accepted the confound), not a ship-blocker (FND-0007/0019).

## Non-negotiable constraints (FND-0016)
1. Frozen seam `forward(inputs:dict)→(tech_logits,policy_logits,tanh(value))`; `_optimize_actor_critic`,
   `compute_gae`, `train_actor_critic_rich` core UNTOUCHED — only the `nn.Module` is swapped.
2. ONNX opset 17: NO `scatter_add`/`scatter_reduce`/`index_add`; NO `nn.MultiheadAttention`/`F.SDPA`.
3. Heads stay {tech, policy} (+value training-only). No new action heads. Terminal reward ±1 + critic only — no shaping.
4. Lockstep widths/versions move together; all fallbacks **fail loud** (raise) vs schema — never silently truncate.
5. Legality preserved (masked heads never pick illegal); determinism + provenance (schema_version + ruleset
   fingerprint) stamped & gated. A rung whose data-gen/eval `onnx_decisions == 0` **aborts fail-loud** (the
   silent FairOpponentModel fallback must never be reported as model results — FND-0012). During a gen/eval
   rung an ONNX **forward error fails loud** (no silent per-decision heuristic fallback that would partially
   corrupt the rung — FND-0047).
6. NaN-safe everywhere masks are applied (mirror `masked_pool` guards) — incl. hand-rolled attention (FND-0025).

## Build sequencing (validate the load-bearing fix first — FND-0015/0001-0003)
- **Phase A (the diagnosed fix + export risk — validate before layering):** D1 emit + D9 contract/bridge +
  D2 embeddings + D3 GNN + D10 parity & small-rung export smoke → train a **GNN-only** structured net and
  check Medium. This is the minimum to test "does restoring 2D locality beat the pool?"
- **Phase B (layer on, spec-mandated):** D4 self-attention + entity↔node join (FND-0036) + D5 cross-attention + D6 trunk split.
- **Phase C:** D7 ladder + D8 throughput guard + the full Tiny/Medium experiment.
(The spec mandates the full stack; sequencing front-loads the diagnosed fix and the export risk — it does not drop scope.)

## ⚠️ Confound (stated plainly, per the user's accepted decision)
v4 adds encoder capacity but **leaves from-scratch-per-round training untouched** — and the recorded v2
diagnosis (memory + RESULTS.md) attributes the Medium regression to *undertraining* under that regime, warning
that *adding capacity could make Medium worse*. The user reviewed this and chose to proceed with v4 as written.
Therefore the plan: (a) holds budget **constant vs v3 rich-critic** for the headline comparison; (b) reports
the **train/eval gap** and the **round-over-round Medium curve** per rung (the v3 decline signature is the
undertraining tell); (c) reports a **null/negative result plainly** (v2 ethos — no Goodhart-tuning toward a win).
Weight-carryover remains the recommended follow-up if the result is ambiguous.

---

## The v3 contract (corrected after adversarial verification)

**Shard blocks** (emitted by Kotlin Featurizer; `VERSION` 2→3):
| block | dtype | width | change |
|---|---|---|---|
| `global` | f32 | 26→**29** | +effective_wrap_radius, +worldWrap bit, +shape ordinal (D1.2) |
| `acting_civ` | f32 | unchanged | — |
| `spatial` | **u8** | **13** (unchanged) | categorical channels only; **NOT** widened (u8 clamps [0,255]) |
| `spatial_coords` | **f32** | **2/tile** (NEW) | tile (x,y) as signed float; shard-only (adjacency source) (D1.1) |
| `own_units`/`opp_units` | f32 | 8→**9** | +`currentTile.zeroBasedIndex` (D1.3) |
| `own_cities`/`opp_cities` | f32 | 16→**17** | +`centerTile.zeroBasedIndex` (D1.3); construction-collision fix (D1.4) |
| `civ_tokens` | f32 | 84 | unchanged |

**ONNX model inputs** (`CONTRACT_VERSION_RICH` 2→3): `global`, `acting_civ`, the 6 token sets + `_mask`
each (unchanged names), **plus NEW** `neighbor_index` [B,N,6] **int64** + `neighbor_mask` [B,N,6] f32.
`spatial_coords` is **shard-only** (the Python adjacency builder reads it; the model never sees raw coords —
locality comes from the GNN over `neighbor_index`). The JVM builds `neighbor_index` at inference from the live
`TileMap`. New constant `NEIGHBOR_INPUT_NAMES` mirrored Kotlin↔Python; added to the OnnxPolicy provenance gate.

**Lockstep sites** (all move together; fallbacks become **fail-loud**, not silent): Featurizer (buildSpatial
channels / new spatial_coords block / buildGlobal head / writeUnit/CityToken widths + construction fix) ·
`SampleSchema` (VERSION, SPATIAL_CHANNELS, OnnxContract.CONTRACT_VERSION_RICH, NEIGHBOR_INPUT_NAMES) ·
`DataPlaneHooks.buildHeaderJson` (gains a Vocab/VocabCounts param to emit spatialChannels/perItem/counts) ·
`contract.py` (versions, fallback→raise, token_specs, neighbor names) · `dataset.py:113` reshape(-1,13)→schema
channel count · `features.py`/`hexgraph.py` · `export_onnx.py` (int64 adj inputs) · `OnnxPolicy.kt`
(FALLBACK_WIDTH 8/16→9/17 + schema assert; int64 path; forwardRich neighbor build; want-list) · parity fixtures.

---

## Deliverables (corrected)

### D1 — Featurizer/contract emit  `[AI_CODE]`
- **D1.1** new `spatial_coords` fixedF32 block `[nTiles*2]`, `tile.position.x.toFloat()/y.toFloat()` per tile
  (HexCoord is **Int** → `.toFloat()` required). `spatial` stays 13 u8 channels. (Resolves C1/C2 critical.)
- **D1.2** `buildGlobal` head 5→8: **pre-resolved** `effWrapRadius = if(shape==rectangular) width/2 else radius`,
  `worldWrap` bit, `shape` ordinal. Surface these as **named** schema fields (read by name, +runtime assert).
- **D1.3** unit token 8→9 (`currentTile.zeroBasedIndex`), city token 16→17 (`centerTile.zeroBasedIndex`,
  inserted unconditionally before the `if(isOwn||hasSpy)` block).
- **D1.4** fix construction collision (Featurizer.kt:205): unit branch gets `+vocab.buildingCount` offset so
  building#k and unit#k never collide. **Unit-tested** (AC7).
- **D1.5** `SampleSchema.VERSION` 2→3; SPATIAL_CHANNELS unchanged; fingerprint changes via new block/header (old
  shards correctly refused — regenerate).

### D2 — Embeddings (model)  `[AI_CODE]`
`nn.Embedding` tables, `num_embeddings = count+1` (row 0 sentinel), **counts read from schema/ONNX metadata
(never hardcoded)**. Shared TERRAINS table (base+feature); resource; improvement; owner_slot & unit_owner_slot
share a **slot table size 42** (`maxCivTokens 40 +2`; reindex 255→41); unit_type_cat (5); road (3/scalar);
city `majority_religion` (religion+1), `current_construction` single table `buildingCount+unitCount+1`
(consistent with D1.4); civ `era`. Numeric fields stay scalars, concatenated pre-projection.

### D3 — Hex GNN (gather, not scatter)  `[AI_CODE]`
New `python/unciv_train/hexgraph.py`: **pure** `build_neighbor_graph(coords, eff_wrap_radius, world_wrap,
shape) → (neighbor_index[N,6] int64, neighbor_mask[N,6])`. Builds `{(x,y)→row}` from `spatial_coords`; for each
tile×direction (clock order `[(+1,+1),(0,+1),(-1,0),(-1,-1),(0,-1),(+1,0)]`) computes (nx,ny), direct lookup, then
world-wrap retry `(nx+R,ny-R)` then `(nx-R,ny+R)` — **exact** mirror of `TileMap.getIfTileExistsOrNull` (branch
A/B/C/D). Node features are padded with a dedicated **zero row at index N**; a missing neighbor → **sentinel
index N** (+mask 0), so the gather contributes zero even if a downstream path forgets the mask (FND-0035).
**Bounds-assert** `0≤idx≤N` (no OOB Gather — FND-0025). GNN: `Gather` neighbors→[N,6,C], message MLP, ×mask,
`ReduceSum`/clamped-mean over degree axis, L×C per rung. Per-node output (not pooled) → cross-attention K/V.
Ops: Gather/Mul/ReduceSum/MatMul only (opset-17 safe). Offsets are engine-verified (deep-scan trace); the clock
order is a **frozen cross-language convention** guarded by the engine-fidelity test (FND-0033).

### D4 — Self-attention over entity sets + entity↔node join  `[AI_CODE]`
First **fuse each entity with its board location** (FND-0036): gather the entity's co-located GNN node embedding
via its tile-index (D1.3) and concat/add into the entity token — grounding the entity on the board (this makes
D1.3's field load-bearing). Then hand-rolled scaled-dot-product (Linear Q/K/V, matmul, /√d, masked softmax,
matmul, out-proj), pre-LN, A_e layers × H heads, presence-masked. **NaN-safe (FND-0025):** a fully-masked entity
set yields **zeros** (mirror `masked_pool`: a row with no real keys → softmax not taken / output zeroed), never
NaN. Entity-set sizes are **capped** by the featurizer (maxCivTokens=40 + unit/city caps) → M bounded, no
unbounded quadratic (FND-0026/0032). owner_slot 255→41 reindex applied to ALL slot fields (FND-0037).

### D5 — Cross-attention  `[AI_CODE]`
Query `[global⊕acting_civ]→model dim` (single query → **O(N)**, not O(N²)); K/V `[GNN nodes ⊕ entity tokens]`
with the union mask; A_x layers × H heads → fixed context vector. The K/V set always contains ≥1 GNN node
(every map has tiles), so the union mask is never fully empty → cross-attention softmax is always well-defined
(no NaN path; FND-0025).

### D6 — Trunk split  `[AI_CODE]`
Shared body `[cross_ctx ⊕ global ⊕ acting_civ] → MLP(trunk_w)`; SEPARATE `policy_body→{tech_head,policy_head}`
and `value_body→value_head` (keep `_small_init_value_head`, tanh). Heads stay {tech,policy}(+value training-only).
New module `StructuredPolicyValueNet` preserves the FROZEN seam `forward(inputs:dict)→(tech,policy,tanh(value))`.

### D7 — Capacity ladder + demand-driven scaling  `[AI_CODE]`
Constructor args `(embed_dim, gnn_layers, gnn_channels, attn_layers, attn_heads, attn_dim, trunk_w)`; rungs
small/medium/large per spec (large = the cost ceiling). Orchestrated in `run_loop.py` (`run_one_round` refactor
returns `model_path`+`rung`). **Concrete thresholds (FND-0021/0006):** **K=3**; "eval still rising" := last eval
win-rate > mean of the prior 2; START small; **SCALE UP** one rung iff (rising **AND** throughput ≥70% baseline
**AND** train/eval-gap-small); **STOP** when a rung's final eval ≤ the previous rung's (within noise band) OR
throughput <70% OR the train step fails. Train/eval-gap is a **calibrated value_loss-trend proxy** (documented
soft signal — the value head is small-init so early value_loss is tiny; optionally an added train-batch argmax
pass). **OOM/failure handling (FND-0010/0011):** the train rung runs under a **memory cap + wall-clock timeout**;
an abnormal exit → "rung rejected (train failed, likely OOM); micro-batching = future work" — reported honestly,
**not** SIGKILL-semantics-as-control-flow. Record signals + decision per rung.

### D8 — Throughput guard (extends existing SimBenchmark)  `[AI_CODE]`
Wrap `OnnxPolicy.forwardRich` in `Timers.timeThis("onnxForward"){…}` (do **not** manually wrap `sim.start()` —
`Simulation.start` owns the timing window). EXTEND `SimBenchmark` with an ONNX-policy rung mode built from the
**training ruleset/2-civ setup (SimulationCiv1/2)** — NOT the 6-major BenchCiv1..6 (fingerprint mismatch would
reject every model; 2-civ also fixes ms/decision attribution). Measure data-gen turns/s + ms/decision; emit a
`BENCH| RUNG verdict=PASS|REJECT` line (the ladder parses that line as source of truth, rc only for hard-crash);
**REJECT a rung < 70% of the heuristic baseline** turns/s. Report turns/s + ms/decision for EVERY rung tried.
Close the per-rung `OrtSession`/`OnnxPolicy` in a `finally` (no native session leak across rungs — FND-0048).

### D9 — Contract growth + multi-tensor bridge  `[AI_CODE]`
`contract.py` CONTRACT_VERSION_RICH 2→3, `NEIGHBOR_INPUT_NAMES`, token_specs reads new channel count, fallback→
**raise**. `export_onnx.export_rich`: append `neighbor_index` (dummy `torch.zeros(1,n_sp,6,dtype=torch.int64)` built
OUTSIDE the float32 sample coercion) + `neighbor_mask`, sharing spatial's `n_spatial` dynamic axis, degree axis
static 6; policy-only; **small-rung export smoke test first**; **atomic export** (.tmp → rename, like ShardFormat
— FND-0014). `OnnxPolicy.kt`: contract gate accepts {1,2,3}; want-list += NEIGHBOR_INPUT_NAMES;
`richTensorsFromArrays` gains an int64 LongBuffer path; `forwardRich` builds neighbor tensors from
`gameInfo.tileMap` via real `getIfTileExistsOrNull` over the 6 offsets (sentinel index N + mask0 for null —
**not `tile.neighbors`**), with a **bounds-check** `0≤idx≤N` before tensor build (FND-0027); all tensors closed in
finally. `SampleSchema.OnnxContract` mirrors. `dataset.py` reshape→schema channel count. `DataPlaneHooks.buildHeaderJson`
threads Vocab to emit spatialChannels/perItem/counts. (Untrusted third-party ONNX loading is out of scope — models
are self-produced + fingerprint-gated; FND-0028.)

### D10 — Parity + experiment  `[AI_CODE]` `[AI_RESEARCH]`
Tests: (1) **shard-roundtrip** — write a u8 spatial + f32 coords block with negative & >255 coords, reload via
`unciv_dataplane.reader`, build adjacency, assert == live-TileMap reference (catches the u8-storage contract the
idealized-float parity misses); (2) **adjacency parity** Python-pure == Kotlin-pure over a synthetic world-wrap map
with ragged edges (both hex + rectangular width/2≠radius); (3) **engine fidelity** Kotlin-pure == live
`getIfTileExistsOrNull`; (4) **rich-logits parity** extended to the wider multi-tensor input incl. neighbor_index/
mask (atol 1e-4, logits incl.) — a shared **pure neighbor-builder** lets `parityRunRich` inject the adj tensors
(no TileMap there); (5) **contract-version-mismatch refusal** test; (6) **construction-collision** unit test (AC7).
Experiment: `run_loop --variant structured` (alias rich-v2); eval curves on Tiny AND Medium at budget held
constant vs v3 rich-critic; final 200-game Medium eval + two-proportion z (`analyze._two_proportion_z`,
structured as p1); report train/eval gap + round-over-round Medium curve per rung.

---

## Worked walkthrough — one Medium decision, end to end
1. **Emit** (Kotlin, gen): Featurizer writes `spatial` (1261×13 u8), `spatial_coords` (1261×2 f32, e.g. tile 530 →
   (x=-3, y=7)), `global` (…, effWrapRadius=21, worldWrap=1, shape=1), unit tokens (…, tileIdx=530), city tokens.
2. **Shard→Python** (train): reader loads blocks (spatial as u8→float, coords as f32). `hexgraph.build_neighbor_graph`
   reads coords → for tile 530, neighbors = rows of (−2,8),(−3,8),(−4,7),(−4,6),(−3,6),(−2,7); off-map dirs → idx 0
   mask 0; world-wrap dirs resolved via ±R. → `neighbor_index[1261,6]`, `neighbor_mask[1261,6]`.
3. **Model.forward(inputs)**: embeddings look up terrain/resource/…; GNN gathers each node's 6 neighbors
   ([1261,6,C]), message-MLP, mask, reduce → per-node embeddings; entity self-attention; cross-attention query
   `[global⊕acting_civ]` reads `[nodes⊕entities]` → context; trunk → `tech_logits`, `policy_logits`, `tanh(value)`.
4. **Export**: `_RichPolicyOnly` drops value; ONNX has inputs `{global, acting_civ, 6×(set+mask), neighbor_index
   int64, neighbor_mask}`; metadata stamps schema_version/fingerprint/contract_version=3/input_names.
5. **Inference** (Kotlin, JVM): `forwardRich` builds the same `neighbor_index` from the live `TileMap`
   (`getIfTileExistsOrNull`), feeds all tensors, runs ORT, masks illegal actions, picks a legal tech/policy.
   *(Illustrative mock values — not measured; sanity-check against the design.)*

## Risks
| # | Risk | L×S | Mitigation |
|---|---|---|---|
| 1 | ONNX export of GNN gather / hand-rolled attention fails at opset 17 | M×H | gather+masked-reduce + hand-rolled attn (no scatter/SDPA) — research-confirmed; **small-rung export smoke test first** |
| 2 | Python adjacency diverges from Kotlin engine → wrong graph, silent | M×H | shared pure fn; 3-way parity (Python-pure==Kotlin-pure==live-engine); world-wrap+ragged+rectangular cases |
| 3 | Coord storage corruption (u8) | (resolved) | separate f32 `spatial_coords` block + shard-roundtrip test |
| 4 | **Confound**: v4 worse on Medium under from-scratch training (diagnosis warned) | M×H | budget held constant; report train/eval gap + Medium curve; report negative plainly; ladder starts small + stops on no-improve |
| 5 | Lockstep miss → silent mis-decode | M×H | fail-loud fallbacks (edits), contract-version refusal test, fingerprint gate |
| 6 | Throughput rung < 70% baseline | M×M | D8 hard gate, ladder rejects; report every rung |
| 7 | Whole-round dense batch OOM on Medium (1261 tiles) | M×M | ladder rejects OOM rung (SIGKILL rc 137); micro-batching noted future work |

## Out of scope
New action heads (v5); self-play (v6); reward shaping; recurrence/world-model; fine unit-type/nation id;
micro-batching the dense batch (future work); edge-list emission; inference circuit-breaker/telemetry (offline
loop; JVM already falls back to FairOpponentModel); fingerprint "cryptographic" hardening (correctness gate, not a
security boundary).

## Verification (what proves it)
- `pytest python/tests/` — parity (rich-logits + adjacency + shard-roundtrip + engine-fidelity + contract-refusal) green
- `./gradlew test` — dataplane unit tests incl. construction-collision (AC7) + legality
- small-rung ONNX export smoke (gather/attn export under opset 17)
- `./gradlew :desktop:simBench --args='onnx <model>'` — per-rung turns/s + ms/decision; ≥70% baseline for the shipped rung
- `run_loop --variant structured` Tiny+Medium curves; final 200-game Medium z-test (AC1); Tiny non-regression
