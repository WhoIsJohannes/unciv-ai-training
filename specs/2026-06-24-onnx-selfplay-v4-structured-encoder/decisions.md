# Decisions — v4 Structured Encoder

## Pre-feature (Phase 1)
- **D-confound:** v4 adds encoder capacity but leaves from-scratch-per-round training (the recorded
  root cause of the Medium regression) untouched. User reviewed and chose **proceed with v4 as
  written**, accepting the confound. Plan must therefore (a) state the confound explicitly, (b)
  pre-register a falsifiable success metric, (c) report a null/negative result plainly (v2 ethos).
- **D-worktree:** spec's worktree+branch were externally deleted mid-session; recreated from master.
- **D-simbench-reuse:** `SimBenchmark` already exists → D8 EXTENDS it (add ONNX mode + 70% gate),
  not creates it. (Strict reuse-over-rebuild.)

## Research-driven (Step 4) — ONNX export realizations
- **D-gnn-export:** D3 GNN uses **neighbor-gather + masked-reduce over a fixed degree-6 axis**, NOT
  edge_index+scatter_add (scatter/index_add export is broken or silently wrong on duplicate target
  indices — pytorch#111159/#65138). Neighbor-index `[N,6]` + neighbor-mask `[N,6]` derived in Python.
- **D-attn-export:** D4/D5 use **hand-rolled scaled-dot-product attention** (Linear+matmul+masked
  softmax), NOT nn.MultiheadAttention/F.scaled_dot_product_attention (no native ONNX attention
  until opset 23).
- **D-no-pyg:** no PyTorch-Geometric dependency (its scatter path is the broken one); ~30 lines of
  stock torch.nn, matching model.py's hand-rolled style.

## Intake council triage (Step 5) — 37 findings; resolutions (Claude-resolvable, no user input needed)

### Accepted → fold into plan
- **FND-0016/0020 (critical) + 0006/0019/0033 — falsifiable success metric + confound guardrail:**
  pre-register AC1 exactly (structured BEATS v3 rich-pool on Medium, two-proportion z, p<0.05, final
  200-game eval; Tiny must not regress). Add a guardrail: report **train/eval gap per rung** (D7
  already) and the **round-over-round Medium curve** (the v3 decline signature). State the confound;
  report negative result plainly. → plan §Acceptance + §Confound.
- **FND-0008/0023/0025 (critical) — ONNX export validation + index safety:** export **smoke test on
  the SMALL rung FIRST** (gate before scaling). Validate neighbor-index bounds in Python
  (clamp/assert 0≤idx<N) so ORT Gather can never read OOB (FND-0025). → D3/D9 + test.
- **FND-0012/0026 (critical/major) — attention memory:** clarify in plan: spatial uses GNN
  (O(N·6), N≈1261), NOT self-attention over tiles; self-attention is only over **small** entity
  sets (O(M²), M≈dozens); cross-attention is a **single fixed query** over N keys (O(N), not O(N²)).
  No O(N²) over tiles. Bound dynamic axes by actual tile count. → plan §Complexity.
- **FND-0007 — spatial-channel god-constant (7 lockstep sites):** Kotlin `SampleSchema.SPATIAL_CHANNELS`
  stays the SSOT; Python reads channel count from schema.json. Make the Python/Kotlin **fallback
  widths fail-loud** (assert against schema, no silent revert — FND-0011/0015/0024). → D1.5/D9 + test.
- **FND-0022/0024 (critical) — migration/error-state test:** add a contract-version-mismatch refusal
  test (old shard / wrong fingerprint → fail-loud, never silently mis-decode). → D10.
- **FND-0036 — cross-language adjacency divergence:** add an **adjacency-parity check** — Python
  builder vs a small JVM-emitted/hand-computed reference incl. world-wrap + ragged-edge tiles. → D3/D10.
- **FND-0034 — construction-namespace collision is a live bug:** already D1.4 + AC7 unit test. Keep.
- **FND-0002/0035 — signed-coord encoding:** decide shard-format handling in design (depends on shard
  dtype from deep-scan). If shards store spatial as float/int32 → coords ride as 2 plain (signed)
  float channels inside `spatial` (no side-tensor, no bloat — answers FND-0002). If shards are u8 →
  documented bias offset. Document hex-geometry semantics (FND-0035). → plan §D1.
- **FND-0001/0003 (skeptic) — no encoder framework / stage behind gates:** implement v4 as ONE
  concrete module (small/medium/large via constructor args), NOT a plugin framework; the
  demand-driven ladder (D7) is the staging. Keep concrete. → plan §Encoder.

### Acknowledged → out of scope (offline RL, not a networked production service)
- **FND-0013/0014 (circuit-breaker/telemetry for inference failures):** the JVM already falls back to
  the heuristic FairOpponentModel if ONNX is unavailable; latency is captured by D8's Timers wrap.
  No production SLO surface here. Note in plan §Out-of-scope.
- **FND-0027 (provenance "cryptographic integrity"):** the ruleset fingerprint is a correctness gate
  for a single-user offline loop, not a security boundary (no adversary in the data path). Out of scope.
- **FND-0011 rollback:** "rollback" = contract version gate already refuses incompatible artifacts;
  datasets are perishable by design (regenerate). Documented, not a new mechanism.

### Deferred / minor
- 8 minor findings (naming, doc nits) folded opportunistically during build.

## Post-design-verification corrections (Step 8 adversarial workflow — all 4 clusters REVISE)
The design workflow's verify pass overturned/sharpened several decisions. Authoritative resolutions:
- **D-coords-storage (was wrong):** `spatial` block is **u8** (`fixedU8`, Observation.writeBlock clamps [0,255]).
  Coords CANNOT be spatial channels. → emit a **separate `spatial_coords` fixedF32 block [nTiles*2]**
  (x,y via `.toFloat()` — HexCoord is Int). SPATIAL_CHANNELS stays **13**. `spatial_coords` is **shard-only**
  (consumed by the Python adjacency builder); it is NOT an ONNX model input.
- **D-neighbor-tensors:** GNN inputs = `neighbor_index` [B,N,6] **int64** + `neighbor_mask` [B,N,6] f32 —
  NEW ONNX inputs (NOT via the token_specs `_mask` loop; NOT folded into spatial). Add `NEIGHBOR_INPUT_NAMES`
  to SampleSchema.OnnxContract + contract.py + the OnnxPolicy provenance want-list (missed lockstep site).
- **D-neighbor-build-sites:** Python train builds neighbor tensors from `spatial_coords` (pure function);
  JVM **inference builds from the live TileMap in `forwardRich`** (has `gameInfo`) via real
  `getIfTileExistsOrNull` over the 6 offsets in clock order (sentinel+mask0 for null) — **NOT `tile.neighbors`**
  (null-filtered, loses degree-6 slots). `richTensorsFromArrays` gains an int64 LongBuffer path.
- **D-pure-neighbor-fn (parity):** a PURE Kotlin `buildNeighborIndex(coords, effWrapRadius, worldWrap, shape)`
  used by `parityRunRich` (no TileMap there); tests chain Python-pure == Kotlin-pure (adjacency parity) and
  Kotlin-pure == live-engine (fidelity) so inference (live engine) is transitively covered.
- **D-wrap-radius:** buildGlobal emits the **pre-resolved effective wrap radius**
  (`shape==rectangular ? width/2 : mapSize.radius`), worldWrap bit, shape ordinal — read by NAMED schema field
  (not a fragile positional offset) + runtime assert.
- **D-export-int64:** export builds neighbor_index dummy `torch.zeros(1,n_sp,6,dtype=torch.int64)` OUTSIDE the
  float32 sample coercion; both adj tensors share spatial's `n_spatial` dynamic axis; degree axis static 6;
  small-rung export smoke test first.
- **D-fail-loud-edits:** dataset.py:113 `reshape(-1,13)`→schema channel count + raise; contract.py:81-82 fallback
  → raise on missing schema field; OnnxPolicy FALLBACK_WIDTH 8/16→9/17 + assert vs schema. These are EDITS.
- **D-header-vocab:** `DataPlaneHooks.buildHeaderJson` gets a Vocab/VocabCounts param (+ all call sites) to emit
  spatialChannels/perItem/vocab counts authoritatively.
- **D-simbench-ruleset:** the ONNX rung bench reuses the training ruleset/2-civ setup (SimulationCiv1/2), NOT
  SimBenchmark's 6-major BenchCiv1..6 (fingerprint mismatch would reject every model); 2-civ topology also fixes
  ms/decision attribution. Remove manual Timers around sim.start() (Simulation.start owns the timing window).
- **D-ladder-signals:** no held-out train win-rate exists; the train/eval-gap underfitting signal uses a
  calibrated value_loss-trend proxy (value head is small-init → tiny early value_loss) OR an added train-batch
  argmax pass; documented as the one soft signal. CPU OOM detected via train-subprocess SIGKILL (rc 137), not a
  Python exception. simBench gate = parse `BENCH| RUNG verdict=...` line as source of truth + rc for hard-crash.
- **Confirmed correct by verify:** opset-17 op choices (gather/masked-reduce/hand-rolled-attn/LayerNorm); the
  6-offset clock order + getIfTileExistsOrNull branch order A/B/C/D; row mapping; frozen seam; z-test reuse.

## Plan-council triage (Step 11 round 1 — 37 findings: 10 crit / 25 maj / 2 min) → plan revised
Correctness fixes folded into plan.md:
- **FND-0025** attention NaN guard: a fully-masked key set → output zeros (mirror masked_pool); per-entity
  self-attn skips empty sets; cross-attn K/V always has ≥1 GNN node so the union is never fully empty.
- **FND-0035** sentinel: pad node features with a dedicated zero row at index N; missing neighbor → index N
  (+mask 0), so even absent the mask the gather contributes zero. Bounds assert 0≤idx≤N.
- **FND-0036** use the entity tile-index: gather each entity's co-located GNN node embedding (by tile-index)
  and fuse into the entity token before self-attention — D1.3's field is now load-bearing.
- **FND-0012** fail-loud if a rung's data-gen/eval `onnx_decisions == 0` (silent FairOpponentModel fallback
  would invalidate the experiment by reporting heuristic results as model results).
- **FND-0010/0011** ladder train-rung runs under a memory cap + wall-clock timeout; abnormal exit → "rung
  rejected (train failed, likely OOM); micro-batching future work" — NOT SIGKILL-semantics-as-control-flow.
- **FND-0014** atomic ONNX export (.tmp → rename), matching ShardFormat's atomic finalize.
- **FND-0027** JVM neighbor_index bounds-check before tensor build (defensive, though zeroBasedIndex is valid).
Spec tightening folded in:
- **FND-0016** new §Non-negotiables. **FND-0030/0017** "budget constant" := identical (rounds=8, gen=16,
  eval=80, opponent=RandomPolicy, seeds, +200-game final) as the v3 rich-critic Medium run.
- **FND-0020/0018** Tiny non-regression := non-inferiority (one-sided two-proportion z, p>0.05 for
  structured<v3 over the final eval). **FND-0021/0006** ladder: K=3; "rising" := last eval > mean of prior 2;
  scale-up iff rising AND throughput≥70% AND train/eval-gap-small (calibrated value_loss-trend proxy, soft);
  STOP if a rung's final eval ≤ prev rung (within noise) OR throughput<70% OR train fails; max rung = large.
- **FND-0015/0001/0002/0003** build sequencing: Phase A (load-bearing, validate first) = D1 emit + D9 bridge +
  D2 embeddings + D3 GNN + D10 parity/export-smoke → GNN-only structured net, check Medium; Phase B = D4 self-
  attn + entity-node join + D5 cross-attn + D6 trunk split; Phase C = D7 ladder + D8 guard + full experiment.
  (Spec mandates the full stack; sequencing front-loads the diagnosed fix + export risk, doesn't drop scope.)
- **FND-0019/0007** exit/pivot: if the best rung doesn't beat v3 on Medium AND train/eval gap is large →
  conclude undertraining-confounded, report plainly, recommend weight-carryover, do NOT keep scaling. A
  negative result is an ACCEPTED reported outcome (user decision), not a ship-blocker.
- **FND-0037** owner_slot 255→41 reindex applied to ALL slot fields (spatial ch7/ch10, unit/city owner_slot).
Out of scope / confirmed:
- **FND-0028** untrusted-ONNX RCE: models are self-produced + fingerprint-gated; no third-party model loading
  in this flow. Out of scope (same misapplied-security lens as intake FND-0027).
- **FND-0033/0034** offsets engine-verified (C2 trace); clock order is a frozen cross-language convention
  guarded by the engine-fidelity test (Kotlin-pure == live getIfTileExistsOrNull); effWrapRadius pre-resolved.
- **FND-0026/0032** entity-set sizes are capped by the featurizer (maxCivTokens=40 + unit/city caps) → M bounded.
- **FND-0013** per-turn tensor alloc GC: measured by D8; acceptable; noted.

## Plan-council round 2 (CAP reached) — non-convergent; triage
Round 2 added 37 findings (74 total, 17 crit) but did NOT converge — the council re-articulated the same themes
(known regime; deterministic merge can't semantically dedup). CAP=2 reached → loop stops. Triage:
- Listed criticals FND-0005/0010/0015/0016/0020/0021/0025/0029/0030/0033 are **carried-forward round-1** findings
  already addressed in the plan revision (the consolidator carries them forward; they are not newly-unresolved).
- Re-statements of addressed concerns: FND-0042 (dual builders → 3-way parity), FND-0047 (fallback → now also
  fails loud on partial/per-decision ONNX errors), FND-0051 (MVP → build sequencing Phase A), FND-0056 (ladder
  thresholds → K=3 + hard gates; gap signal is an acknowledged soft heuristic), FND-0061 (entity caps exist),
  FND-0062 (bounds-check + sentinel=N).
- Genuinely NEW + folded in: **FND-0048** per-rung OrtSession closed in finally (no native leak);
  **FND-0047** partial per-decision ONNX fallback now fails loud.
- Out of scope / misapplied (offline single-user loop, not a networked service): DoS/tensor-bomb/RCE framings.
Decision: all actionable findings across both rounds are addressed; remaining "criticals" are
carried-forward-resolved, soft-signal-inherent, or out-of-scope. Proceed to the user approval gate (the user
adjudicates final sufficiency with the council verdict visible).

## Ship-council triage (Phase 4 Step 18 — 28 findings: 3 crit / 15 maj)
- **FND-0009/0021 (crit) NaN-grad in masked-softmax backward — FIXED**: replaced `masked_fill(-inf)`
  with a finite `-1e9` bias in `_masked_softmax_attend` (the -inf forward is zeroed by `where` but the
  softmax-jacobian backward NaNs even so). Added `test_attention_backward_finite_on_fully_masked_set`
  (backprops with every entity set masked + all nodes isolated; asserts finite grads). 12 attn/train tests green.
- **FND-0010 (crit) ORT threading**: v4 did NOT change the shared-session / intraOp=1 design (pre-existing
  v2/v3); a single session is shared across Simulation threads (ORT run is thread-safe). Out of scope.
- **FND-0001 bench base mutation**: bench-onnx runs two Simulations on the same `base` template — the same
  pattern eval uses; Simulation clones the template per game (else concurrent games would interfere), so
  baseline/onnx start from equivalent states. Low risk; the verdict ratio is a guard, not a precise metric.
- **FND-0004/0013 duplication** (HEX_OFFSETS Kotlin↔Python, _ALIASES seam): guarded by the adjacency-parity
  + contract fail-loud tests; logged to cleanup-opportunities.
- **FND-0007/0019 map-dims positional** (GLOBAL_MAPDIM_OFFSET=5): impl is positional + a worldWrap∈{0,1}
  runtime assert (not a named schema field as the plan aspired). The assert guards drift; acceptable.
- **FND-0017 coexistence**: model contract coexistence (gate accepts {1,2,3}) still holds; v1/v2 SHARDS are
  refused (perishable, SCHEMA_VERSION=3) by design.
- **FND-0023 per-decision neighbor rebuild**: correctness-safe; a per-game cache is a future optimization
  (the graph is static within a game). Logged as future work.
- **FND-0024 rectangular/flatEarth**: Unciv is hex-based for ALL shapes; getIfTileExistsOrNull + the hex
  offsets apply to every shape (only the wrap radius differs, pre-resolved). Adjacency-fidelity tested hex;
  a rectangular case would be more thorough (noted).
- Remaining majors/minors (OOM bound, opaque exceptions, clamp sentinel): low-risk / documented / cleanup.
