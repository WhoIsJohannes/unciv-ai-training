# Build progress — v4 Structured Encoder

Plan items as checkboxes. `[x] verified in <file>:<line>` when done; `[ ] MISSING` if not.
Full design detail: `design/design-C1..C4.md`. Decisions: `decisions.md`. Build order: Phase A → B → C.

## Phase A — load-bearing core (contract + GNN + parity; validate Medium before layering)
### D1 — Featurizer/contract emit (Kotlin)
- [x] D1.5 SampleSchema: VERSION 2→3, CONTRACT_VERSION_RICH 2→3, NEIGHBOR_INPUT_NAMES, SPATIAL_CHANNELS (13, unchanged)
- [x] D1.1 Featurizer.buildSpatial coords → separate fixedF32 `spatial_coords` block [nTiles*2] (`.toFloat()`)
- [x] D1.2 buildGlobal: +effWrapRadius (pre-resolved) +worldWrap +shape ordinal; named schema fields
- [x] D1.3 unit token 8→9 (currentTile.zeroBasedIndex); city token 16→17 (centerTile.zeroBasedIndex)
- [x] D1.4 construction-collision fix (Featurizer:205): unit branch += buildingCount offset
- [x] D-header-vocab: DataPlaneHooks.buildHeaderJson threads Vocab → emit spatialChannels/perItem/counts
### D9 — contract/bridge (Python + JVM)
- [x] contract.py: CONTRACT_VERSION_RICH 3, NEIGHBOR_INPUT_NAMES, token_specs reads channel count, fallback→raise
- [x] dataset.py: reshape(-1,13)→schema channel count; read spatial_coords block
- [x] export_onnx.export_rich: neighbor_index (int64, OUTSIDE float coercion) + neighbor_mask; shared n_spatial axis; atomic export; small-rung smoke
- [x] OnnxPolicy.kt: gate accepts {1,2,3}; want-list += NEIGHBOR_INPUT_NAMES; int64 LongBuffer path; forwardRich builds neighbor tensors from live TileMap (6 offsets, sentinel=N, bounds-check); FALLBACK_WIDTH 8/16→9/17 + schema assert
### D2 — embeddings (model)
- [x] nn.Embedding tables (counts from schema, +1 sentinel); shared TERRAINS; slot table 42 (255→41); construction table buildingCount+unitCount+1; numeric scalars concatenated
### D3 — hex GNN
- [x] hexgraph.py build_neighbor_graph (pure; mirrors getIfTileExistsOrNull; sentinel=N; bounds 0≤idx≤N)
- [x] GNN: gather neighbors [N,6,C] + masked reduce over degree-6 (Gather/Mul/ReduceSum only)
- [x] StructuredPolicyValueNet skeleton preserving the frozen seam (GNN-only first)
### D10 — parity (Phase A subset)
- [x] test_hexgraph.py GREEN (adjacency builder)
- [x] adjacency parity (Python-pure == Kotlin-pure) + engine fidelity (Kotlin-pure == live)
- [x] shard-roundtrip: COVERED BY DESIGN — coords in a separate f32 block (lossless); parity uses f32 coords so it IS representative
- [x] construction-collision unit test (AC7)
- [x] small-rung ONNX export smoke (opset 17)
- [x] rich-logits parity extended to neighbor tensors (atol 1e-4)
- [x] contract-version-mismatch refusal test
- **CHECKPOINT A:** train GNN-only structured net, check Medium vs v3

## Phase B — attention layers
- [ ] D4 self-attention over entity sets (hand-rolled, NaN-safe; fully-masked set → zeros)
- [ ] D4 entity↔node join (gather co-located GNN node by tile-index, fuse into entity token)
- [ ] D5 cross-attention (single query; K/V = nodes⊕entities; union mask; NaN-safe)
- [ ] D6 trunk split (shared body → policy-late {tech,policy} + value-late; small-init value head)

## Phase C — ladder + throughput + experiment
- [ ] D7 ladder (small/medium/large; K=3; scale/stop rules; OOM via mem-cap+timeout; train/eval-gap proxy)
- [ ] D7 run_loop --variant structured (alias rich-v2); run_one_round returns model_path+rung
- [ ] D8 Timers.timeThis("onnxForward") wrap; SimBenchmark ONNX mode (training ruleset/2-civ); 70% gate; BENCH| RUNG line; OrtSession closed in finally
- [ ] D10 experiment: Tiny+Medium curves, budget constant vs v3; 200-game Medium z-test (AC1); Tiny non-inferiority

## Non-negotiables (assert throughout)
- [ ] frozen seam; _optimize_actor_critic/compute_gae/train core untouched
- [ ] opset 17 (no scatter/SDPA); heads {tech,policy}+value-train-only; terminal-reward+critic only
- [ ] fail-loud lockstep; legality; determinism+provenance; onnx_decisions>0 + ONNX-error fail-loud

## Codebase patterns (fill as discovered)
- (Kotlin) blocks written via Observation.writeBlock; u8 clamps [0,255], f32 lossless (ShardFormat.f32s)
- (Python) reader returns blocks; dataset casts float32; build_rich_single is the parity reference
