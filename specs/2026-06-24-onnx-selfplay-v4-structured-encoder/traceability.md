# Traceability — v4 Structured Encoder (plan item → impl → test)

| Deliverable | Implemented in | Covering test / verification |
|---|---|---|
| **D1.1** tile coords (f32 `spatial_coords` block) | `Featurizer.buildSpatialCoords`, `SampleSchema.BLOCK_SPATIAL_COORDS` | `:core:compileKotlin`; end-to-end smoke (gen emits it); adjacency-fidelity test reads it |
| **D1.2** global map dims (effWrapRadius/worldWrap/shape) | `Featurizer.buildGlobal` | adjacency-fidelity test (uses emitted dims); `features._read_mapdims` asserts worldWrap∈{0,1} |
| **D1.3** entity tile-index (unit f8, city f12) | `Featurizer.writeUnitToken`/`writeCityToken` | entity↔node join (Phase B) + structured-train test |
| **D1.4** construction-collision fix | `Vocab.constructionCode`, `Featurizer.writeCityToken` | **AC7** `ConstructionCodeTest` (injective over buildings∪units) ✓ |
| **D1.5** schema VERSION 2→3 + lockstep | `SampleSchema.VERSION`, `schema.py SCHEMA_VERSION`, reader | fail-loud version gate (caught in smoke → reader bumped); `test_contract_failloud` |
| **D2** embeddings (counts from schema) | `model._SpatialEmbed`; `contract.vocab_counts_from_schema`; `DataPlaneHooks` vocabCounts | `test_contract_failloud` (plural→singular + fail-loud); structured smoke/parity |
| **D3** hex GNN (gather, no scatter) | `hexgraph.build_neighbor_graph`; `model._GatherGNNLayer` | `test_hexgraph` (4) ✓; adjacency-fidelity (hexgraph==engine) ✓; export op-set has no scatter |
| **D4** self-attn + entity↔node join | `model._MHA/_AttnBlock` + join | `test_structured_attn` (NaN-safe, ORT≈torch); structured-train (medium) |
| **D5** cross-attention (single query) | `model` cross-attn | `test_structured_attn`; export op-set (no Attention/MHA op) |
| **D6** trunk split | `model` policy_body/value_body (small-init value) | structured smoke (value∈[-1,1]); `test_export_rich_drops_value` (value head dropped) |
| **D7** ladder + `--variant structured` | `train.train_actor_critic_structured`; `run_loop` dispatch + `--rung` | `test_structured_train` (both rungs); **end-to-end smoke (onnx_dec=150)** ✓ |
| **D8** throughput guard | `SelfPlayRunner.benchOnnx`; `OnnxPolicy` Timers wrap; eval JSON metrics | **bench-onnx smoke (verdict=PASS, ms/decision)** ✓ |
| **D9** contract v3 + bridge | `contract` v3/neighbor names; `export_onnx` int64 neighbors; `OnnxPolicy` v3 path | **v3 logits parity (JVM↔Python atol 1e-4)** ✓; provenance inventory check |
| **D10** parity + experiment | `test_parity` (logits + adjacency-fidelity); experiment via `run_loop` | parity ✓; **experiment = the remaining empirical run (AC1)** |

## Gaps
- **D10 experiment (AC1/AC2):** the only remaining item — the multi-hour Medium/Tiny runs + 200-game z-test vs v3's 14.7%, and the rung sweep. All *code* is in place and the pipeline runs end-to-end (smoke onnx_dec=150); this is a compute/wall-clock task, not a code gap.
- Full demand-driven auto-ladder loop: `--rung` (manual per-rung) is implemented; the auto-scaling sweep across rungs (AC2's "chosen by the rule") is run as a per-rung sweep + the rule applied in the results analysis.
