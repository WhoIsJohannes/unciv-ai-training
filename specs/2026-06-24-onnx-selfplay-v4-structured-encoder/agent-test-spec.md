# Test spec — v4 Structured Encoder (test_mode = integration)

Repo has no agentic-test harness (no `test:agentic*`, no `agentic-tests/`) → **integration** mode:
pytest (Python) + Gradle JUnit (Kotlin), the repo's existing frameworks.

## RED-first (written now, fails until built)
- **`python/tests/test_hexgraph.py`** — the adjacency builder (D3), the feature's heart. RED now (no
  `hexgraph` module → import error). Asserts: shapes/int dtype, in-bounds indices (FND-0025), interior tile
  has all 6 neighbors in frozen clock order, edge tile masks absent neighbors, world-wrap doesn't lose neighbors.

## Full matrix (implemented during build — D10 / AC7)
1. **Adjacency parity** — Python `build_neighbor_graph` == Kotlin pure `buildNeighborIndex` over a synthetic
   map (hex + rectangular width/2≠radius, world-wrap + ragged edges). (FND-0036)
2. **Engine fidelity** — Kotlin pure `buildNeighborIndex` == live `TileMap.getIfTileExistsOrNull` over a real
   map (so inference, which uses the live engine, is transitively covered).
3. **Shard-roundtrip** — write a u8 `spatial` + f32 `spatial_coords` block with negative & >255 coords, reload
   via `unciv_dataplane.reader`, build adjacency, assert == live-TileMap reference (catches the u8-storage
   contract the idealized-float logits parity misses).
4. **Rich-logits parity** — extend `test_jvm_python_rich_logits_match` to the wider multi-tensor input incl.
   `neighbor_index`/`neighbor_mask` (atol 1e-4, logits incl.); a shared pure neighbor-builder lets
   `parityRunRich` inject the adj tensors (no TileMap there).
5. **Contract-version-mismatch refusal** — old shard / wrong fingerprint → fail-loud, never silently mis-decode.
6. **Construction-collision unit test (Kotlin, AC7)** — building#k code ≠ unit#k code after the D1.4 fix.
7. **Legality** (existing `OnnxPolicyLegalityTest`) stays green — masked heads never pick illegal.
8. **Export smoke** — small-rung ONNX export under opset 17 (gather-GNN + hand-rolled attention) loads in ORT.

## Verify RED
`cd python && python -m pytest tests/test_hexgraph.py` → fails (ModuleNotFoundError: unciv_train.hexgraph).
