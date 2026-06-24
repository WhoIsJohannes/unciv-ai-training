# Build Output — v4 Structured Encoder

**Branch** `onnx-selfplay-loop` · **Base** master `5ac0a3cf6` · **Mode** BUILD · **Size** L

## Summary
Replaced the v3 rich-critic's permutation-invariant masked-pool spatial encoder with a structured
encoder: categorical **embeddings** + a hex-aware **gather-GNN** over true adjacency + hand-rolled
**self/cross-attention**, behind a 3-rung capacity ladder and a throughput guard. Contract bumped
v2→v3 in lockstep across Kotlin emit/bridge + Python train/export. The frozen training core
(`_optimize_actor_critic`/`compute_gae`, terminal-reward+critic) is untouched — only the `nn.Module`
is swapped. **The full pipeline runs end-to-end** (gen v3 shards → train → export → JVM eval with
live-engine neighbor tensors); the empirical experiment (AC1) is the one remaining (compute-bound) step.

## Files changed (16 commits on the branch)
**Kotlin (emit + bridge):** `SampleSchema.kt` (VERSION 3, CONTRACT_VERSION_STRUCTURED, NEIGHBOR_INPUT_NAMES,
spatial_coords block), `Featurizer.kt` (coords block, map dims, tile-index, construction fix +
`buildSpatialCoords`), `Vocab.kt` (`constructionCode`), `DataPlaneHooks.kt` (vocabCounts in header),
`OnnxPolicy.kt` (v3 gate + int64 live-TileMap neighbor tensors + Timers wrap), `SelfPlayRunner.kt`
(parity adj-parsing, `adjacency-dump`, `bench-onnx`, eval throughput).
**Python:** `hexgraph.py` (NEW adjacency builder), `model.py` (StructuredPolicyValueNet: embeddings +
GNN + attention), `contract.py` (v3 + neighbor names + fail-loud + vocab_counts seam), `features.py`
(neighbor tensors), `dataset.py` (schema channel count + spatial_coords), `export_onnx.py` (int64
neighbor inputs + atomic), `train.py` (train_actor_critic_structured), `run_loop.py` (--variant
structured/--rung dispatch), `unciv_dataplane/schema.py` (SCHEMA_VERSION 3).
**Tests:** `test_hexgraph.py`, `test_structured_smoke.py`, `test_structured_attn.py`,
`test_structured_train.py`, `test_contract_failloud.py`, `ConstructionCodeTest.kt`, `test_parity.py`
(v3 logits + adjacency-fidelity).

## Gate status
- `:core` / `:desktop` compile ✓ · `:tests:test ConstructionCodeTest` (AC7) ✓
- Python suite (non-gradle): **39 passed, 1 skipped** ✓
- Gradle-gated parity: v1 blind + v2 rich + **v3 structured logits parity** + **adjacency-fidelity** = 4 passed ✓
- End-to-end smoke: `--variant structured` 1-round Tiny → **onnx_dec=150, diverged=0** ✓
- Medium feasibility (medium rung): **onnx_dec=97, no OOM, no divergence** ✓
- D8 `bench-onnx`: **verdict=PASS** (ms/decision reported) ✓
- Pre-existing failure (not a regression): `test_same_seed_byte_identical_shards` (v2 AC5 engine limitation, file untouched)

## Test results — see `traceability.md` for the per-deliverable map.

## Open issues / remaining
- **AC1/AC2 experiment** (the only remaining item — compute-bound, not a code gap): budget-constant
  Medium run (structured vs v3's 14.7%, two-proportion z), Tiny non-inferiority, rung sweep with the
  demand-driven rule. Gen-16 dense-batch OOM is being probed; the ladder rejects an OOM rung (the
  small/GNN-only rung is the lightest + tests the locality hypothesis directly).
- Cleanup debt (deferred, non-blocking): `model.py`/`SelfPlayRunner.kt` size; HEX_OFFSETS cross-language
  duplication (guarded by the adjacency-parity test). See `.feature-workflow/cleanup-opportunities.md`.

## Plan fidelity
All 10 deliverables (D1–D10) implemented; D10's experiment is the remaining empirical run. No scope
dropped. Non-negotiables held: frozen seam, opset-17 (no scatter/SDPA — verified in exported op-set),
{tech,policy}+value-train-only heads, terminal-reward+critic, fail-loud lockstep, legality, provenance.

## Security checklist
No secrets; offline RL loop (no network/auth/PII surface); provenance fingerprint gates model loading;
untrusted-ONNX loading out of scope (models self-produced).
