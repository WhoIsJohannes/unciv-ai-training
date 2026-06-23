# ONNX I/O Contract — self-play policy net

Single source of truth: tensor **names** are fixed (mirrored in Kotlin `SampleSchema.OnnxContract`
and Python `unciv_train.contract`); tensor **widths** are runtime-derived from the generated
`schema.json` (the GnK vocab sizes) and **never hardcoded**. The cross-boundary PARITY test
(`python/tests/test_parity.py`) guards that the JVM `OrtSession` and a Python `onnxruntime` session
produce identical logits (atol 1e-4) for one shared (model, observation).

## Tensors
| Tensor | Name | Shape | dtype | Meaning |
|---|---|---|---|---|
| input | `obs` | `[batch, input_w]` | float32 | `concat(observation block "global", block "acting_civ")` |
| output | `tech_logits` | `[batch, tech_w]` | float32 | per-tech logits (mask + argmax/sample over legal) |
| output | `policy_logits` | `[batch, policy_w]` | float32 | per-policy logits |

`batch` is a dynamic axis. For GnK (resolved empirically at build): `global_w=26`, `acting_w=173`
→ **`input_w=199`**, **`tech_w=80`**, **`policy_w=70`**. These come from `schema.json` `layout[*].len`
(`global`, `acting_civ`, `mask_tech`, `mask_policy`); the trainer reads them via
`contract.dims_from_schema()`.

## Metadata (`metadata_props`) — the provenance gate
`export_onnx.export()` stamps these; the JVM reads them via `session.getMetadata().getCustomMetadata()`
and EVAL **refuses** a model whose values don't match the live engine (criterion 6):

| Key | Value |
|---|---|
| `schema_version` | `SampleSchema.VERSION` (currently `2`) |
| `ruleset_fingerprint` | `RulesetFingerprint.compute(GnK)` (SHA-256 over the canonical vocab) |
| `contract_version` | `1` |
| `input_width` / `tech_width` / `policy_width` | the resolved GnK widths |

## Legality (mask) — train/infer agreement
Both sides set illegal logits to −inf (JVM: filter to the legal index set in `MaskedChoice.choose`;
Python train: `-1e9` before `log_softmax`). The per-step legal mask is the shard's `mask_tech` /
`mask_policy` block — the SAME mask the engine used at decision time (the control path passes that
mask to `chooseIndex`). So a trained action's legality and an inferred action's legality agree by
construction.

## Versioning
Export opset 17 (compatible with onnxruntime ≥ 1.17 — pinned 1.19.2 on both JVM and Python). A
`SampleSchema.VERSION` bump (Kotlin) requires the lockstep `unciv_dataplane.schema.SCHEMA_VERSION`
bump and regeneration — the reader refuses old shards (datasets are perishable by design).
