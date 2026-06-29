# Progress — selfplay-v7-construction (Phase 3 Build)

## Codebase patterns (reuse)
- VARIABLE block = `[u16 count] + count×perItem` (Observation.writeBlock); `varU8`/`varF32` helpers.
- Masked sampling: `MaskedChoice.chooseWithLogp` (single RNG draw, stable log-softmax). Reuse for construction.
- Controlled-civ heuristic skip: `if (DataPlaneHooks.controls(civ)) return` (NextTurnAutomation.adoptPolicy:302).
- own_cities order: `x.cities.sortedBy{it.id}` capped at `caps.maxOwnCities` → centralize in `orderedOwnCities`.
- Net (experiments): `StructuredPolicyValueNet` + `train_actor_critic_structured` (NOT RichPolicyValueNet).
- Python→JVM flag = positional arg in `SelfPlayRunner.gen/eval` → `SampleConfig`.
- Fail-loud contract pattern: `test_contract_failloud.py`.

## Build items (from plan.md)

### Group 1 — Foundation (schema + Vocab)
- [ ] SampleSchema.VERSION 4→5; add OUTPUT_CONSTRUCTION="construction_logits" + block name consts
- [ ] schema.py SCHEMA_VERSION 4→5 (lockstep)
- [ ] Vocab.constructionId(idx): String? — inverts 0-indexed mask space (idx<buildingCount→building; else unit)

### Group 2 — Featurizer ordering
- [ ] Featurizer.orderedOwnCities(civ): List<City> shared helper; reuse in mask build + decision loop + recorder

### Group 3 — Control seam (Kotlin)
- [ ] PolicyProvider.chooseConstructionWithLogp(civ, city, cityRow, legalMask, turn) default uniform-legal
- [ ] SampleConfig.controlConstruction: Boolean
- [ ] SelfPlayRunner gen/eval parse positional arg[8] → SampleConfig

### Group 4 — DataPlaneHooks + ConstructionAutomation
- [ ] chooseAndApply: per-city construction loop (decide-when-idle; null-guard PR4; record idx/logp)
- [ ] recordStep: append construction_action + construction_logp VARIABLE blocks
- [ ] fallback counter (PR2)
- [ ] ConstructionAutomation.chooseNextConstruction: controlled-civ + pre-filled guard

### Group 5 — OnnxPolicy
- [ ] Memo holds per-city construction logits; ONE forward/turn; index row=cityRow; MaskedChoice sample
- [ ] fallback (missing/NaN/no-legal → −1) + dim cross-check vs constrW (PR3)

### Group 6 — Net (Python)
- [ ] StructuredPolicyValueNet: capture own_cities pre-pool embeddings; construction_head; forward → 4-tuple
- [ ] export_onnx: construction output + dynamic city axis (value still dropped)

### Group 7 — Train (Python)
- [ ] dataset.py: load construction_action/logp + mask_construction; pad to max Ncities; TrainTrajectory fields
- [ ] train.py: live logp += Σ_cities _masked_logp(construction); stored old_logp includes it; entropy term

### Group 8 — Driver
- [ ] run_loop.py: --control-construction flag → gradle positional arg; fresh OUT_ROOT/empty replay (PR6)
- [ ] run_v7.sh: 4 arms (small/medium × OFF/ON), K=4, seed 4242424, resumable
- [ ] analyze_v7.py: per-rung ON-vs-OFF z-test + vs 50%; ON-arm fallback assertion (PR2)

### Group 9 — Tests
- [ ] FairnessAndDeterminismTests: maskParity_construction + determinism + constructionId round-trip
- [ ] OnnxPolicyLegalityTest: per-city legality (recorded==applied, in-mask)
- [ ] Python: no-op zero-summand (AC#5 deterministic oracle PR1); test_v7_construction green

### Group 10 — Gates
- [ ] gradle :tests:test green; pytest green
- [ ] bench-onnx PRE-run throughput ≥70% (PR5)
