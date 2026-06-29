# Build Output — selfplay-v7-construction

## Summary
First PER-ENTITY control head: per-city PRODUCTION (construction). The learned structured policy now
controls each non-puppet city's production every turn, on top of the civ-global tech/policy heads. Built
on v5 continual + v6 replay (both intact). All pre-run acceptance criteria green; AC#3 (EFFECT) pending the
4-arm Medium run.

## Files changed (20 edited/created + tests + spec)
**Kotlin (engine):**
- `SampleSchema.kt` — VERSION 4→5; `OUTPUT_CONSTRUCTION`/`META_CONSTRUCTION_WIDTH`; `BLOCK_CONSTRUCTION_ACTION`/`_LOGP`.
- `Vocab.kt` — `constructionId(idx)` (inverse of the 0-indexed mask space; NOT 1-indexed constructionCode).
- `Featurizer.kt` — `orderedOwnCities(civ, maxOwnCities)` shared helper (single source of the own_cities order).
- `DataPlaneHooks.kt` — per-city construction decision loop (per-turn control + null/legality guards); recordStep appends 2 VARIABLE blocks; `constructionControlled(city)` guard accessor.
- `ConstructionAutomation.kt` — controlled-civ guard (skip heuristic for policy-driven cities).
- `PolicyProvider.kt` — `chooseConstructionWithLogp` (default abstain) + RandomPolicy uniform-legal.
- `RoutingPolicy.kt` — route construction per-civ. `SampleConfig.kt` — `controlConstruction`.
- `OnnxPolicy.kt` (desktop) — per-city construction memo + row indexing + fallback counter (PR2) + load-time dim cross-check (PR3) + require-output-when-ON.
- `SelfPlayRunner.kt` (desktop) — positional `controlConstruction` arg for gen/eval/bench-onnx; EVAL_RESULT carries `construction_fallbacks`.
**Python (training):**
- `schema.py` SCHEMA_VERSION 4→5; `contract.py` construction output/meta names.
- `model.py` — `construction_head` on `StructuredPolicyValueNet`; `forward(..., with_construction=False)` (FROZEN 3-tuple seam preserved).
- `export_onnx.py` — `construction_logits` output + dynamic city axis (structured only; value still dropped).
- `dataset.py` — load construction blocks per step. `train.py` — construction logp summand (live + stored old_logp) + entropy (acting-gated → no-op-safe) + ragged pad.
- `run_loop.py` — `--control-construction {on,off}` threaded to gen/eval + the structured trainer.
- `run_v7.sh` (new, 4 arms + PR5 bench pre-gate), `analyze_v7.py` (new, ON-vs-OFF + 50% framing), `analyze_v5.py` (+`--control-construction` so ON ceilings use the lever).
**Tests:** `test_v7_construction.py` (new — schema/no-op/parity/ORT), `ConstructionCodeTest.kt` (+constructionId/mask), `test_continual_resume.py` (structured export now emits construction).

## Gate status (all pre-run ACs green)
- **AC#1 LEGALITY** ✅ gen-ON smoke: 1106 decisions / 474 steps, **0 illegal** (recorded ∈ mask; recorded==applied).
- **AC#2 PARITY** ✅ ORT-vs-torch construction logits atol 1e-4 (`test_construction_logits_ort_matches_torch`) + constructionId mask-space round-trip (Kotlin).
- **AC#4 SCHEMA** ✅ v5 lockstep; round-trip with no reader change; v4 shard refused.
- **AC#5 NO-OP** ✅ deterministic zero-summand bit-identical oracle (`test_construction_offarm_is_bit_identical_noop`) + under replay.
- **AC#6 THROUGHPUT** ✅ bench-onnx construction-ON ratio **1.957 ≥ 0.70** (the per-city head is cheaper than the heuristic's cost-effectiveness eval).
- **Determinism** — `test_same_seed_byte_identical_shards` is a PRE-EXISTING flake on this box (verified: clean master also produces non-identical shards). v7 construction blocks matched across runs; NOT a regression.
- **Build**: core+desktop Kotlin compile; `:tests:test` dataplane suite green; python suite green (1 pre-existing determinism flake).
- **End-to-end ON pipeline** validated: gen-ON → dataset construction load → structured trainer construction summand → export with construction head → bench-onnx loads + drives it.

## Pending
- **AC#3 EFFECT** — the 4-arm Medium run (`run_v7.sh`): small/medium × OFF/ON, K=4, seed 4242424, then per-arm 200-game ceiling + `analyze_v7` ON-vs-OFF z-test + 50%-break-even framing. Launched as a resumable background batch (user-approved "drive to completion").

## Plan fidelity
All 6 design decisions (A–F) implemented; all 6 plan-council refinements (PR1–PR6) folded. One build-time
🔄 REFRAMED deviation (D-build-1): per-turn construction control instead of the inert perpetual-only gate
(the lever recorded 0 decisions otherwise — council M29, proven by smoke). Frozen invariants intact:
terminal-only ±1 reward; tech/policy/value heads + dims; PPO clip/value/GAE math; v5 continual + v6 replay.
Heuristic-only asserted: unit movement, promotion, great-person, diplomatic-vote.

## Security checklist
No new secrets (gitleaks N/A — local training repo). New inputs are self-generated shards (single-trust
local training; ONNX-file-signing / replay-poisoning out of scope per council triage). `torch.load(weights_only=True)`
already used by the warm-start path (pickle-RCE pre-mitigated). No PII; no endpoints; no auth surface.
