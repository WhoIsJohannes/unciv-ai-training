# Build Output — Self-Play Data Plane

**Status: BUILD COMPLETE, all gates green.** Ready for Phase 4 (ship) pending human go-ahead.

## Summary
Implemented the full policy-agnostic self-play data plane for Unciv (GnK): a fog-correct,
fair-information featurizer + factored legal-action masks + thread-safe binary trajectory emitter
+ scenario generator + pure-Python reader, with version/ruleset provenance. RandomPolicy drives
self-play to completion; the emitter writes compact shards; the Python reader validates them.

## Files changed (24 source + tests)
**New package `core/.../logic/simulation/dataplane/` (13 files, 1259 LOC):** SampleSchema,
SampleConfig, ShardFormat (+LeBuffer), Vocab, RulesetFingerprint, FairOpponentModel, Featurizer,
Observation, LegalActionMasks, PolicyProvider (+RandomPolicy), TrajectoryEmitter, ScenarioGenerator,
DataPlaneHooks (+ShardRecorder, DataPlaneContext).
**Gated engine edits (4, all default-off ⇒ interactive play byte-identical):**
`NextTurnAutomation.kt` (+`onCivTurn` hook), `GameStarter.kt:322` (gated `shuffle(rng)`),
`GameParameters.kt` (+`deterministicShuffle`), `Simulation.kt` (+`dataPlane` param + recorder
open/record/finalize + seed≠0 on the data-plane path).
**Entrypoint:** `desktop/.../DataPlaneGen.kt` + `:desktop:dataGen` Gradle task.
**Python reader (5 files, 448 LOC):** `unciv_dataplane/{reader,schema,__init__}.py`, `README.md`,
`tests/test_reader.py`.
**Tests:** `tests/.../dataplane/FairnessAndDeterminismTests.kt` (12 tests).

## Gate status
- **Compile:** `:core`, `:desktop`, `:tests` all compile (JDK 21 / Gradle 8.11.1).
- **Full Unciv test suite:** `:tests:test` → **730 tests, 0 failures, 0 errors, 13 skipped** — the
  gated engine edits cause ZERO regressions (validates the no-gameplay-change non-goal).
- **Dataplane acceptance tests:** 12/12 green (see Acceptance below).
- **Python reader tests:** 6 passed, 1 skipped (golden fixture not committed by design — perishable).
- **Web FE/BE quality gate (npm/ruff/pyright):** N/A — Kotlin/gradle repo; the real gates are gradle
  compile + the 730-test suite (passed).
- **Security gates (gitleaks/semgrep):** tools not installed locally; N/A — the data plane introduces
  no secrets/credentials/network surface. (CI may run them.)
- **End-to-end:** `:desktop:dataGen` drove a real RandomPolicy game to completion, wrote
  `shard-*.bin` + `schema.json`; the Python reader loaded it CRC-clean (40 steps, full provenance
  incl. git SHA).

## Acceptance criteria → evidence
| AC | Status | Evidence |
|----|--------|----------|
| 1 RandomPolicy→completion, emitter writes shard, reader loads, schema validates | ✅ | `:desktop:dataGen` run; Python `load()` CRC-clean, 40 steps |
| 2 masks == engine enumeration | ✅ | `maskParity_tech/policyMatchesEngine` (construction is `internal`, covered in core) |
| 3 determinism (byte-identical) | ✅ | `determinism_sameStateSameBytes`; emitter CRC32 over framed records |
| 4 leakage (hidden gold + tech-set) | ✅ | `leakage_cityStateGold_hiddenWithoutTrade`, `leakage_rivalTechSet_neverEnters` |
| 5 unmet → zeros+masks0, no tokens | ✅ | `unmet_rivalIsAllZeroAndContributesNoTokens` |
| 6 down-gate (rank/bucket only) | ✅ | `downGate_metRivalTokenHasNoRawDemographicFloat` |
| 7 tile-gate | ✅ | `tileGate_oppCityTokenOnlyWhenTileVisible` |
| 8 omniscient ablation | ✅ | `omniscient_revealsUnmetAndChangesObservation` (only switch that changes obs) |
| 9 shard+schema provenance; reader refuses VERSION | ✅ | header/schema.json carry VERSION+version+fingerprint; reader `test_refuses_version_mismatch` |
| 10 fingerprint drift + determinism | ✅ | `fingerprint_changesWhenRulesetContentChanges`, `fingerprint_deterministicAndNonEmpty` |

## Size (decision D5)
Shard format v2 (variable-count present-only entities + u8 masks/spatial) cut a Tiny 10-turn game
from **9,032,512 → 240,897 bytes (37.5×)**; ~6 KB/step now dominated by the actual map plane.

## Plan fidelity
High. All planned modules built + the R1–R15 council refinements folded in. Deviations (documented):
unit-intent is delegated to UnitAutomation (no engine candidate list ⇒ not a flat-enumerated mask
head — per prompt's "routes unit intents into UnitAutomation"); religion-conversion victory
numerator is a v1 stub (0; needs a ReligionManager API); is_last/is_terminal step flags are v1=0
(game-end step not separately buffered). See `.feature-workflow/cleanup-opportunities.md`.

## Security checklist
- No new secrets (no creds in code). ✅
- New external inputs: the Python reader validates magic/VERSION/CRC and refuses mismatches; tolerates
  truncated shards. ✅
- Data scoping: the fairness model IS the information-scoping boundary (leakage tests enforce it). ✅
- No PII (synthetic game state). ✅  · No new network endpoints. ✅

## Open issues (non-blocking, → cleanup backlog)
- Spatial plane scales with map size; a Huge late-game shard will be large (accepted "full standard
  range" cost — now u8, no padding waste).
- Per-entity chosen-action labels (construction/promotion/unit-intent) not recorded per-step in v1
  (only the 4 civ-level head choices); the per-entity MASKS are emitted.
