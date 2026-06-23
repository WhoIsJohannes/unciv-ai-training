# Build Progress — Self-Play Data Plane

Legend: `[ ]` todo · `[~]` in progress · `[x] verified in <file>:<line>` · `[ ] MISSING: <what>`

## Codebase patterns (reuse, don't reinvent)
- Headless game: `UncivGame(true)` + `RulesetCache.loadRulesets(consoleMode=true)` + `GameStarter.startNewGame(GameSetupInfo)`.
- Seeded RNG: `gameContext.stateBasedRandom(label, seed=31)` → `Random(hashOf(label.hashCode(), seed, ctx.hashCode()))`; ctx folds gameId/turns/civID. `civ.state` is the civ's GameContext.
- Ruleset collections are `LinkedHashMap` (deterministic order); `religions` is `ArrayList<String>` → sort.
- Tests: `@RunWith(GdxTestRunner::class)` + `com.unciv.testing.TestGame`.
- Vision: `civ.viewableTiles: Set<Tile>` (current), `tile.isExplored(civ)` (ever).
- JVM writes BIG-endian by default → emitter MUST use `ByteBuffer.order(LITTLE_ENDIAN)`.
- Build/test: `JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home ./gradlew :tests:test` / `:core:compileKotlin`.

## Foundation (format + provenance — standalone)
- [x] `dataplane/SampleSchema.kt` — core/.../dataplane/SampleSchema.kt (VERSION=1, MAGIC, channels) — VERSION + layout constants + dtype tags
- [x] `dataplane/SampleConfig.kt` — core/.../dataplane/SampleConfig.kt (+Caps) — config + Caps + omniscientOpponents/strict/expected*
- [x] `dataplane/ShardFormat.kt` — core/.../dataplane/ShardFormat.kt (LeBuffer, CRC32, magic) — magic, framing, little-endian writer helpers, CRC32
- [x] `dataplane/Vocab.kt` — Vocab.kt (canonical sections incl. tileImprovements; religions sorted) — id↔index from ruleset canonical order (religions sorted) + enums
- [x] `dataplane/RulesetFingerprint.kt` — RulesetFingerprint.kt (SHA-256, length-prefixed) — fingerprint 84edfaa9… — SHA-256 over canonical entity ids + vocab order

## Fairness + features
- [x] `dataplane/FairOpponentModel.kt` — FairOpponentModel.kt (fair encode + masks; leakage/unmet/downgate tests pass) — per-attribute fair encode + availability masks (R1 canonical order, R2 meta-out-of-obs, R5 spy-tech, R9 rank/bucket)
- [x] `dataplane/Featurizer.kt` — Featurizer.kt (scalar+entity-lists+spatial; tile/spy gates) — GLOBAL/ACTING-CIV/CIV/DIPLO/CITY/UNIT tokens + spatial planes; Observation(bytes, accessors)
- [x] `dataplane/LegalActionMasks.kt` — LegalActionMasks.kt (tech/policy/gp/construction/promotion; parity tests pass) — 7 factored heads from engine enumeration

## Policy + emitter + generator + hooks
- [x] `dataplane/PolicyProvider.kt` — PolicyProvider.kt + RandomPolicy (uniform-legal; units→UnitAutomation) — interface + RandomPolicy (uniform-legal; units→UnitAutomation)
- [x] `dataplane/TrajectoryEmitter.kt` — TrajectoryEmitter.kt (one-file-per-shard, LE, CRC32 over framed region, atomic .tmp) — one-file-per-shard, LE, workerId guard, atomic .tmp (R10/R11), calculateChecksum=records-only (R6)
- [x] `dataplane/ScenarioGenerator.kt` — ScenarioGenerator.kt (full range + maxMapRadius cap; spectator; maxAttempts=100) — randomized GnK envelope + guardrails (land≥80, maxAttempts=100 R13, seed≠0, deterministic gameId)
- [x] `dataplane/DataPlaneHooks.kt` — DataPlaneHooks.kt (startup check, registry, header/schema.json, ShardRecorder) — Simulation glue + startup fingerprint/version check

## Engine edits (gated, default-off)
- [x] `NextTurnAutomation.kt:38` — onCivTurn hook (null=unchanged) — verified core compiles — optional `policyProvider: PolicyProvider? = null`
- [x] `GameParameters.kt` — deterministicShuffle=false field — `deterministicShuffle: Boolean = false`
- [x] `GameStarter.kt:322` — gated shuffle(rng) — gated `shuffle(rng)`
- [x] `Simulation.kt` — dataPlane param + recorder open/record/finalize + seed!=0 on data-plane path — accept SampleConfig?+PolicyProvider; open/record/finalize hooks; seed≠0

## Entrypoint + reader + docs
- [x] `desktop/.../DataPlaneGen.kt` — DataPlaneGen.kt — ran end-to-end, wrote shard+schema.json — headless gen main
- [x] `desktop/build.gradle.kts` — :desktop:dataGen JavaExec task — `:desktop:dataGen` JavaExec task
- [x] `python/unciv_dataplane/reader.py` — reader.py (LE, VERSION-refuse, CRC, truncation-tolerant) — 6 tests pass — LE reader, VERSION refuse, CRC verify, provenance, truncation-tolerant
- [x] `python/unciv_dataplane/schema.py` — schema.py (SCHEMA_VERSION mirror, load_schema) — schema.json loader + VERSION mirror
- [x] `python/README.md` — python/README.md (pin-one-version + format spec) — pin-one-version discipline + format spec
- [x] `python/tests/test_reader.py` — test_reader.py — 6 passed, 1 skipped (golden pending) — golden fixture load + refuse/warn

## Tests (acceptance — RED→GREEN)
- [x] MaskParityTests (AC2) — tech+policy parity GREEN (construction internal — covered in core)
- [x] DeterminismTests + DeterminismHarness (AC3) — determinism_sameStateSameBytes GREEN
- [x] LeakageTests (AC4) — RED file exists — leakage city-state-gold + tech-set GREEN
- [x] UnmetTests (AC5) — unmet all-zero/no-tokens GREEN
- [x] DownGateTests (AC6) — no raw demographic float GREEN
- [x] TileGateTests (AC7) — opp city token tile-gated GREEN
- [x] OmniscientAblationTests (AC8) — omniscient flips invisibility GREEN
- [x] ProvenanceTests (AC9, AC10) — fingerprint determinism+drift, schema version GREEN; reader refuses VERSION (python)
- [x] RunToCompletionTests (AC1) — END-TO-END: dataGen→shard→python reader CRC-verified 40 steps
- [x] ScenarioGeneratorGuardrailTests (R13)
