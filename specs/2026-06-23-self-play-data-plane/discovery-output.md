# Discovery Output — Self-Play Data Plane for Unciv

**Mode:** BUILD | **Size:** L | **Date:** 2026-06-23 | **Branch:** `self-play-data-plane`
**Base commit:** `4debb125f` | **Worktree:** `/Users/j/Unciv-self-play-data-plane`

## Feature (1-line)
A policy-agnostic, deterministic self-play **data plane** for the Unciv engine: a fog-correct featurizer + factored legal-action masks + thread-safe binary trajectory emitter + scenario generator + pure-Python reader, with a principled **fair opponent-information model** (policy sees only what a human could) and **version/ruleset provenance** (datasets self-identifying, perishable/regenerable — never migrated). NO neural net, NO training loop — data plane only.

## Context category & docs
- **Category:** backend (architecture, determinism, thread-safety, information-flow correctness; no UI).
- **Routed invariant docs:** N/A — the installed `docs/invariants/*.md` are generic boilerplate for an unrelated Next.js/FastAPI eval platform. Real conventions = Unciv `docs/Developers/Coding-standards.md` + existing Kotlin idioms. The "security" surface here is the prompt's own **info-flow/leakage fairness model**.
- **Domain preset:** data-pipeline → council roles `practitioner` (always-on), `data_privacy_legal` (maps to the leakage/fairness lens), `cost_efficiency` (data-gen throughput / tensor size).

## User decisions (Round 1)
1. **Determinism scope = GATE TO SIM PATH.** Add a `deterministicShuffle: Boolean = false` flag threaded through `GameSetupInfo`/`GameParameters`; only the data-plane/sim path enables it. `GameStarter.kt:322` shuffle routes through `gameContext.stateBasedRandom("GameStarter")` ONLY when the flag is set — interactive play stays byte-identical (honors "no gameplay changes" non-goal). Policy RNG reuses the engine's existing `stateBasedRandom(label)` keyed on `(gameId, civ, turn)`. Generator enforces `seed != 0` (Simulation currently sets `mapParameters.seed = 0` — a determinism bug to fix on the sim path).
2. **Scenario envelope = FULL STANDARD RANGE.** Generator randomizes across all map sizes (incl. Large/Huge) and up to Unciv's max players, with guardrails (land-tiles/civ ≥ 80; MapRegions "too many players" retry). Accepted tradeoff: larger tensors, slower generation, maximally general data.

## Light scan summary (16 seams verified — full report: `codebase-scan-light.md`)
- **PolicyProvider seam:** `NextTurnAutomation.automateCivMoves(civInfo, tradeAndChangeState)` at `automateCivMoves` (NextTurnAutomation.kt:38–40). No provider exists yet → add injection here; default `RandomPolicy` routes unit intents into existing `UnitAutomation` sub-routines.
- **Emitter seam:** `Simulation` (Simulation.kt) — constructor 21–27, init 55–69, per-turn loop 108–114, finalize/victory 118–130. Threading: `launch(Dispatchers.Default + CoroutineName("simulation-$threadId"))` (line 99) → **one shard per `CoroutineName`**. `@Synchronized` already used for shared counters.
- **Determinism:** engine already has a pervasive seeded RNG: `GameContext(...).stateBasedRandom(label)` (used by `MapRegions.assignRegions`, etc.). Only `GameStarter.kt:322` `otherPlayers.shuffle()` is unseeded. `Simulation.start()` sets `seed = 0`.
- **Fair-info sources (all confirmed):** `Civilization.knows()` (384–385); `getStatForRanking(RankingType)` (836–850) → Score/Population/Growth(food)/Production/Gold/Territory/Force/Happiness/Technologies/Culture; `calculateTotalScore()` (912); `shouldHideCivCount()` (1246–1253); `hasExplored(tile)`=`tile.isExplored(civ)` (201); `gold` (131); era via `tech.era` (TechManager:30); `tech.researchedTechnologies.size` (847, tech COUNT — list NEVER encoded); `tech.techsInProgress` (HIDDEN).
- **Trade slots:** `civInfo.getPerTurnResourcesWithOriginsForTrade()`, `getStockpiledResourcesForTrade()`, `gold`, `stats.statsForNextTurn.gold` (TradeLogic.kt:60–74).
- **Espionage:** `EspionageManager.getSpiesInCity(city)` (82) for spy-present gate; `getTechsToSteal(otherCiv)` (72–80) for the tech-steal-availability bit (flag only, never the list).
- **Vision/fog:** `civ.viewableTiles: Set<Tile>` (91) for "currently visible"; `tile.isExplored(civ)` for "ever explored". No `VisibleTilesManager`.
- **Spatial key:** `Tile.zeroBasedIndex` (101) = index into `TileMap.tileList: ArrayList<Tile>` (39). Map sizes: predefined Tiny(r10)…Huge(r40 ≈5k tiles); larger `Civ5Huge(r128 ≈50k)` set exists — **design must determine which are generator-selectable** to set the spatial cap.
- **Vocab source:** all `Ruleset` collections are `LinkedHashMap` (techs/units/buildings/policies/policyBranches/tileResources/unitPromotions/terrains/nations/eras/victories…) → deterministic canonical order for vocab + RulesetFingerprint. `religions` is an `ArrayList<String>`.
- **Candidate-enum APIs (masks — reuse, do NOT re-derive):** Tech `TechManager.canBeResearched(name)` (176); Policy `PolicyManager.isAdoptable(policy, checkEra)` (221); Promotion `UnitPromotions.getAvailablePromotions(): Sequence<Promotion>` (182); City-construction list, Great-person, Diplomatic-vote → **locate in design** (not found in light scan).
- **Provenance:** `UncivGame.VERSION.{text,number}` (Versioning.kt). Ruleset via `RulesetCache.getComplexRuleset(mods=linkedSetOf(), "Civ V - Gods & Kings")`; `BaseRuleset.Civ_V_GnK.fullName = "Civ V - Gods & Kings"`.
- **Victory progress:** `VictoryManager.currentsSpaceshipParts: Counter<String>` (sic — typo persisted in saves), `hasEverWonDiplomaticVote`; `Victory.requiredSpaceshipPartsAsCounter`, milestones (CompletePolicyBranches, WorldReligion, …); `originalMajorCapitalsOwned` to locate in design.
- **Tests:** module `tests/` (`./gradlew tests:test`), JUnit + Mockito + `@RunWith(GdxTestRunner::class)`. **`TestGame` helper EXISTS** at `tests/src/com/unciv/testing/TestGame.kt` (scan agent wrongly said absent) — reuse for headless game setup in acceptance tests.
- **Serialization:** libGDX `com.badlogic.gdx.utils.Json` for saves (NOT kotlinx). Binary emitter will be a custom little-endian format (not Json) for the trajectory shards.
- **Entry point:** `desktop/src/com/unciv/app/desktop/` (the user's `SimBenchmark.kt` + `simBench` Gradle task live in the MAIN worktree as reference) → add a `:desktop:` JavaExec task for the data-gen runner.

## Reference context (main worktree, not in this branch)
User's uncommitted `SimBenchmark.kt` + `desktop/build.gradle.kts` `simBench` task: a headless `Simulation` throughput benchmark (Medium map, 6 majors + 6 city-states, GnK, A* on/off, JSON-snapshot cost). Motivates this feature and demonstrates the exact `Simulation`/`GameStarter`/`RulesetCache` setup path the generator will reuse.

## Open questions → for Phase 2 design
- Exact max-map cap (Huge r40 vs Civ5Huge r128) that the generator can produce, which locks the spatial-tensor cap and `SampleSchema.VERSION`. Leading: spatial plane sized to each game's actual `tileList` length (self-described in provenance), per-entity tokens fixed-width to the max envelope.
- Per-tile spatial channel set (leading: terrain/feature/resource/improvement/road+river/owner/city-center/unit-presence+type+owner/visibility-state — fog-masked).
- City-construction / great-person / diplomatic-vote candidate-enumeration entry points.
- Binary shard format details (header layout, per-step record framing, checksum) + `schema.json` generation (emit from Kotlin at sim start).
- Fixed-width caps: max major civs, max city-states, max cities-per-civ, max units (token-group dims).

## Phase 2 build notes (must hold)
- Fairness model is the spec (per-attribute encoding table); `omniscientOpponents: Boolean = false` is the ONLY switch that changes observations.
- VERSION bump rule mirrors `CURRENT_COMPATIBILITY_NUMBER` (Versioning.kt); layout-affecting ruleset changes caught via RulesetFingerprint.
- 10 acceptance tests: functional (run-to-completion, mask-parity, determinism/checksum + golden parity), fairness (leakage, unmet, down-gate, tile-gate, omniscient-ablation), provenance (shard provenance + reader-refuse, fingerprint-drift).
