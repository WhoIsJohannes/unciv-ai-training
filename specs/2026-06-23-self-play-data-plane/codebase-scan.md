# Deep Codebase Scan — Unciv Self-Play Data Plane (Phase 2)

Scan depth: deep. All seams traced against the current worktree. Supersedes/extends `codebase-scan-light.md`.

## A. Determinism / RNG

### A1. `GameContext.stateBasedRandom()` — signature & seeding
`core/src/com/unciv/models/ruleset/unique/GameContext.kt:90-91`
```kotlin
@Readonly
fun stateBasedRandom(caller: String, seed: Int=31) =
    Random(hashOf(caller.hashCode(), seed, this.hashCode()))
```
Returns a `kotlin.random.Random` seeded by hash of `caller` label + `seed` + `GameContext.hashCode()`. The context hash folds in `gameInfo.gameId` (136), `gameInfo.turns` (135), `civInfo.civID`, `city.id`, `unit.name`, `tile.position` (137-147). `Civilization.state = GameContext(civInfo = this)` (`Civilization.kt:148`). So `civ.state.stateBasedRandom("policy-<turn>")` derives the policy RNG from `(gameId, civ, turn)` — exactly the prompt's requirement, NO new RNG plumbing needed.

### A2. `GameStarter.startNewGame()` + the shuffle bug
`core/src/com/unciv/logic/GameStarter.kt:43-48`
```kotlin
fun startNewGame(gameSetupInfo: GameSetupInfo): GameInfo = GameStarter(gameSetupInfo).gameInfo
private constructor(private val gameSetupInfo: GameSetupInfo) {
    private val gameInfo = GameInfo()
    private val rng = GameContext(gameInfo = gameInfo).stateBasedRandom("GameStarter")
```
Line 322: `if (newGameParameters.shufflePlayerOrder) otherPlayers.shuffle()` — UNSEEDED. Fix (gated): `otherPlayers.shuffle(rng)` when `deterministicShuffle=true`, else leave `.shuffle()` (interactive play unchanged — see D1).

### A3. `GameInfo` — gameId / nextTurn / turn boundary
`core/src/com/unciv/logic/GameInfo.kt`: `gameId = randomGameId()` (114, String), `gameParameters` (109), `nextTurn()` (346) increments `turns++` (355). Per-civ loop 399-422: `TurnManager(player).startTurn()` → `automateTurn()` → `endTurn()`. Hookable before `startTurn` (399) and after `endTurn` (418).

### A4. `Simulation.kt` — shard structure & threading (EXACT hook lines)
`core/src/com/unciv/logic/simulation/Simulation.kt`:
- (a) shard init: **101-104** (`GameSetupInfo(newGameInfo)` → `GameStarter.startNewGame`)
- (b) per-turn record: **108-114** (`gameInfo.nextTurn()` / `step.saveTurnStats(gameInfo)`)
- (c) finalize: **118-130** (victory check, record winner, final stats)
- threading: `launch(Dispatchers.Default + CoroutineName("simulation-$threadId"))` (99) → one CoroutineName per worker, read via `coroutineContext[CoroutineName]`. `@Synchronized` on `add()`/`updateCounter()` (145, 150).
- `SimulationStep` fields: `turns`, `victoryType:String?`, `winner:String?`, `currentPlayer:String`, per-civ turn-stat maps.

> NOTE: spec's hook line numbers (97/134/141) were approximate; use 101-104 / 108-114 / 118-130.

## B. Action-mask candidate enumeration (reuse — do NOT re-derive)
- **City construction:** `CityConstructions.getBuildableBuildings(): Sequence<Building>` (109) + `getConstructableUnits()` (113), both filter `isBuildable(this)`. Identify by `name` → `ruleset.buildings[name]` / `ruleset.units[name]`.
- **Great person:** `GreatPersonManager.getGreatPeople()` (108-112) = ruleset units `isGreatPerson`, civ-equivalent, available; candidate trigger via `getNewGreatPerson()` (64) vs points.
- **Diplomatic vote:** voter picks among `votingCiv.diplomacyFunctions.getKnownCivsSorted(false)` + Abstain (`DiplomaticVotePickerScreen.kt:21-34`). Excludes unmet/self.
- **Promotion:** `UnitPromotions.getAvailablePromotions(): Sequence<Promotion>` (182) filtered by `isAvailable()` (not-already, unitType matches, prereqs).
- **Unit-intent + target:** `UnitAutomation.automateUnitMoves(unit)` (34) dispatches to `CivilianUnitAutomation.automateCivilianUnit` (42), `AirUnitAutomation.automateFighter/Missile/Bomber` (65-70), and ground tries `tryAccompanySettlerOrGreatPerson` (74) → `tryAttacking` (89, targets via `TargetHelper`) → `HeadTowardsEnemyCityAutomation.tryHeadTowardsEnemyCity` (94) → `tryExplore` (110). RandomPolicy routes unit intents into these sub-routines.

## C. Fair-info model sources
### C10. Civilization
`getKnownCivs(): Sequence<Civilization>` (367), `getDiplomacyManager(civID)` (352), `getMilitaryMight(): Int` (856, cached), `getHappiness(): Float` (461), `getCapital(): City?` (387), `cities: ArrayList<City>`. `RankingType` (`ui/screens/victoryscreen/RankingType.kt:8-33`) = Score, Population, Growth, Production, Gold, Territory, Force, Happiness, Technologies, Culture — all via `getStatForRanking(category): Int` (836-850).
### C11. DiplomacyManager
`core/.../diplomacy/DiplomacyManager.kt`: `opinionOfOtherCiv(): Float = diplomaticModifiers.values.sum()` (314-320), `relationshipLevel(): RelationshipLevel` (379-385; Unforgivable…Ally), `diplomaticModifiers: HashMap<DiplomaticModifiers,Float>`, `DiplomacyFlags` enum (53-101: War/DefensivePact/Denouncement/… with turn counters).
### C12. Politics web (third-party masking to MIRROR)
`GlobalPoliticsOverviewTable.kt`: era `civ.tech.era.name` (146), policy branches (150+), wonders (132), score via `getStatForRanking(Score)`. Identity mask `getCivName(otherciv)` (185-190): returns name iff `viewingPlayer.knows(otherciv) || otherciv==viewingPlayer` else `"an unknown civilization"`. **The featurizer must apply the same met-gate for any third-party identity.**
### C13. Down-gate leak (confirm)
`VictoryScreenDemographics.kt:32-39` ranks ALL alive major civs with RAW per-civ values (no met-gate); `Civilization.statsHistory` (`CivRankingHistory`) exposes the full time-series for all civs. → DOWN-GATE to rank/bucket + identity-free best/avg/worst; never emit the raw per-civ float for these categories.
### C14. Victory progress
`VictoryManager.kt`: `currentsSpaceshipParts: Counter<String>` (20, sic typo), `hasEverWonDiplomaticVote` (21). `Victory.requiredSpaceshipPartsAsCounter` (503-508), milestones incl. CompletePolicyBranches/WorldReligion (513-521). Completed-branches via `policies.adoptedPolicies.count{ !Policy.isBranchCompleteByName(it) }`. **`originalMajorCapitalsOwned` DOES NOT EXIST** in this codebase — domination progress must be derived (e.g. count cities flagged as original capitals owned vs founder being a major) or omitted with denom-mask=0.
### C15. Espionage
`EspionageManager.kt`: `getSpiesInCity(city): List<Spy>` (82) — spy-present gate (encode boolean only). `getTechsToSteal(otherCiv): Set<String>` (72-80) — returns the tech NAME SET. Fair model encodes only `.isNotEmpty()` (the prompt's deliberate choice; the names are HIDDEN). `Spy` state machine (isSetUpForEspionage / Spy.action) NOT fully read — verify in build for the "set-up spy in O's city" gate.
### C16. City surface bits (tile-gated)
`City.kt`: `population: CityPopulationManager` (80, `.population.population` for SIZE), `health: Int` (77, max 200), `religion: CityReligionManager` (83, majority), `isPuppet` (103), `isInResistance()` (263), `isBeingRazed` (100), air units via `getCenterTile().airUnits` (462). Defensive strength: no direct field — compute via `CityCombatant(city)` (verify in build).

## D. Vocab + provenance
### D17. Ruleset collections
`Ruleset.kt:126-155`: all `LinkedHashMap` (beliefs, buildings, eras, nations, policies, policyBranches, technologies, terrains, tileImprovements, tileResources, units, unitPromotions, unitTypes, victories, cityStateTypes, …). EXCEPTION: `religions: ArrayList<String>` (137) — **sort canonically before building the vocab** (determinism + fingerprint stability; security finding). `Era.eraNumber: Int` is an arbitrary designer field — canonical era order = iterate `Ruleset.eras` (LinkedHashMap), NOT sort by eraNumber. Enums for vocab: `ResourceType` (Luxury/Strategic/Bonus), `TerrainType` (Land/Water/TerrainFeature/NaturalWonder), `RankingType` (10).
### D18. Provenance
`UncivGame.VERSION.{text,number}` (`Versioning.kt`) ← `BuildConfig.appVersion` ("4.20.15") / `appCodeNumber`. `CURRENT_COMPATIBILITY_NUMBER = 4` (Versioning.kt:47) — mirror for `SampleSchema.VERSION` bump discipline. **No git-SHA build constant** → git SHA is best-effort at gen-time (`git rev-parse HEAD` from the desktop runner; omit if unavailable). Ruleset: `RulesetCache.getComplexRuleset(mods=linkedSetOf(), "Civ V - Gods & Kings")`; `BaseRuleset.Civ_V_GnK.fullName`.

## E. Entry point + build + collisions
### E19. Desktop entrypoint / headless init
`desktop/src/com/unciv/app/desktop/DesktopLauncher.kt:40-56`: `UncivGame(true)` (headless) → `UncivGame.Current = game` → `settings = GameSettings()` → `RulesetCache.loadRulesets(consoleMode=true)` → `GameStarter.startNewGame(gameSetupInfo)`. New `:desktop:` JavaExec task mirrors the user's `simBench` pattern (in main worktree).
### E20. TestGame helper — CONFIRMED EXISTS (light scan wrong)
`tests/src/com/unciv/testing/TestGame.kt`: `makeHexagonalMap(radius)`/`makeRectangularMap(w,h)`, `addCiv()`, `addBarbarianCiv()`, `addCity(civ,tile,…)`, `addUnit(name,civ,tile)`, ruleset-mod creators (`createBaseUnit/Building/Resource/Wonder/Policy/PolicyBranch/UnitPromotion`), `addReligion(foundingCiv)`. Fields: `ruleset`, `gameInfo`, `tileMap`. Perfect for leakage/tile-gate/unmet/down-gate acceptance tests (meet/unmeet civs, set vision, place cities/units on specific tiles).
### E21. Collisions / thread-safety
Existing `com.unciv.logic.simulation.*` (Simulation, SimulationStep, MutableInt). New `com.unciv.logic.simulation.dataplane.*` — no clash. Reuse `@Synchronized`/`@Volatile`; emitter design = ONE file per shard keyed by CoroutineName → no shared writer → structural thread-safety (preferred over @Synchronized which is unsafe across coroutine suspension — architect finding).

## Top surprises
1. **Determinism is nearly free** — `stateBasedRandom(label)` already derives from `(gameId, turns, civID, …)`. Policy RNG = `civ.state.stateBasedRandom("policy-$turn")`; the only gap is the GameStarter:322 unseeded shuffle (gate-fix). No new RNG framework.
2. **`originalMajorCapitalsOwned` does NOT exist** — spec references a non-existent field. Domination victory numerator must be derived from city original-capital flags, or emitted with denom-mask=0. (Plan must decide.)
3. **No git-SHA build constant** — provenance pins on `UncivGame.VERSION.number` + `RulesetFingerprint`; git SHA is best-effort at gen-time only.
4. **`getTechsToSteal` returns the tech NAME SET, not a flag** (matches council security/domain findings). Fair model encodes ONLY `.isNotEmpty()`; the names join the HIDDEN/never-encoded set. Same treatment for `currentsSpaceshipParts` (encode count, never part names).
5. **`@Synchronized` is unsafe across coroutine suspension** — the emitter should be one-file-per-shard (per CoroutineName) with no shared writer, making thread-safety structural rather than lock-based.
6. **`religions` is the lone `ArrayList` ruleset collection** — must be sorted canonically before vocab build, else vocab indices could vary (fingerprint instability / covert ordering channel).
7. **Demographics/statsHistory leak is real and broad** — `VictoryScreenDemographics` + `statsHistory` expose raw per-civ values for ALL alive majors with no met-gate. The down-gate (rank/bucket + identity-free aggregates) is mandatory and is the heart of the fairness work.
8. **Max-map decision still open** — Huge (r40 ≈5k tiles) vs Civ5Huge (r128 ≈50k) sets the spatial cap and `SampleSchema.VERSION`. Plan must pick (verify which is generator-selectable in standard GnK setup).
