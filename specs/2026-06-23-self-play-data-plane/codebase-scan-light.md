# Unciv Codebase Scan - Self-Play Data Plane Feature
**Date:** 2026-06-23  
**Scan Depth:** Medium - Verify seam references from specification

---

## 1. NextTurnAutomation.automateCivMoves

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/logic/automation/civilization/NextTurnAutomation.kt`  
**Lines:** 38–40

### Full Signature
```kotlin
object NextTurnAutomation {
    fun automateCivMoves(civInfo: Civilization,
                         tradeAndChangeState: Boolean = true): Unit = timeThis("automateCivMoves") {
```

### First 15 Lines
```kotlin
38  fun automateCivMoves(civInfo: Civilization,
39                       /** set false for 'forced' automation, such as skip turn */
40                       tradeAndChangeState: Boolean = true): Unit = timeThis("automateCivMoves") {
41      if (civInfo.isBarbarian) return BarbarianAutomation(civInfo).automate()
42      if (civInfo.isSpectator()) return // When there's a spectator in multiplayer games...
43
44      respondToPopupAlerts(civInfo)
45      TradeAutomation.respondToTradeRequests(civInfo, tradeAndChangeState)
46
47      if (tradeAndChangeState && civInfo.isMajorCiv()) {
48          if (!civInfo.gameInfo.ruleset.modOptions.hasUnique(UniqueType.DiplomaticRelationshipsCannotChange)) {
49              DiplomacyAutomation.declareWar(civInfo)
50              DiplomacyAutomation.offerPeaceTreaty(civInfo)
```

**Status:** ✓ Correct. No `PolicyProvider` injection evident here (would be via dependency injection at caller level). The function takes only `civInfo` and `tradeAndChangeState`.

---

## 2. Simulation Class & Threading Model

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/logic/simulation/Simulation.kt`  
**Lines:** 21–27 (constructor), 55–69 (init), 85–142 (start method)

### Class Definition & Constructor
```kotlin
@ExperimentalTime
class Simulation(
    private val newGameInfo: GameInfo,
    val simulationsPerThread: Int = 1,
    private val threadsNumber: Int = 1,
    private val maxTurns: Int = 500,
    private val statTurns: List<Int> = listOf()
)
```

### Init Block (lines 55–69)
```kotlin
init {
    for (civ in majorCivs) {
        this.numWins[civ] = MutableInt(0)
        winRateByVictory[civ] = mutableMapOf()
        for (victory in UncivGame.Current.gameInfo!!.ruleset.victories.keys)
            winRateByVictory[civ]!![victory] = MutableInt(0)
        winTurnByVictory[civ] = mutableMapOf()
        for (victory in UncivGame.Current.gameInfo!!.ruleset.victories.keys)
            winTurnByVictory[civ]!![victory] = MutableInt(0)
    }
    initHash(summaryStatsPop)
    initHash(summaryStatsProd)
    initHash(summaryStatsCities)
    initHash(summaryStatsAvgPop)
}
```

### Threading Model (lines 85–104)
```kotlin
fun start() = runBlocking {
    startTime = System.currentTimeMillis()
    println("Starting new game with major civs: ...")
    newGameInfo.gameParameters.shufflePlayerOrder = true
    
    Timers.singleton.startTiming()
    val jobs = (1..threadsNumber).map { threadId ->
        launch(Dispatchers.Default + CoroutineName("simulation-$threadId")) {
            repeat(simulationsPerThread) {
                val step = SimulationStep(newGameInfo, statTurns)
                val gameSetupInfo = GameSetupInfo(newGameInfo)
                gameSetupInfo.mapParameters.seed = 0
                val gameInfo = GameStarter.startNewGame(gameSetupInfo)
                // ... per-turn simulation loop
```

**Key Findings:**
- **Coroutine Model:** Uses `launch(Dispatchers.Default + CoroutineName("simulation-$threadId"))` – one CoroutineName per worker thread (line 99).
- **Per-Step/Per-Turn:** Loop at lines 108–114 iterates `statTurns` and calls `gameInfo.nextTurn()` and `step.update(gameInfo)`.
- **Finalize:** After loop ends, checks victory condition (lines 118–130), records winner/turns.
- **Thread Safety:** Uses `@Synchronized` decorator on `add()` and `updateCounter()` (lines 145, 150).

---

## 3. GameStarter – Deterministic Shuffle

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/logic/GameStarter.kt`  
**Lines:** 322

### Shuffle Logic
```kotlin
// Shuffle Major Civs
if (newGameParameters.shufflePlayerOrder) otherPlayers.shuffle()
```

**Context (lines 318–327):**
```kotlin
// ensure Spectators always first players
val spectators = chosenPlayers.filter { it.chosenCiv == Constants.spectator }
val otherPlayers = chosenPlayers.filterNot { it.chosenCiv == Constants.spectator }.toMutableList()

// Shuffle Major Civs
if (newGameParameters.shufflePlayerOrder) otherPlayers.shuffle()

chosenPlayers.clear()
chosenPlayers.addAll(spectators)
chosenPlayers.addAll(otherPlayers)
```

**Status:** ⚠️ **CONCERN** – `otherPlayers.shuffle()` uses Kotlin's default random (JVM Math.random), NOT the game's seeded RNG. For determinism, this should use:
```kotlin
if (newGameParameters.shufflePlayerOrder) otherPlayers.shuffle(rng)
```
where `rng = GameContext(gameInfo = gameInfo).stateBasedRandom("GameStarter")` (line 48).

---

## 4. MapRegions – "Too Many Players" Retry

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/logic/map/mapgenerator/mapregions/MapRegions.kt`  
**Lines:** 373–381

### Relevant Code
```kotlin
// Finally assign the remaining civs randomly
for (civ in randomCivs) {
    val rng = civ.state.stateBasedRandom("MapRegions.assignRegions")
    // throws if regions.size < civilizations.size or if the assigning mismatched - leads to popup on newgame screen
    val startRegion = unpickedRegions.random(rng)
    logAssignRegion(true, BiasTypes.Random, civ, startRegion)
    assignCivToRegion(civ, startRegion)
    unpickedRegions.remove(startRegion)
}
```

**Status:** ✓ Uses `civ.state.stateBasedRandom("MapRegions.assignRegions")` for deterministic selection. Exception thrown if `regions.size < civilizations.size` (line 376 comment).

---

## 5. Civilization – Key Fields & Methods

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/logic/civilization/Civilization.kt`

### 5a. `knows()` Methods (lines 384–385)
```kotlin
@Readonly fun knows(otherCivName: String) = diplomacy.containsKey(otherCivName)
@Readonly fun knows(otherCiv: Civilization) = knows(otherCiv.civID)
```

### 5b. `getStatForRanking()` (lines 836–850)
```kotlin
@Readonly
fun getStatForRanking(category: RankingType): Int {
    return if (isDefeated()) 0
    else when (category) {
            RankingType.Score -> calculateTotalScore().toInt()
            RankingType.Population -> cities.sumOf { it.population.population }
            RankingType.Growth -> stats.statsForNextTurn.food.roundToInt()
            RankingType.Production -> stats.statsForNextTurn.production.roundToInt()
            RankingType.Gold -> gold
            RankingType.Territory -> cities.sumOf { it.tiles.size }
            RankingType.Force -> getMilitaryMight()
            RankingType.Happiness -> getHappiness()
            RankingType.Technologies -> tech.researchedTechnologies.size
            RankingType.Culture -> policies.adoptedPolicies.count { !Policy.isBranchCompleteByName(it) }
    }
}
```

**Note:** Line 847 shows `tech.researchedTechnologies.size` (not `.techsResearched.size`).

### 5c. `calculateTotalScore()` (line 912)
```kotlin
@Readonly fun calculateTotalScore() = calculateScoreBreakdown().values.sum()
```

### 5d. `shouldHideCivCount()` (lines 1246–1253)
```kotlin
@Readonly
fun shouldHideCivCount(): Boolean {
    if (!gameInfo.gameParameters.randomNumberOfPlayers) return false
    val knownCivs = 1 + getKnownCivs().count { it.isMajorCiv() }
    if (knownCivs >= gameInfo.gameParameters.maxNumberOfPlayers) return false
    if (hasUnique(UniqueType.OneTimeRevealEntireMap)) return false
    // Other ideas? viewableTiles.size == gameInfo.tileMap.tileList.size seems not quite useful...
    return true
}
```

### 5e. `hasExplored()` (line 201)
```kotlin
@Readonly fun hasExplored(tile: Tile) = tile.isExplored(this)
```

### 5f. `gold` Field (line 131)
```kotlin
var gold = 0
    private set
```

### 5g. Era Access via TechManager
```kotlin
// Lines 148–149
var tech = TechManager()
// TechManager.kt lines 29–30:
@Transient
var era: Era = Era()
```

---

## 6. TechManager – Key Members

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/logic/civilization/managers/TechManager.kt`

### Key Transient & Serialized Members
```kotlin
28  class TechManager : IsPartOfGameInfoSerialization {
29      @Transient
30      var era: Era = Era()
31
32      @Transient
33      lateinit var civInfo: Civilization
34      
35      @Transient
36      var researchedTechnologies = ArrayList<Technology>()
37      
38      @Transient
39      internal var techUniques = UniqueMap()
    
    // Serialized fields:
65      var techsResearched = HashSet<String>()
66      var techsInProgress = HashMap<String, Int>()
71      private var overflowScience = 0
```

### Era-Entered Logic
**Note:** Era transitions likely handled elsewhere; TechManager.era is a transient Civ-state marker, not a progression tracker.

### canBeResearched() (line 176)
```kotlin
fun canBeResearched(techName: String): Boolean {
```

---

## 7. TradeLogic – Resource & Gold APIs

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/logic/trade/TradeLogic.kt`

### 7a. `getPerTurnResourcesWithOriginsForTrade()` (line 60)
```kotlin
for (entry in civInfo.getPerTurnResourcesWithOriginsForTrade()
    .filterNot { it.resource.resourceType == ResourceType.Bonus }
    .filter { it.origin == Constants.tradable }
)
```

**Note:** Method call is on `civInfo` (Civilization), not TradeLogic. Signature not visible in this file.

### 7b. `getStockpiledResourcesForTrade()` (line 69)
```kotlin
for (entry in civInfo.getStockpiledResourcesForTrade()){
    offers.add(TradeOffer(entry.resource.name, TradeOfferType.Stockpiled_Resource, entry.amount, speed = civInfo.gameInfo.speed))
}
```

### 7c. Gold & Gold-Per-Turn (lines 73–74)
```kotlin
offers.add(TradeOffer(Constants.flatGold, TradeOfferType.Gold, civInfo.gold, speed = civInfo.gameInfo.speed))
offers.add(TradeOffer(Constants.goldPerTurn, TradeOfferType.Gold_Per_Turn, civInfo.stats.statsForNextTurn.gold.toInt(), civInfo.gameInfo.speed))
```

**Treasury Gold:** `civInfo.gold` (field).  
**Gold Per Turn:** `civInfo.stats.statsForNextTurn.gold` (from CivInfoStatsForNextTurn).

---

## 8. EspionageManager – Spy Presence & Techs to Steal

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/logic/civilization/managers/EspionageManager.kt`

### 8a. Spy Presence in City (line 82)
```kotlin
@Readonly fun getSpiesInCity(city: City): List<Spy> = spyList.filter { it.getCityOrNull() == city }
```

### 8b. getTechsToSteal() (lines 72–80)
```kotlin
@Readonly
fun getTechsToSteal(otherCiv: Civilization): Set<String> {
    val techsToSteal = mutableSetOf<String>()
    for (tech in otherCiv.tech.techsResearched) {
        if (civInfo.tech.isResearched(tech)) continue
        if (!civInfo.tech.canBeResearched(tech)) continue
        techsToSteal.add(tech)
    }
    return techsToSteal
}
```

**Status:** ✓ Accesses `otherCiv.tech.techsResearched` (HashSet<String>) and checks `canBeResearched()`.

---

## 9. Tile – zeroBasedIndex & Spatial Key

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/logic/map/tile/Tile.kt`

### zeroBasedIndex Definition (line 101)
```kotlin
var zeroBasedIndex: Int = 0
```

### TileMap – Tile Ordering

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/logic/map/TileMap.kt`

- **Primary storage:** `tileList: ArrayList<Tile>` (line 39)
- **Performance index:** `tileMatrix: ArrayList<ArrayList<Tile?>>` (line 93)
- Each Tile's `zeroBasedIndex` corresponds to its position in `tileList` (1D flattened index).

---

## 10. Vision & Fog of War APIs

**File(s):** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/logic/civilization/Civilization.kt`

### 10a. "Is Tile Visible to Civ X"
```kotlin
@Transient
var viewableTiles = setOf<Tile>()  // Line 91
```
**Method:** Check `civ.viewableTiles.contains(tile)` or use `tile in civ.viewableTiles`.

### 10b. "Has X Explored Tile T"
```kotlin
@Readonly fun hasExplored(tile: Tile) = tile.isExplored(this)  // Line 201
```
**Implementation:** Delegates to `tile.isExplored(civ)` (method on Tile, checks `exploredBy` HashSet).

**Note:** No `VisibleTilesManager` class found; visibility managed directly via `viewableTiles` transient set.

---

## 11. Ruleset – Collections & Map Types

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/models/ruleset/Ruleset.kt`

### Collection Fields (lines 126–155)
All use **LinkedHashMap** for deterministic iteration order:

```kotlin
val beliefs = LinkedHashMap<String, Belief>()
val buildings = LinkedHashMap<String, Building>()
val difficulties = LinkedHashMap<String, Difficulty>()
val eras = LinkedHashMap<String, Era>()
val speeds = LinkedHashMap<String, Speed>()
val nations = LinkedHashMap<String, Nation>()
val policies = LinkedHashMap<String, Policy>()
val policyBranches = LinkedHashMap<String, PolicyBranch>()
val specialists = LinkedHashMap<String, Specialist>()
val technologies = LinkedHashMap<String, Technology>()
val terrains = LinkedHashMap<String, Terrain>()
val tileImprovements = LinkedHashMap<String, TileImprovement>()
val tileResources = LinkedHashMap<String, TileResource>()
val tutorials = LinkedHashMap<String, Tutorial>()
val units = LinkedHashMap<String, BaseUnit>()
val unitPromotions = LinkedHashMap<String, Promotion>()
val unitNameGroups = LinkedHashMap<String, UnitNameGroup>()
val unitTypes = LinkedHashMap<String, UnitType>()
val victories = LinkedHashMap<String, Victory>()
val cityStateTypes = LinkedHashMap<String, CityStateType>()
val personalities = LinkedHashMap<String, Personality>()
val events = LinkedHashMap<String, Event>()
```

**Special Cases:**
- `religions: ArrayList<String>()` (line 137) – list not map
- `globalUniques: GlobalUniques()` (line 133)

**Status:** ✓ All use `LinkedHashMap` for deterministic ordering.

---

## 12. GameParameters & MapParameters – Scenario Setup

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/models/metadata/GameParameters.kt`

### Key Fields for Scenario Generation
```kotlin
var players = ArrayList<Player>().apply {
    add(Player(playerType = PlayerType.Human))
    repeat(3) { add(Player()) }
}

var randomNumberOfPlayers = false
var minNumberOfPlayers = 3
var maxNumberOfPlayers = 3

var numberOfCityStates = 6
var baseRuleset: String = BaseRuleset.Civ_V_GnK.fullName  // Line 53
```

### BaseRuleset Enum

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/models/metadata/BaseRuleset.kt`

```kotlin
@Suppress("EnumEntryName")
enum class BaseRuleset(val fullName: String) {
    Civ_V_Vanilla("Civ V - Vanilla"),
    Civ_V_GnK("Civ V - Gods & Kings"),
}
```

**Status:** ✓ `BaseRuleset.Civ_V_GnK.fullName` = `"Civ V - Gods & Kings"`

### MapParameters

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/logic/map/MapParameters.kt`

```kotlin
class MapParameters : IsPartOfGameInfoSerialization {
    var name = ""
    var type = MapType.pangaea
    var shape = MapShape.hexagonal
    var mapSize = MapSize.Medium
    var mapResources = MapResourceSetting.default.label
    var seed = 0  // (visible in line 103: gameSetupInfo.mapParameters.seed = 0)
```

---

## 13. App Version / BuildConfig

**Files:**
- `/Users/j/Unciv-self-play-data-plane/buildSrc/src/main/kotlin/BuildConfig.kt` (generation)
- `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/logic/Versioning.kt` (wrapper)

### Version Class (lines 16–25)
```kotlin
data class Version(
    val text: String,
    val number: Int
) : IsPartOfGameInfoSerialization {
    @Suppress("unused") // used by json serialization
    internal constructor() : this("", -1)

    @Pure fun toNiceString() = "[$text] (Build [$number])".tr()
    @Pure fun toSerializeString() = "$text (Build $number)"
}
```

### Runtime Access
```kotlin
// UncivGame.VERSION (instance created via workflow editing source)
UncivGame.VERSION.text     // e.g., "4.13.18"
UncivGame.VERSION.number   // e.g., 13183
```

**Status:** ✓ Version accessible via `UncivGame.Companion.VERSION`.

---

## 14. Victory Progress

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/logic/civilization/managers/VictoryManager.kt`

### Spaceship Parts (line 20)
```kotlin
var currentsSpaceshipParts = Counter<String>()  // Note: typo "currents" but saved in save files
```

### Diplomatic Victory (line 21)
```kotlin
var hasEverWonDiplomaticVote = false
```

**File:** `/Users/j/Unciv-self-play-data-plane/core/src/com/unciv/models/ruleset/Victory.kt`

### Required Spaceship Parts (lines 62–69)
```kotlin
val requiredSpaceshipParts = ArrayList<String>()

val requiredSpaceshipPartsAsCounter by lazy {
    val parts = Counter<String>()
    for (spaceshipPart in requiredSpaceshipParts)
        parts.add(spaceshipPart, 1)
    parts
}
```

### Milestones & Policy Branches (lines 60–61)
```kotlin
val milestones = ArrayList<String>()
val milestoneObjects by lazy { milestones.map { Milestone(it, this) }}
```

Milestones include:
- `AddedSSPartsInCapital("Add all [comment] in capital")`
- `CompletePolicyBranches("Complete [amount] Policy branches")`
- `WorldReligion("Become the world religion")`
- etc.

---

## 15. Existing Tests Layout

**Directory:** `/Users/j/Unciv-self-play-data-plane/tests/src/com/unciv/`

**Framework:** JUnit + GdxTestRunner (custom libGDX runner)

### Example Test File
`/Users/j/Unciv-self-play-data-plane/tests/src/com/unciv/ui/screens/victoryscreen/RankingTypeTests.kt`

```kotlin
import com.unciv.testing.GdxTestRunner
@RunWith(GdxTestRunner::class)
class RankingTypeTests { ... }
```

**Module:** `/Users/j/Unciv-self-play-data-plane/tests/` (separate Gradle module, see `tests/build.gradle.kts`)

**Status:** ⚠️ No existing `TestGame` helper found in quick scan. May need to create one.

---

## 16. Candidate-Enumeration APIs

### Technologies – `canBeResearched()`
```kotlin
fun canBeResearched(techName: String): Boolean  // TechManager.kt:176
```

### Policies – `isAdoptable()`
```kotlin
fun isAdoptable(policy: Policy, checkEra: Boolean = true): Boolean  // PolicyManager.kt:221
```

### Promotions – `getAvailablePromotions()`
```kotlin
fun getAvailablePromotions(): Sequence<Promotion>  // UnitPromotions.kt:182
```

### Great Persons & Diplomatic Votes
**Note:** Not immediately visible in quick scan; likely in Victory-related managers or CityState functions.

---

## Utility & Infrastructure

### Serialization
- **Framework:** libGDX `Json` (com.badlogic.gdx.utils.Json)
- **Alternate:** kotlinx.serialization (for API v2 multiplayer, not game saves)
- **Save Format:** JSON via Gdx.Json

### RulesetCache – Loading GnK
```kotlin
// RulesetCache.kt lines 112, 140–141
fun getComplexRuleset(parameters: GameParameters) =
    getComplexRuleset(parameters.mods, parameters.baseRuleset)

// To get GnK ruleset:
val gnkRuleset = RulesetCache.getComplexRuleset(
    mods = linkedSetOf(),  // no mods
    optionalBaseRuleset = "Civ V - Gods & Kings"
)
// or use BaseRuleset.Civ_V_GnK.fullName
```

### Package Structure
**Core package:** `com.unciv.logic.*` and `com.unciv.models.ruleset.*`  
**New packages should live:** 
- `com.unciv.logic.simulation.*` (already exists)
- New: `com.unciv.logic.simulation.dataplane.*` or `com.unciv.dataplane.*`

### Desktop Module
**Path:** `/Users/j/Unciv-self-play-data-plane/desktop/src/com/unciv/app/desktop/`  
**Note:** `SimBenchmark.kt` exists in MAIN worktree (per spec) but not found in this tree.

---

## Summary of Spec Drifts

| Seam | Spec Line | Actual Line | Status |
|------|-----------|-------------|--------|
| NextTurnAutomation.automateCivMoves | ~38 | **38–40** | ✓ Correct |
| Simulation init | ~97 | **55–69** | ✓ Found (different line) |
| Simulation per-turn loop | ~134 | **108–114** | ✓ Found |
| Simulation finalize | ~141 | **118–130** | ✓ Found |
| CoroutineName usage | one per worker | **line 99** | ✓ Confirmed |
| GameStarter shuffle | ~322 | **322** | ⚠️ **Uses wrong RNG** (see section 3) |
| MapRegions "too many" | ~376 | **373–381** | ✓ Found (comment at 376) |
| Civilization.knows() | ~384–385 | **384–385** | ✓ Exact match |
| Civ.getStatForRanking | ~840–850 | **836–850** | ✓ Found |
| tech.researchedTechnologies | ~847 | **847** | ✓ Confirmed `.size` |
| Civ.calculateTotalScore() | not stated | **912** | ✓ Found |
| Civ.shouldHideCivCount() | ~1246–1253 | **1246–1253** | ✓ Exact match |
| Civ.hasExplored() | not stated | **201** | ✓ Found |
| Civ.gold | not stated | **131** | ✓ Field confirmed |
| TechManager.era | not stated | **30** | ✓ Transient confirmed |
| TechManager.researchedTechnologies | not stated | **36** | ✓ Transient confirmed |
| TechManager.techsInProgress | not stated | **71** | ✓ Serialized confirmed |
| TradeLogic getResources | ~60–74 | **60–74** | ✓ Methods on Civilization, not TradeLogic |
| EspionageManager getTechsToSteal | ~64–80 | **72–80** | ✓ Found (line diff) |
| Tile.zeroBasedIndex | not stated | **101** | ✓ Found |
| Ruleset collections | not stated | **126–155** | ✓ All LinkedHashMap confirmed |
| GameParameters baseRuleset | not stated | **53** | ✓ Confirmed |
| BaseRuleset enum | not stated | **BaseRuleset.kt** | ✓ Confirmed Civ V - Gods & Kings |
| VictoryManager SS parts | not stated | **20** | ✓ Found (typo: "currents") |

---

## Recommendations

1. **Fix GameStarter.shuffle()** – Use seeded RNG instead of Kotlin's default shuffle.
2. **Policy injection for NextTurnAutomation** – Needs to be added at caller level (not visible in this function's signature yet).
3. **Create TestGame helper** – No existing test utility found for headless game setup.
4. **Document vision APIs** – `viewableTiles` and `tile.isExplored()` are the canonical APIs (no VisibleTilesManager).
5. **Candidate enumeration** – `canBeResearched`, `isAdoptable`, `getAvailablePromotions` are the main entry points.

