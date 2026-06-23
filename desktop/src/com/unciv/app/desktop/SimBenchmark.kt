package com.unciv.app.desktop

import com.unciv.Constants
import com.unciv.UncivGame
import com.unciv.logic.GameInfo
import com.unciv.logic.GameStarter
import com.unciv.logic.civilization.PlayerType
import com.unciv.logic.files.UncivFiles
import com.unciv.logic.map.MapParameters
import com.unciv.logic.map.MapSize
import com.unciv.logic.simulation.Simulation
import com.unciv.models.metadata.BaseRuleset
import com.unciv.models.metadata.GameParameters
import com.unciv.models.metadata.GameSettings
import com.unciv.models.metadata.GameSetupInfo
import com.unciv.models.metadata.Player
import com.unciv.models.ruleset.RulesetCache
import com.unciv.models.ruleset.Speed
import com.unciv.models.ruleset.nation.Nation
import com.unciv.models.skins.SkinCache
import com.unciv.models.tilesets.TileSetCache
import com.unciv.utils.Log
import kotlin.time.ExperimentalTime

/**
 * Throughput / "cost to produce training data" benchmark — REALISTIC config.
 *
 * Medium map, 6 major civs, 6 city-states, barbarians on (≈ a default Unciv game).
 * Measures, on THIS machine:
 *  - raw headless turns/s, single- and multi-thread, with A* pathfinding OFF vs ON (same seed)
 *  - the marginal cost of dumping a full game-state JSON snapshot (a would-be ML sample)
 *
 * Run via:  ./gradlew :desktop:simBench      All summary lines prefixed "BENCH|".
 */
@OptIn(ExperimentalTime::class)
internal object SimBenchmark {

    private const val NUM_MAJOR_CIVS = 6
    private const val NUM_CITY_STATES = 6
    private const val BARBARIANS = true
    private const val MAX_TURNS = 500
    private const val MT_CAP_TURNS = 300   // cap for multi-thread batch so it stays bounded

    private class BatchResult(val games: Int, val turns: Int, val seconds: Double) {
        val turnsPerSec get() = turns / seconds
        val gamesPerSec get() = games / seconds
        val avgTurnsPerGame get() = if (games > 0) turns.toDouble() / games else 0.0
    }

    @JvmStatic
    fun main(args: Array<String>) {
        Log.backend = DesktopLogBackend()

        val game = UncivGame(true)
        UncivGame.Current = game
        UncivGame.Current.settings = GameSettings().apply {
            showTutorials = false
            turnsBetweenAutosaves = 10000
        }
        RulesetCache.loadRulesets(true)
        TileSetCache.loadTileSetConfigs(true)
        SkinCache.loadSkinConfigs(true)

        val newGame = buildBaseGame()
        UncivGame.Current.gameInfo = newGame

        val cores = Runtime.getRuntime().availableProcessors()
        val threads = minOf(cores, 8)
        println("BENCH| cores=$cores benchThreads=$threads map=Medium majorCivs=$NUM_MAJOR_CIVS " +
            "cityStates=$NUM_CITY_STATES barbarians=$BARBARIANS maxTurns=$MAX_TURNS ruleset=Civ_V_GnK")

        // ---- warmup (JIT) ----
        println("BENCH| warming up JIT ...")
        setAStar(false); runBoundedTurns(newGame, 30)
        setAStar(true);  runBoundedTurns(newGame, 30)

        // ---- PART A1: single-thread turns/s, A* OFF vs ON, SAME seed (apples to apples) ----
        setAStar(false)
        val s1Off = runBoundedTurns(newGame, 70)
        setAStar(true)
        val s1On = runBoundedTurns(newGame, 70)
        println("BENCH| A1 single-thread (same seed, 70 turns):")
        println("BENCH|    A* OFF: ${"%.1f".format(s1Off.turnsPerSec)} turns/s (${"%.2f".format(s1Off.seconds)}s)")
        println("BENCH|    A* ON : ${"%.1f".format(s1On.turnsPerSec)} turns/s (${"%.2f".format(s1On.seconds)}s)  " +
            "=> ${pctFaster(s1Off.turnsPerSec, s1On.turnsPerSec)}")

        // ---- PART A2: multi-thread aggregate throughput, A* OFF vs ON ----
        println("BENCH| A2 running $threads threads x 1 game (cap $MT_CAP_TURNS turns) ...")
        setAStar(false)
        val mOff = runFullGames(newGame, gamesPerThread = 1, threads = threads, maxTurns = MT_CAP_TURNS)
        setAStar(true)
        val mOn = runFullGames(newGame, gamesPerThread = 1, threads = threads, maxTurns = MT_CAP_TURNS)
        println("BENCH| A2 multi-thread x$threads aggregate:")
        println("BENCH|    A* OFF: ${"%.1f".format(mOff.turnsPerSec)} turns/s, ${"%.2f".format(mOff.gamesPerSec)} games/s, " +
            "avg ${"%.0f".format(mOff.avgTurnsPerGame)} turns/game (${mOff.games} games, ${"%.1f".format(mOff.seconds)}s)")
        println("BENCH|    A* ON : ${"%.1f".format(mOn.turnsPerSec)} turns/s, ${"%.2f".format(mOn.gamesPerSec)} games/s, " +
            "avg ${"%.0f".format(mOn.avgTurnsPerGame)} turns/game (${mOn.games} games, ${"%.1f".format(mOn.seconds)}s)  " +
            "=> ${pctFaster(mOff.turnsPerSec, mOn.turnsPerSec)}")

        // ---- PART B: serialization cost of one full-state training sample (realistic mid-game) ----
        println("BENCH| B building a representative mid-game state ...")
        setAStar(false)
        var serMsPlain = Double.NaN; var serMsGz = Double.NaN
        var plainBytes = 0; var gzBytes = 0; var sampleTurn = -1; var unitCount = -1; var cityCount = -1
        try {
            val sample = GameStarter.startNewGame(GameSetupInfo(newGame).apply { mapParameters.seed = 0 })
            sample.simulateUntilWin = true
            sample.simulateMaxTurns = 120
            sample.nextTurn()
            sampleTurn = sample.turns
            unitCount = sample.civilizations.sumOf { it.units.getCivUnits().count() }
            cityCount = sample.civilizations.sumOf { it.cities.size }
            val iters = 40
            repeat(5) { UncivFiles.gameInfoToString(sample, forceZip = false, updateChecksum = false) }
            var t0 = System.nanoTime()
            repeat(iters) { plainBytes = UncivFiles.gameInfoToString(sample, forceZip = false, updateChecksum = false).length }
            serMsPlain = (System.nanoTime() - t0) / 1e6 / iters
            repeat(3) { UncivFiles.gameInfoToString(sample, forceZip = true, updateChecksum = false) }
            t0 = System.nanoTime()
            repeat(iters) { gzBytes = UncivFiles.gameInfoToString(sample, forceZip = true, updateChecksum = false).length }
            serMsGz = (System.nanoTime() - t0) / 1e6 / iters
        } catch (ex: Exception) {
            println("BENCH| B serialization measurement failed: ${ex.message}")
        }

        // ---- SUMMARY + derived cost model (use the faster of A*off/on for multi headline) ----
        val bestMulti = if (mOn.turnsPerSec >= mOff.turnsPerSec) mOn else mOff
        val bestTag = if (mOn.turnsPerSec >= mOff.turnsPerSec) "A* ON" else "A* OFF"
        println("\nBENCH|========== SUMMARY (Medium, ${NUM_MAJOR_CIVS}civ+${NUM_CITY_STATES}CS, barbs=$BARBARIANS) ==========")
        println("BENCH| single-thread:   ${"%.1f".format(s1Off.turnsPerSec)} (A*off) / ${"%.1f".format(s1On.turnsPerSec)} (A*on) turns/s")
        println("BENCH| multi-thread x$threads: ${"%.1f".format(mOff.turnsPerSec)} (A*off) / ${"%.1f".format(mOn.turnsPerSec)} (A*on) turns/s")
        println("BENCH| best multi:      ${"%.1f".format(bestMulti.turnsPerSec)} turns/s ($bestTag), avg ${"%.0f".format(bestMulti.avgTurnsPerGame)} turns/game")
        val gameTurnsPerHour = bestMulti.turnsPerSec * 3600
        val decisionsPerHour = gameTurnsPerHour * NUM_MAJOR_CIVS   // 1 strategic decision per major civ per game-turn
        println("BENCH| major-civ decisions/hour ~ ${"%,.0f".format(decisionsPerHour)} (~${"%,.0f".format(decisionsPerHour * 24)}/day)")
        if (!serMsPlain.isNaN()) {
            println("BENCH| state sample @turn $sampleTurn ($unitCount units, $cityCount cities): " +
                "serialize ${"%.2f".format(serMsPlain)} ms, ${"%.0f".format(plainBytes / 1024.0)} KB plain / ${"%.1f".format(gzBytes / 1024.0)} KB gzip")
            val gbHrPlain = decisionsPerHour * plainBytes / 1e9
            val gbHrGz = decisionsPerHour * gzBytes / 1e9
            println("BENCH| full-state dump every decision => ${"%.0f".format(gbHrPlain)} GB/hr plain / ${"%.0f".format(gbHrGz)} GB/hr gzip")
        }
        println("BENCH|=============================")

        System.out.flush()
        System.exit(0)
    }

    private fun setAStar(on: Boolean) { UncivGame.Current.settings.useAStarPathfinding = on }

    private fun pctFaster(base: Double, other: Double): String {
        val d = (other - base) / base * 100
        return if (d >= 0) "A* ON +${"%.0f".format(d)}% faster" else "A* ON ${"%.0f".format(d)}% (slower)"
    }

    /** Medium map, 6 generic major civs + 6 city-states + barbarians (≈ a default game). */
    private fun buildBaseGame(): GameInfo {
        val ruleset = RulesetCache[BaseRuleset.Civ_V_GnK.fullName]!!
        val majors = (1..NUM_MAJOR_CIVS).map { i ->
            Nation().apply { name = "BenchCiv$i" }.also { ruleset.nations[it.name] = it }
        }
        val gameParameters = GameParameters().apply {
            difficulty = "King"
            numberOfCityStates = NUM_CITY_STATES
            speed = Speed.DEFAULT
            noBarbarians = !BARBARIANS
            players = ArrayList<Player>().apply {
                for (n in majors) add(Player(n))
                add(Player(Constants.spectator, PlayerType.Human))
            }
        }
        gameParameters.players.last().setNationTransient(ruleset)
        val mapParameters = MapParameters().apply {
            mapSize = MapSize.Medium
            noRuins = true
            noNaturalWonders = true
        }
        val newGame = GameStarter.startNewGame(GameSetupInfo(gameParameters, mapParameters))
        newGame.gameParameters.victoryTypes = ArrayList(newGame.ruleset.victories.keys)
        return newGame
    }

    /** Simulate a single fresh game (seed 0) forward [turnsToSimulate] turns; single-thread timing. */
    private fun runBoundedTurns(base: GameInfo, turnsToSimulate: Int): BatchResult {
        val g = GameStarter.startNewGame(GameSetupInfo(base).apply { mapParameters.seed = 0 })
        g.simulateUntilWin = true
        val startTurn = g.turns
        val t0 = System.nanoTime()
        g.simulateMaxTurns = startTurn + turnsToSimulate
        g.nextTurn()
        val secs = (System.nanoTime() - t0) / 1e9
        return BatchResult(games = 1, turns = g.turns - startTurn, seconds = secs)
    }

    /** Multithreaded batch of games (to victory or maxTurns); aggregate timing. */
    private fun runFullGames(base: GameInfo, gamesPerThread: Int, threads: Int, maxTurns: Int): BatchResult {
        val sim = Simulation(base, gamesPerThread, threads, maxTurns)
        val t0 = System.nanoTime()
        sim.start()
        val secs = (System.nanoTime() - t0) / 1e9
        val totalTurns = sim.steps.sumOf { it.turns }
        return BatchResult(games = sim.steps.size, turns = totalTurns, seconds = secs)
    }
}
