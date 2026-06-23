package com.unciv.logic.simulation

import com.unciv.Constants
import com.unciv.UncivGame
import com.unciv.logic.GameInfo
import com.unciv.logic.GameStarter
import com.unciv.logic.automation.Timers
import com.unciv.logic.simulation.dataplane.DataPlaneHooks
import com.unciv.models.metadata.GameSetupInfo
import kotlinx.coroutines.CoroutineName
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlin.math.max
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.ExperimentalTime

@ExperimentalTime
class Simulation(
    private val newGameInfo: GameInfo,
    val simulationsPerThread: Int = 1,
    private val threadsNumber: Int = 1,
    private val maxTurns: Int = 500,
    private val statTurns: List<Int> = listOf(),
    /** Opt-in self-play data plane. Null ⇒ existing behavior (no featurization, no shards). */
    private val dataPlane: com.unciv.logic.simulation.dataplane.DataPlaneContext? = null,
    /** Self-play win rule (D7): when a game reaches the turn cap with no formal victory, award the
     *  win to the highest-score alive major (draw on a tie). Off ⇒ legacy "Draw at cap" behavior,
     *  so ConsoleLauncher/SimBenchmark are unchanged. Set by the self-play GENERATE + EVAL runner. */
    private val scoreLeaderOnTimeout: Boolean = false,
    /** Base seed offset for the self-play data-plane path so each loop ROUND generates different
     *  games (and EVAL reproduces with a fixed base). 0 ⇒ legacy per-(thread,iteration) seeds. */
    private val seedBase: Long = 0,
) {
    private val maxSimulations = threadsNumber * simulationsPerThread
    private val majorCivs = newGameInfo.civilizations.filter { !it.isSpectator() && it.isMajorCiv() }.map { it.civID }
    private val numMajorCivs = newGameInfo.civilizations.filter { !it.isSpectator() && it.isMajorCiv()  }.size
    private var startTime: Long = 0
    var steps = ArrayList<SimulationStep>()
    var numWins = mutableMapOf<String, MutableInt>()
    private var summaryStatsPop = HashMap<String, HashMap<Int, HashMap<Stat, MutableInt>>>() // [civ][turn][stat]=value
    private var summaryStatsProd = HashMap<String, HashMap<Int, HashMap<Stat, MutableInt>>>() // [civ][turn][stat]=value
    private var summaryStatsCities = HashMap<String, HashMap<Int, HashMap<Stat, MutableInt>>>() // [civ][turn][stat]=value
    private var summaryStatsAvgPop = HashMap<String, HashMap<Int, HashMap<Stat, MutableInt>>>() // [civ][turn][stat]=value
    private var winRateByVictory = HashMap<String, MutableMap<String, MutableInt>>()
    private var winTurnByVictory = HashMap<String, MutableMap<String, MutableInt>>()
    private var avgSpeed = 0f
    private var avgDuration: Duration = Duration.ZERO
    private var totalTurns = 0
    private var totalDuration: Duration = Duration.ZERO
    private var stepCounter: Int = 0
    enum class Stat {
        SUM,
        NUM
    }
    // print flags
    private val printPop = true
    private val printProd = false
    private val printCityCnt = false
    private val printAvgCityPop = false

    init{
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
    
    // Need to initialize the values
    // Later will iterate with flatMap
    private fun initHash(summary: HashMap<String, HashMap<Int, HashMap<Stat, MutableInt>>>) {
        for (civ in majorCivs) {
            for (turn in statTurns) {
                summary.getOrPut(civ) { hashMapOf() }.getOrPut(turn){hashMapOf()}[Stat.SUM] = MutableInt(0)
                summary.getOrPut(civ) { hashMapOf() }.getOrPut(turn){hashMapOf()}[Stat.NUM] = MutableInt(0)
            }
            val turn = -1 // end of game
            summary.getOrPut(civ) { hashMapOf() }.getOrPut(turn){hashMapOf()}[Stat.SUM] = MutableInt(0)
            summary.getOrPut(civ) { hashMapOf() }.getOrPut(turn){hashMapOf()}[Stat.NUM] = MutableInt(0)
        }
    }

    fun start() = runBlocking {
        startTime = System.currentTimeMillis()

        println(
            "Starting new game with major civs: " +
                newGameInfo.civilizations.filter { it.isMajorCiv() }.joinToString { it.civID } +
                " and minor civs: " +
                newGameInfo.civilizations.filter { it.isCityState }.joinToString { it.civID }
        )

        newGameInfo.gameParameters.shufflePlayerOrder = true

        // Data-plane hook (init): install the per-civ recording hook for this run (no-op otherwise).
        dataPlane?.let { DataPlaneHooks.install(it.policy) }

        Timers.singleton.startTiming()
        val jobs = (1..threadsNumber).map { threadId ->
            launch(Dispatchers.Default + CoroutineName("simulation-$threadId")) {
                val workerName = coroutineContext[CoroutineName]?.name ?: "simulation-$threadId"
                repeat(simulationsPerThread) { iteration ->
                    val step = SimulationStep(newGameInfo, statTurns)
                    val gameSetupInfo = GameSetupInfo(newGameInfo)
                    // Determinism: the data-plane path uses a non-zero seed + seeded shuffle so a
                    // recorded scenario replays byte-identically. Default benchmark path keeps seed=0.
                    if (dataPlane != null) {
                        gameSetupInfo.mapParameters.seed = seedBase + (threadId.toLong() shl 20) + iteration + 1
                        gameSetupInfo.gameParameters.deterministicShuffle = true
                    } else {
                        gameSetupInfo.mapParameters.seed = 0
                    }
                    val gameInfo = GameStarter.startNewGame(gameSetupInfo)

                    // Data-plane hook (register): the installed policy CONTROLS this game's deciding
                    // civs (tech+policy); shards are EMITTED only when the emitter is enabled
                    // (GENERATE). EVAL registers for control with emit=false (no shards).
                    if (dataPlane != null) {
                        gameInfo.gameId = "$workerName-$iteration"
                        DataPlaneHooks.registerGame(gameInfo, dataPlane.vocab, dataPlane.config,
                            dataPlane.fingerprint, gameInfo.gameId, gameSetupInfo.mapParameters.seed,
                            "shard-$workerName-$iteration", emit = dataPlane.config.enabled)
                    }

                    gameInfo.simulateUntilWin = true

                    for (turn in statTurns) {
                        gameInfo.simulateMaxTurns = turn
                        gameInfo.nextTurn()
                        step.update(gameInfo)
                        if (step.victoryType != null) break
                        step.saveTurnStats(gameInfo)
                    }

                    step.update(gameInfo)

                    if (step.victoryType == null) {
                        gameInfo.simulateMaxTurns = maxTurns
                        gameInfo.nextTurn()
                        step.update(gameInfo)
                    }

                    if (step.victoryType != null) {
                        step.saveTurnStats(gameInfo)
                        step.winner = step.currentPlayer
                        println("${step.winner} won ${step.victoryType} victory on turn ${step.turns}")
                    } else {
                        // Self-play win rule (D7): decide a turn-cap game by total score (draw on a tie).
                        if (scoreLeaderOnTimeout)
                            step.winner = com.unciv.logic.simulation.SimStats.scoreLeader(gameInfo)?.civID
                        println("Max simulation ${step.turns} turns reached: ${step.winner?.let { "$it leads on score" } ?: "Draw"}")
                    }

                    // Data-plane hook (finalize): emit the per-civ terminal reward, publish the shard,
                    // and unregister (no-op for EVAL's control-only games).
                    if (dataPlane != null) DataPlaneHooks.finalizeGame(gameInfo, step.winner)

                    // ⚠️ these need to be thread-safe
                    updateCounter(threadId)
                    add(step)
                    print()
                }
            }
        }

        jobs.joinAll()
        // Data-plane hook (teardown): clear the recording hook + registry (restores normal behavior).
        dataPlane?.let { DataPlaneHooks.uninstall() }
        Timers.singleton.endTiming()
    }

    @Suppress("UNUSED_PARAMETER")   // used when activating debug output
    @Synchronized fun add(step: SimulationStep, threadId: Int = 1) {
        steps.add(step)
    }

    @Suppress("UNUSED_PARAMETER")   // used when activating debug output
    @Synchronized fun updateCounter(threadId: Int = 1) {
        stepCounter++
        println("Simulation step ($stepCounter/$maxSimulations)")
    }

    @Synchronized
    fun print(){
        getStats()
        println(text())
    }
    
    private fun summaryStatSet(summaryHash: HashMap<String, HashMap<Int, HashMap<Stat, MutableInt>>>,
                    civ: String, turn: Int, stat:  MutableMap<String, MutableMap<Int, MutableInt>>) {
        if (stat[civ]!![turn]!!.value != -1) {
            summaryHash[civ]!![turn]!![Stat.SUM]!!.add(stat[civ]!![turn]!!.value)
            summaryHash[civ]!![turn]!![Stat.NUM]!!.inc()
            //println("civ ${civ} @ ${turn} value ${stat[civ]!![turn]!!.value}")
        }
    }

    private fun getStats() {
        // win Rate
        numWins.values.forEach { it.value = 0 }
        winRateByVictory.flatMap { it.value.values }.forEach { it.value = 0 }
        winTurnByVictory.flatMap { it.value.values }.forEach { it.value = 0 }
        // reset to 0
        summaryStatsPop.flatMap { it.value.values }.forEach {
            it.values.forEach { it.value = 0 }
        }
        summaryStatsProd.flatMap { it.value.values }.forEach {
            it.values.forEach { it.value = 0 }
        }
        summaryStatsCities.flatMap { it.value.values }.forEach {
            it.values.forEach { it.value = 0 }
        }
        summaryStatsAvgPop.flatMap { it.value.values }.forEach {
            it.values.forEach { it.value = 0 }
        }
        steps.forEach {
            val winner = it.winner
            if (winner != null) {
                numWins[winner]?.inc()
                // victoryType is null for a score-leader-at-cap win (D7) — count it in numWins only.
                val vt = it.victoryType
                if (vt != null) {
                    winRateByVictory[winner]?.get(vt)?.inc()
                    winTurnByVictory[winner]?.get(vt)?.add(it.turns)
                }
            }
            for (civ in majorCivs) {
                for (turn in statTurns) {
                    summaryStatSet(summaryStatsPop, civ, turn, it.turnStatsPop)
                    summaryStatSet(summaryStatsProd, civ, turn, it.turnStatsProd)
                    summaryStatSet(summaryStatsCities, civ, turn, it.turnStatsCities)
                    if (it.turnStatsPop[civ]!![turn]!!.value != -1 && it.turnStatsCities[civ]!![turn]!!.value != -1) {
                        if (it.turnStatsCities[civ]!![turn]!!.value != 0) // if no cities, avgpop=0
                            summaryStatsAvgPop[civ]!![turn]!![Stat.SUM]!!.add(it.turnStatsPop[civ]!![turn]!!.value/it.turnStatsCities[civ]!![turn]!!.value)
                        summaryStatsAvgPop[civ]!![turn]!![Stat.NUM]!!.inc()
                    }
                }
                val turn = -1 // end of game
                summaryStatSet(summaryStatsPop, civ, turn, it.turnStatsPop)
                summaryStatSet(summaryStatsProd, civ, turn, it.turnStatsProd)
                summaryStatSet(summaryStatsCities, civ, turn, it.turnStatsCities)
                if (it.turnStatsCities[civ]!![turn]!!.value != 0) // if no cities, avgpop=0
                    summaryStatsAvgPop[civ]!![turn]!![Stat.SUM]!!.add(it.turnStatsPop[civ]!![turn]!!.value/it.turnStatsCities[civ]!![turn]!!.value)
                summaryStatsAvgPop[civ]!![turn]!![Stat.NUM]!!.inc()
            }
        }
        totalTurns = steps.sumOf { it.turns }
        totalDuration = (System.currentTimeMillis() - startTime).milliseconds
        avgSpeed = totalTurns.toFloat() / totalDuration.inWholeSeconds
        avgDuration = totalDuration / steps.size
    }
    
    // Helper text formatter
    private fun summaryStatsText(summaryStats: HashMap<Stat, MutableInt>,
                                 turn: Int, statStr: String): String {
        val turnStr = if(turn == -1) "END" else turn
        return "@$turnStr: $statStr avg=${summaryStats[Stat.SUM]!!.value.toFloat() / summaryStats[Stat.NUM]!!.value.toFloat()} cnt=${summaryStats[Stat.NUM]!!.value}\n"
    }

    fun text(): String {
        var outString = ""
        for (civ in majorCivs) {

            val numSteps = max(steps.size, 1)
            val expWinRate = 1f / numMajorCivs
            if (numWins[civ]!!.value == 0) continue
            val winRate = String.format("%.1f", numWins[civ]!!.value * 100f / numSteps)

            outString += "\n$civ:\n"
            outString += "$winRate% total win rate \n"
            if (numSteps * expWinRate >= 10 && numSteps * (1 - expWinRate) >= 10) {
                // large enough sample size, binomial distribution approximates the Normal Curve
                val pval = SimStats.binomialTest(
                    numWins[civ]!!.value.toDouble(),
                    numSteps.toDouble(),
                    expWinRate.toDouble(),
                    "greater"
                )
                outString += "one-tail binomial pval = $pval\n"
            }
            for (victory in UncivGame.Current.gameInfo!!.ruleset.victories.keys) {
                val winsVictory =
                    winRateByVictory[civ]!![victory]!!.value * 100 / max(numWins[civ]!!.value, 1)
                outString += "$victory: $winsVictory%    "
            }
            outString += "\n"
            for (victory in UncivGame.Current.gameInfo!!.ruleset.victories.keys) {
                val winsTurns =
                    winTurnByVictory[civ]!![victory]!!.value / max(winRateByVictory[civ]!![victory]!!.value, 1)
                outString += "$victory: $winsTurns    "
            }
            outString += "avg turns\n"
            for (turn in statTurns) {
                if(printPop)
                    outString += summaryStatsText(summaryStatsPop[civ]!![turn]!!, turn, "popSum")
                if(printProd)
                    outString += summaryStatsText(summaryStatsProd[civ]!![turn]!!, turn, "prodSum")
                if(printCityCnt)
                    outString += summaryStatsText(summaryStatsCities[civ]!![turn]!!, turn, "cityCount")
                if(printAvgCityPop)
                    outString += summaryStatsText(summaryStatsAvgPop[civ]!![turn]!!, turn, "avgCityPop")
            }
            val turn = -1 // end of match
            if(printPop)
                outString += summaryStatsText(summaryStatsPop[civ]!![turn]!!, turn, "popSum")
            if(printProd)
                outString += summaryStatsText(summaryStatsProd[civ]!![turn]!!, turn, "prodSum")
            if(printCityCnt)
                outString += summaryStatsText(summaryStatsCities[civ]!![turn]!!, turn, "cityCount")
            if(printAvgCityPop)
                outString += summaryStatsText(summaryStatsAvgPop[civ]!![turn]!!, turn, "avgCityPop")
        }
        outString += "\nAverage speed: %.1f turns/s \n".format(avgSpeed)
        outString += "Average game duration: $avgDuration\n"
        outString += "Total time: $totalDuration\n"

        return outString
    }
}
