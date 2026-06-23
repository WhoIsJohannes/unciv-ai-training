package com.unciv.app.desktop

import com.unciv.UncivGame
import com.unciv.logic.GameStarter
import com.unciv.logic.simulation.Simulation
import com.unciv.logic.simulation.dataplane.DataPlaneContext
import com.unciv.logic.simulation.dataplane.DataPlaneHooks
import com.unciv.logic.simulation.dataplane.RandomPolicy
import com.unciv.logic.simulation.dataplane.RulesetFingerprint
import com.unciv.logic.simulation.dataplane.SampleCaps
import com.unciv.logic.simulation.dataplane.SampleConfig
import com.unciv.logic.simulation.dataplane.ScenarioGenerator
import com.unciv.logic.simulation.dataplane.Vocab
import com.unciv.models.metadata.GameSettings
import com.unciv.models.ruleset.RulesetCache
import com.unciv.utils.Log
import kotlin.time.ExperimentalTime

/**
 * Headless self-play data-generation runner (mirrors SimBenchmark's bootstrap). Generates one
 * randomized GnK scenario and drives RandomPolicy self-play to completion with the trajectory
 * emitter enabled, writing shards + `schema.json` to the output directory.
 *
 *   ./gradlew :desktop:dataGen --args="<outputDir> <maxTurns> <episodes> <seedBase>"
 */
@OptIn(ExperimentalTime::class)
object DataPlaneGen {

    @JvmStatic
    fun main(args: Array<String>) {
        Log.backend = DesktopLogBackend()
        val outputDir = args.getOrNull(0) ?: "dataplane-shards"
        val maxTurns = args.getOrNull(1)?.toIntOrNull() ?: 60
        val episodes = args.getOrNull(2)?.toIntOrNull() ?: 1
        val seedBase = args.getOrNull(3)?.toLongOrNull() ?: 12345L

        val game = UncivGame(true)
        UncivGame.Current = game
        UncivGame.Current.settings = GameSettings().apply {
            showTutorials = false
            turnsBetweenAutosaves = 10000
        }
        RulesetCache.loadRulesets(consoleMode = true)

        val maxMapRadius = args.getOrNull(4)?.toIntOrNull() ?: Int.MAX_VALUE
        val caps = SampleCaps.DEFAULT
        val gen = ScenarioGenerator(caps, maxMapRadius = maxMapRadius)
        val spec = gen.generate(seedBase, episode = 0)
        println("BENCH| scenario: ${spec.episodeLogJson}")

        val newGameInfo = GameStarter.startNewGame(spec.gameSetupInfo)
        UncivGame.Current.gameInfo = newGameInfo
        val ruleset = newGameInfo.ruleset
        val vocab = Vocab(ruleset)
        val fingerprint = RulesetFingerprint.compute(ruleset)
        println("BENCH| rulesetFingerprint=$fingerprint schemaVersion=${com.unciv.logic.simulation.dataplane.SampleSchema.VERSION}")

        val config = SampleConfig(enabled = true, outputDir = outputDir, deterministicShuffle = true, caps = caps)
        DataPlaneHooks.startupCheck(config, fingerprint)

        val policy = RandomPolicy(DataPlaneHooks.defaultRngFor())
        val ctx = DataPlaneContext(config, vocab, policy, fingerprint)

        val sim = Simulation(
            newGameInfo,
            simulationsPerThread = episodes,
            threadsNumber = 1,
            maxTurns = maxTurns,
            statTurns = listOf(),
            dataPlane = ctx,
        )
        sim.start()
        println("BENCH| data-plane generation complete; shards + schema.json in '$outputDir'")
    }
}
