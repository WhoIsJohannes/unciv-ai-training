package com.unciv.app.desktop

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.unciv.Constants
import com.unciv.UncivGame
import com.unciv.logic.GameInfo
import com.unciv.logic.GameStarter
import com.unciv.logic.civilization.PlayerType
import com.unciv.logic.map.MapParameters
import com.unciv.logic.map.MapShape
import com.unciv.logic.map.MapSize
import com.unciv.logic.map.MirroringType
import com.unciv.logic.simulation.SimStats
import com.unciv.logic.simulation.Simulation
import com.unciv.logic.simulation.dataplane.DataPlaneContext
import com.unciv.logic.simulation.dataplane.DataPlaneHooks
import com.unciv.logic.simulation.dataplane.Featurizer
import com.unciv.logic.simulation.dataplane.PolicyProvider
import com.unciv.logic.simulation.dataplane.RandomPolicy
import com.unciv.logic.simulation.dataplane.RoutingPolicy
import com.unciv.logic.simulation.dataplane.RulesetFingerprint
import com.unciv.logic.simulation.dataplane.SampleCaps
import com.unciv.logic.simulation.dataplane.SampleConfig
import com.unciv.logic.simulation.dataplane.SampleSchema
import com.unciv.logic.simulation.dataplane.Vocab
import com.unciv.models.metadata.GameParameters
import com.unciv.models.metadata.GameSettings
import com.unciv.models.metadata.GameSetupInfo
import com.unciv.models.metadata.BaseRuleset
import com.unciv.models.metadata.Player
import com.unciv.models.ruleset.Ruleset
import com.unciv.models.ruleset.RulesetCache
import com.unciv.models.ruleset.Speed
import com.unciv.models.ruleset.nation.Nation
import com.unciv.models.skins.SkinCache
import com.unciv.models.tilesets.TileSetCache
import com.unciv.utils.Log
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.time.ExperimentalTime

/**
 * Headless self-play entrypoint — the JVM half of the round loop (`python/unciv_train/run_loop.py`).
 * The learner civ is the pinned nation [Constants.simulationCiv1]; the opponent is
 * [Constants.simulationCiv2] (RandomPolicy — a stationary opponent, v1). Map config = the Tiny 2-civ
 * GnK setup cloned from `ConsoleLauncher` so games play to a real victory under a high turn cap.
 *
 * Modes (arg[0]):
 *  - `gen <model|random> <outDir> <nGames> <maxTurns> <threads> <seed>` — generate trajectory shards
 *    (learner driven by the model, or RandomPolicy for round 0). Emitter ON.
 *  - `eval <model> <mGames> <maxTurns> <threads> <seed>` — OnnxPolicy(learner) vs RandomPolicy; prints
 *    one machine-readable `EVAL_RESULT {json}` line (games, wins, winrate, binomial pval, onnx_decisions).
 *  - `parity-dump <seed> <obsOut>` — write a fixed `concat(global,acting_civ)` observation vector.
 *  - `parity-run <model> <obsIn> <logitsOut>` — run the model on that vector, write JVM logits (the
 *    JVM side of the anti-drift PARITY test).
 */
object SelfPlayRunner {

    private const val LEARNER = Constants.simulationCiv1
    private const val OPPONENT = Constants.simulationCiv2

    @ExperimentalTime
    @JvmStatic
    fun main(args: Array<String>) {
        val mode = args.getOrNull(0) ?: error("usage: SelfPlayRunner <gen|eval|parity-dump|parity-run> ...")
        when (mode) {
            "gen" -> { bootstrap(); gen(args) }
            "eval" -> { bootstrap(); eval(args) }
            "parity-dump" -> { bootstrap(); parityDump(args) }
            "parity-run" -> parityRun(args)   // ORT only — no engine bootstrap needed
            "parity-dump-rich" -> { bootstrap(); parityDumpRich(args) }
            "parity-run-rich" -> parityRunRich(args)   // ORT only — multi-tensor v2/v3 contract
            "adjacency-dump" -> { bootstrap(); adjacencyDump(args) }   // v3 FND-0036 fidelity harness
            "bench-onnx" -> { bootstrap(); benchOnnx(args) }            // D8 throughput guard (70% gate)
            "trace" -> { bootstrap(); trace(args) }
            else -> error("unknown mode '$mode'")
        }
    }

    private fun bootstrap() {
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
    }

    /** GnK ruleset with the two pinned simulation nations registered (mirrors ConsoleLauncher). */
    private fun setupRuleset(): Ruleset {
        val ruleset = RulesetCache[BaseRuleset.Civ_V_GnK.fullName]!!
        ruleset.nations[LEARNER] = Nation().apply { name = LEARNER }
        ruleset.nations[OPPONENT] = Nation().apply { name = OPPONENT }
        return ruleset
    }

    private fun gameParameters(ruleset: Ruleset): GameParameters = GameParameters().apply {
        difficulty = "King"
        numberOfCityStates = 0
        speed = Speed.DEFAULT
        noBarbarians = true
        players = ArrayList<Player>().apply {
            add(Player(ruleset.nations[LEARNER]!!))
            add(Player(ruleset.nations[OPPONENT]!!))
            add(Player(Constants.spectator, PlayerType.Human))
        }
        players.last().setNationTransient(ruleset)
    }

    /** Resolve a map-size NAME (CLI arg) to its predefined [MapSize]. Fails LOUD on an unknown name
     *  (council 🟡) — a silent Tiny fallback would corrupt a Medium experiment on a typo. */
    private fun resolveMapSize(name: String): MapSize = when (name) {
        "Tiny" -> MapSize.Tiny
        "Small" -> MapSize.Small
        "Medium" -> MapSize.Medium
        "Large" -> MapSize.Large
        "Huge" -> MapSize.Huge
        else -> error("unknown map size '$name' (expected Tiny|Small|Medium|Large|Huge)")
    }

    private fun mapParameters(seed: Long = 0, mapSizeName: String = "Tiny"): MapParameters = MapParameters().apply {
        mapSize = resolveMapSize(mapSizeName)
        noRuins = true
        noNaturalWonders = true
        legendaryStart = true
        strategicBalance = true
        mirroring = MirroringType.aroundCenterTile
        waterThreshold -= 0.1f
        if (seed != 0L) this.seed = seed
    }

    /** Template game info Simulation re-starts each iteration from (with per-iteration seeds). */
    @ExperimentalTime
    private fun buildBaseGameInfo(ruleset: Ruleset, mapSizeName: String = "Tiny"): GameInfo {
        val gsi = GameSetupInfo(gameParameters(ruleset), mapParameters(0, mapSizeName))
        val newGame = GameStarter.startNewGame(gsi)
        newGame.gameParameters.victoryTypes = ArrayList(newGame.ruleset.victories.keys)
        UncivGame.Current.gameInfo = newGame
        return newGame
    }

    @ExperimentalTime
    private fun gen(args: Array<String>) {
        val modelArg = args.getOrNull(1) ?: "random"
        val outDir = args.getOrNull(2) ?: "selfplay-shards"
        val nGames = args.getOrNull(3)?.toIntOrNull() ?: 24
        val maxTurns = args.getOrNull(4)?.toIntOrNull() ?: 325
        val threads = args.getOrNull(5)?.toIntOrNull() ?: 1
        val seed = args.getOrNull(6)?.toLongOrNull() ?: 1L
        val mapSizeName = args.getOrNull(7) ?: "Tiny"

        val ruleset = setupRuleset()
        val fingerprint = RulesetFingerprint.compute(ruleset)
        val vocab = Vocab(ruleset)
        val config = SampleConfig(enabled = true, outputDir = outDir, deterministicShuffle = true, caps = SampleCaps.DEFAULT)
        DataPlaneHooks.startupCheck(config, fingerprint)

        val learner: PolicyProvider =
            if (modelArg == "random") RandomPolicy(DataPlaneHooks.defaultRngFor())
            else OnnxPolicy(modelArg, vocab, config, DataPlaneHooks.defaultRngFor(), eval = false, SampleSchema.VERSION, fingerprint)
        val policy = RoutingPolicy(LEARNER, learner, RandomPolicy(DataPlaneHooks.defaultRngFor()))

        val base = buildBaseGameInfo(ruleset, mapSizeName)
        val perThread = (nGames + threads - 1) / threads
        val sim = Simulation(
            base, simulationsPerThread = perThread, threadsNumber = threads, maxTurns = maxTurns,
            statTurns = listOf(), dataPlane = DataPlaneContext(config, vocab, policy, fingerprint),
            scoreLeaderOnTimeout = true, seedBase = seed,
        )
        sim.start()
        (learner as? OnnxPolicy)?.close()
        println("SELFPLAY_GEN_DONE games=${sim.steps.size} dir=$outDir model=$modelArg seed=$seed mapSize=$mapSizeName")
    }

    @ExperimentalTime
    private fun eval(args: Array<String>) {
        val modelArg = args.getOrNull(1) ?: error("eval: model path required")
        val mGames = args.getOrNull(2)?.toIntOrNull() ?: 100
        val maxTurns = args.getOrNull(3)?.toIntOrNull() ?: 325
        val threads = args.getOrNull(4)?.toIntOrNull() ?: 1
        val seed = args.getOrNull(5)?.toLongOrNull() ?: 999_000L
        val mapSizeName = args.getOrNull(6) ?: "Tiny"

        val ruleset = setupRuleset()
        val fingerprint = RulesetFingerprint.compute(ruleset)
        val vocab = Vocab(ruleset)
        // Control-only (no shards): emit=false because config.enabled=false.
        val config = SampleConfig(enabled = false, deterministicShuffle = true, caps = SampleCaps.DEFAULT)

        val onnx = OnnxPolicy(modelArg, vocab, config, DataPlaneHooks.defaultRngFor(), eval = true, SampleSchema.VERSION, fingerprint)
        val policy = RoutingPolicy(LEARNER, onnx, RandomPolicy(DataPlaneHooks.defaultRngFor()))

        val base = buildBaseGameInfo(ruleset, mapSizeName)
        val perThread = (mGames + threads - 1) / threads
        val sim = Simulation(
            base, simulationsPerThread = perThread, threadsNumber = threads, maxTurns = maxTurns,
            statTurns = listOf(), dataPlane = DataPlaneContext(config, vocab, policy, fingerprint),
            scoreLeaderOnTimeout = true, seedBase = seed,
        )
        val t0 = System.nanoTime()
        sim.start()
        val wallS = (System.nanoTime() - t0) / 1e9
        val games = sim.steps.size
        val turns = sim.steps.sumOf { it.turns }
        val wins = sim.steps.count { it.winner == LEARNER }
        val winrate = if (games > 0) wins.toDouble() / games else 0.0
        val pval = if (games > 0) SimStats.binomialTest(wins.toDouble(), games.toDouble(), 0.5, "greater") else 1.0
        val decisions = onnx.decisionCount()
        onnx.close()
        // D8 throughput fields: turns/s (data-gen rate) + ms/decision (ONNX forward cost proxy).
        val turnsPerSec = if (wallS > 0) turns / wallS else 0.0
        val msPerDecision = if (decisions > 0) wallS * 1000.0 / decisions else -1.0
        println(
            "EVAL_RESULT {" +
                "\"games\":$games,\"wins\":$wins,\"winrate\":$winrate,\"pval\":$pval," +
                "\"learner\":\"$LEARNER\",\"seed\":$seed,\"onnx_decisions\":$decisions," +
                "\"turns\":$turns,\"wall_clock_s\":$wallS,\"turns_per_sec\":$turnsPerSec,\"ms_per_decision\":$msPerDecision}"
        )
    }

    @ExperimentalTime
    private fun parityDump(args: Array<String>) {
        val seed = args.getOrNull(1)?.toLongOrNull() ?: 4242L
        val obsOut = args.getOrNull(2) ?: "parity-obs.csv"
        val ruleset = setupRuleset()
        val vocab = Vocab(ruleset)
        val config = SampleConfig(enabled = false, caps = SampleCaps.DEFAULT)
        val gsi = GameSetupInfo(gameParameters(ruleset), mapParameters(seed))
        val game = GameStarter.startNewGame(gsi)
        UncivGame.Current.gameInfo = game
        val learner = game.civilizations.first { it.civID == LEARNER }
        val obs = Featurizer(game, vocab, config).observe(learner)
        val input = obs.block("global") + obs.block("acting_civ")
        File(obsOut).writeText(input.joinToString(","))
        println("PARITY_DUMP width=${input.size} seed=$seed -> $obsOut")
    }

    /** Run ONE game (learner vs RandomPolicy) turn-by-turn, logging per-civ stats to a CSV so a human
     *  can SEE how the game develops, and saving the final game as a loadable Unciv save (open it in
     *  the desktop app: Load Game → it shows the map/cities/winner). */
    @ExperimentalTime
    private fun trace(args: Array<String>) {
        val modelArg = args.getOrNull(1) ?: "random"
        val seed = args.getOrNull(2)?.toLongOrNull() ?: 12345L
        val maxTurns = args.getOrNull(3)?.toIntOrNull() ?: 1000
        val outCsv = args.getOrNull(4) ?: "game-trace.csv"
        val saveFile = args.getOrNull(5)

        val ruleset = setupRuleset()
        val fingerprint = RulesetFingerprint.compute(ruleset)
        val vocab = Vocab(ruleset)
        val config = SampleConfig(enabled = false, caps = SampleCaps.DEFAULT)
        val learner: PolicyProvider =
            if (modelArg == "random") RandomPolicy(DataPlaneHooks.defaultRngFor())
            else OnnxPolicy(modelArg, vocab, config, DataPlaneHooks.defaultRngFor(), eval = true, SampleSchema.VERSION, fingerprint)
        val policy = RoutingPolicy(LEARNER, learner, RandomPolicy(DataPlaneHooks.defaultRngFor()))
        DataPlaneHooks.install(policy)

        val game = GameStarter.startNewGame(GameSetupInfo(gameParameters(ruleset), mapParameters(seed)))
        game.gameParameters.victoryTypes = ArrayList(game.ruleset.victories.keys)
        UncivGame.Current.gameInfo = game
        game.gameId = "trace-$seed"
        DataPlaneHooks.registerGame(game, vocab, config, fingerprint, game.gameId, seed, "trace", emit = false)

        val majors = game.civilizations.filter { it.isMajorCiv() && !it.isSpectator() }
        val csv = StringBuilder("turn," + majors.joinToString(",") { c ->
            "${c.civID}_score,${c.civID}_cities,${c.civID}_techs,${c.civID}_era"
        } + "\n")

        game.simulateUntilWin = true
        var winner: String? = null
        var victoryType: String? = null
        while (game.turns < maxTurns) {
            game.simulateMaxTurns = game.turns + 1
            game.nextTurn()
            csv.append(game.turns)
            for (c in majors) csv.append(",${c.calculateTotalScore()},${c.cities.size},${c.tech.researchedTechnologies.size},${c.tech.era.eraNumber}")
            csv.append("\n")
            victoryType = game.getCurrentPlayerCivilization().victoryManager.getVictoryTypeAchieved()
            if (victoryType != null) { winner = game.currentPlayer; break }
            if (majors.count { it.isAlive() } <= 1) { winner = majors.firstOrNull { it.isAlive() }?.civID; break }
        }
        if (winner == null) winner = SimStats.scoreLeader(game)?.civID
        DataPlaneHooks.finalizeGame(game, winner)
        DataPlaneHooks.uninstall()

        File(outCsv).writeText(csv.toString())
        var savedTo = ""
        if (saveFile != null) try {
            File(saveFile).writeText(com.unciv.logic.files.UncivFiles.gameInfoToString(game))
            savedTo = " save=$saveFile"
        } catch (e: Exception) { savedTo = " (save failed: ${e.message})" }

        println("TRACE_DONE turns=${game.turns} winner=${winner ?: "draw"} " +
            "victory=${victoryType ?: "none (score-leader/draw)"} learner=$LEARNER csv=$outCsv$savedTo")
    }

    private fun parityRun(args: Array<String>) {
        val model = args.getOrNull(1) ?: error("parity-run: model path required")
        val obsIn = args.getOrNull(2) ?: error("parity-run: obs file required")
        val logitsOut = args.getOrNull(3) ?: "parity-jvm-logits.json"
        val input = File(obsIn).readText().trim().split(",").map { it.trim().toFloat() }.toFloatArray()
        val env = OrtEnvironment.getEnvironment()
        val session = env.createSession(File(model).absolutePath, OrtSession.SessionOptions())
        try {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, input.size.toLong())).use { t ->
                session.run(mapOf(SampleSchema.OnnxContract.INPUT_NAME to t)).use { res ->
                    @Suppress("UNCHECKED_CAST")
                    val tech = (res.get(SampleSchema.OnnxContract.OUTPUT_TECH).get() as OnnxTensor).value as Array<FloatArray>
                    @Suppress("UNCHECKED_CAST")
                    val policy = (res.get(SampleSchema.OnnxContract.OUTPUT_POLICY).get() as OnnxTensor).value as Array<FloatArray>
                    File(logitsOut).writeText(
                        "{\"tech\":[${tech[0].joinToString(",")}],\"policy\":[${policy[0].joinToString(",")}]}"
                    )
                }
            }
        } finally {
            session.close()
        }
        println("PARITY_RUN -> $logitsOut")
    }

    // --- Rich (contract v2) multi-tensor parity ---------------------------------------------------
    // Fixture text format (shared with the Python test, one block per line):
    //   "<name> vec <floats...>"                      — global, acting_civ
    //   "<name> set <count> <width> <floats...>"       — spatial + entity token sets (flat row-major)

    @ExperimentalTime
    private fun parityDumpRich(args: Array<String>) {
        val seed = args.getOrNull(1)?.toLongOrNull() ?: 4242L
        val obsOut = args.getOrNull(2) ?: "parity-obs-rich.txt"
        val mapSizeName = args.getOrNull(3) ?: "Tiny"
        val ruleset = setupRuleset()
        val vocab = Vocab(ruleset)
        val config = SampleConfig(enabled = false, caps = SampleCaps.DEFAULT)
        val gsi = GameSetupInfo(gameParameters(ruleset), mapParameters(seed, mapSizeName))
        val game = GameStarter.startNewGame(gsi)
        UncivGame.Current.gameInfo = game
        val learner = game.civilizations.first { it.civID == LEARNER }
        val obs = Featurizer(game, vocab, config).observe(learner)
        val sb = StringBuilder()
        sb.appendLine("global vec " + obs.block("global").joinToString(" "))
        sb.appendLine("acting_civ vec " + obs.block("acting_civ").joinToString(" "))
        for (name in SampleSchema.OnnxContract.RICH_TOKEN_NAMES) {
            val width = if (name == "spatial") SampleSchema.NUM_SPATIAL_CHANNELS
                        else obs.blocks.first { it.name == name }.perItem
            val vals = obs.block(name)
            val count = if (width > 0) vals.size / width else 0
            sb.appendLine("$name set $count $width " + vals.joinToString(" "))
        }
        File(obsOut).writeText(sb.toString())
        println("PARITY_DUMP_RICH seed=$seed mapSize=$mapSizeName -> $obsOut")
    }

    private fun parityRunRich(args: Array<String>) {
        val model = args.getOrNull(1) ?: error("parity-run-rich: model path required")
        val obsIn = args.getOrNull(2) ?: error("parity-run-rich: obs file required")
        val logitsOut = args.getOrNull(3) ?: "parity-jvm-logits-rich.json"

        var global = FloatArray(0)
        var acting = FloatArray(0)
        val tokens = ArrayList<Triple<String, FloatArray, Int>>()
        var neighborIdx: LongArray? = null   // v3 structured: hex-GNN neighbor inputs from the fixture
        var neighborMask: FloatArray? = null
        var neighborN = 0
        var neighborDeg = SampleSchema.OnnxContract.HEX_DEGREE
        for (raw in File(obsIn).readLines()) {
            val line = raw.trim()
            if (line.isEmpty()) continue
            val t = line.split(Regex("\\s+"))
            val name = t[0]
            when (t[1]) {
                "vec" -> {
                    val v = FloatArray(t.size - 2) { t[it + 2].toFloat() }
                    if (name == "global") global = v else if (name == "acting_civ") acting = v
                }
                "set" -> {
                    val count = t[2].toInt(); val width = t[3].toInt()
                    val v = FloatArray(count * width) { t[it + 4].toFloat() }
                    tokens.add(Triple(name, v, width))
                }
                "adj" -> {   // <name> adj <N> <deg> <N*deg ints/floats> — neighbor_index (int64) / neighbor_mask
                    val nn = t[2].toInt(); val deg = t[3].toInt()
                    neighborN = nn; neighborDeg = deg
                    if (name == SampleSchema.OnnxContract.INPUT_NEIGHBOR_INDEX)
                        neighborIdx = LongArray(nn * deg) { t[it + 4].toFloat().toLong() }
                    else if (name == SampleSchema.OnnxContract.INPUT_NEIGHBOR_MASK)
                        neighborMask = FloatArray(nn * deg) { t[it + 4].toFloat() }
                }
            }
        }
        val env = OrtEnvironment.getEnvironment()
        val session = env.createSession(File(model).absolutePath, OrtSession.SessionOptions())
        // build tensors INSIDE the try so a build/parse failure still closes the session (council 🟡)
        var inputs: LinkedHashMap<String, OnnxTensor>? = null
        try {
            inputs = OnnxPolicy.richTensorsFromArrays(env, global, acting, tokens)
            val nIdx = neighborIdx; val nMask = neighborMask
            if (nIdx != null && nMask != null) {   // v3 structured: feed fixture-built neighbor tensors
                inputs[SampleSchema.OnnxContract.INPUT_NEIGHBOR_INDEX] =
                    OnnxTensor.createTensor(env, LongBuffer.wrap(nIdx), longArrayOf(1, neighborN.toLong(), neighborDeg.toLong()))
                inputs[SampleSchema.OnnxContract.INPUT_NEIGHBOR_MASK] =
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(nMask), longArrayOf(1, neighborN.toLong(), neighborDeg.toLong()))
            }
            session.run(inputs).use { res ->
                @Suppress("UNCHECKED_CAST")
                val tech = (res.get(SampleSchema.OnnxContract.OUTPUT_TECH).get() as OnnxTensor).value as Array<FloatArray>
                @Suppress("UNCHECKED_CAST")
                val policy = (res.get(SampleSchema.OnnxContract.OUTPUT_POLICY).get() as OnnxTensor).value as Array<FloatArray>
                File(logitsOut).writeText(
                    "{\"tech\":[${tech[0].joinToString(",")}],\"policy\":[${policy[0].joinToString(",")}]}"
                )
            }
        } finally {
            inputs?.values?.forEach { tt -> try { tt.close() } catch (_: Exception) {} }
            session.close()
        }
        println("PARITY_RUN_RICH -> $logitsOut")
    }

    /** v3 FND-0036 fidelity dump: emit per-tile (x,y) + the LIVE engine's degree-6 neighbor
     *  zeroBasedIndices (getIfTileExistsOrNull + the SAME [OnnxPolicy.HEX_OFFSETS] used at inference)
     *  on a WORLD-WRAP map, so a Python test can assert hexgraph.build_neighbor_graph reproduces it. */
    @ExperimentalTime
    private fun adjacencyDump(args: Array<String>) {
        val seed = args.getOrNull(1)?.toLongOrNull() ?: 4242L
        val out = args.getOrNull(2) ?: "adjacency-dump.json"
        val mapSizeName = args.getOrNull(3) ?: "Tiny"
        val ruleset = setupRuleset()
        val mp = mapParameters(seed, mapSizeName).apply { worldWrap = true }  // exercise the wrap branch
        val game = GameStarter.startNewGame(GameSetupInfo(gameParameters(ruleset), mp))
        UncivGame.Current.gameInfo = game
        val tm = game.tileMap
        val n = tm.tileList.size
        val deg = SampleSchema.OnnxContract.HEX_DEGREE
        val coords = Array(n) { intArrayOf(0, 0) }
        val live = Array(n) { IntArray(deg) { -1 } }
        for (tile in tm.tileList) {
            val r = tile.zeroBasedIndex
            if (r < 0 || r >= n) continue
            val x = tile.position.x.toInt(); val y = tile.position.y.toInt()
            coords[r] = intArrayOf(x, y)
            for (d in 0 until deg) {
                val nb = tm.getIfTileExistsOrNull(x + OnnxPolicy.HEX_OFFSETS[d][0], y + OnnxPolicy.HEX_OFFSETS[d][1])
                live[r][d] = nb?.zeroBasedIndex ?: -1
            }
        }
        val effWrapRadius = if (mp.shape == MapShape.rectangular) mp.mapSize.width / 2 else mp.mapSize.radius
        val shapeOrd = when (mp.shape) { MapShape.rectangular -> 0; MapShape.flatEarth -> 2; else -> 1 }
        val sb = StringBuilder("{")
        sb.append("\"nTiles\":$n,\"effWrapRadius\":$effWrapRadius,")
        sb.append("\"worldWrap\":${if (mp.worldWrap) 1 else 0},\"shape\":$shapeOrd,")
        sb.append("\"coords\":[").append((0 until n).joinToString(",") { "[${coords[it][0]},${coords[it][1]}]" }).append("],")
        sb.append("\"live\":[").append((0 until n).joinToString(",") { live[it].joinToString(",", "[", "]") }).append("]}")
        File(out).writeText(sb.toString())
        println("ADJACENCY_DUMP nTiles=$n worldWrap=${mp.worldWrap} -> $out")
    }

    /** D8 throughput guard: measure data-gen turns/s on the SAME 2-civ training config (correct
     *  ruleset fingerprint — NOT SimBenchmark's 6-major BenchCiv config which would fail the OnnxPolicy
     *  gate) for a heuristic baseline (RandomPolicy) vs the ONNX rung, and emit a PASS/REJECT verdict
     *  (REJECT if onnx turns/s < 70% of baseline). The ladder parses the "BENCH| RUNG ... verdict=" line. */
    @ExperimentalTime
    private fun benchOnnx(args: Array<String>) {
        val modelArg = args.getOrNull(1) ?: error("bench-onnx: model path required")
        val turns = args.getOrNull(2)?.toIntOrNull() ?: 200
        val mapSizeName = args.getOrNull(3) ?: "Medium"
        val threads = args.getOrNull(4)?.toIntOrNull() ?: 1
        val seed = args.getOrNull(5)?.toLongOrNull() ?: 777_000L
        val ruleset = setupRuleset()
        val fingerprint = RulesetFingerprint.compute(ruleset)
        val vocab = Vocab(ruleset)
        val config = SampleConfig(enabled = false, deterministicShuffle = true, caps = SampleCaps.DEFAULT)
        val base = buildBaseGameInfo(ruleset, mapSizeName)

        fun runWith(policy: PolicyProvider): Triple<Double, Int, Double> {  // (turns/s, turns, wallS)
            val sim = Simulation(
                base, simulationsPerThread = 1, threadsNumber = threads, maxTurns = turns,
                statTurns = listOf(), dataPlane = DataPlaneContext(config, vocab, policy, fingerprint),
                scoreLeaderOnTimeout = true, seedBase = seed,
            )
            val t0 = System.nanoTime(); sim.start(); val s = (System.nanoTime() - t0) / 1e9
            val tt = sim.steps.sumOf { it.turns }
            return Triple(if (s > 0) tt / s else 0.0, tt, s)
        }

        // Heuristic baseline (no ONNX) — both sides RandomPolicy on the same config.
        val (baselineTps, _, _) = runWith(
            RoutingPolicy(LEARNER, RandomPolicy(DataPlaneHooks.defaultRngFor()), RandomPolicy(DataPlaneHooks.defaultRngFor())))
        // ONNX rung — learner routed to the net.
        val onnx = OnnxPolicy(modelArg, vocab, config, DataPlaneHooks.defaultRngFor(), eval = true, SampleSchema.VERSION, fingerprint)
        try {
            val (onnxTps, _, onnxWallS) = runWith(RoutingPolicy(LEARNER, onnx, RandomPolicy(DataPlaneHooks.defaultRngFor())))
            val decisions = onnx.decisionCount()
            val msPerDecision = if (decisions > 0) onnxWallS * 1000.0 / decisions else -1.0
            val ratio = if (baselineTps > 0) onnxTps / baselineTps else 0.0
            val verdict = if (ratio >= 0.70) "PASS" else "REJECT"
            println("BENCH| RUNG model=$modelArg map=$mapSizeName baseline_tps=${"%.2f".format(baselineTps)} " +
                "onnx_tps=${"%.2f".format(onnxTps)} ratio=${"%.3f".format(ratio)} " +
                "ms_per_decision=${"%.3f".format(msPerDecision)} onnx_decisions=$decisions verdict=$verdict")
        } finally {
            onnx.close()   // closes the OrtSession (no native leak across rungs — FND-0048)
        }
    }
}
