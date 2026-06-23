package com.unciv.logic.simulation.dataplane

import com.unciv.UncivGame
import com.unciv.logic.GameInfo
import com.unciv.logic.automation.civilization.NextTurnAutomation
import com.unciv.logic.civilization.Civilization
import com.unciv.models.ruleset.unique.GameContext
import com.unciv.utils.Log
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import kotlin.random.Random

/** Per-run wiring bundle handed to [Simulation] when the data plane is enabled. */
class DataPlaneContext(
    val config: SampleConfig,
    val vocab: Vocab,
    val policy: PolicyProvider,
    val fingerprint: String,
)

/**
 * Glue between the engine and the data plane: the startup version/fingerprint check, the shard
 * header/`schema.json` builder, the deterministic policy-RNG factory, and the per-worker
 * [ShardRecorder] that featurizes + records one trajectory step per deciding-civ turn.
 */
object DataPlaneHooks {

    /** Refuse (strict) or warn (default) on a fingerprint/schema-version mismatch at sim start. */
    fun startupCheck(config: SampleConfig, liveFingerprint: String) {
        val problems = buildList {
            config.expectedSchemaVersion?.let {
                if (it != SampleSchema.VERSION) add("schemaVersion expected=$it live=${SampleSchema.VERSION}")
            }
            config.expectedRulesetFingerprint?.let {
                if (it != liveFingerprint) add("rulesetFingerprint expected=$it live=$liveFingerprint")
            }
        }
        if (problems.isEmpty()) return
        val msg = "data plane provenance mismatch (datasets are perishable — regenerate, don't migrate): " +
            problems.joinToString("; ")
        if (config.strictVersioning) throw IllegalStateException(msg)
        Log.error("WARN: %s", msg)
    }

    /** Policy RNG derived from (gameId, civ, turn) via the engine's existing state-based RNG. */
    fun defaultRngFor(): (Civilization, Int) -> Random =
        { civ, turn -> GameContext(civInfo = civ).stateBasedRandom("dataplane-policy-$turn") }

    // ---- per-game recorder routing (the static onCivTurn hook dispatches to the right shard) ----
    private val recorders = ConcurrentHashMap<GameInfo, ShardRecorder>()

    /** Install the civ-turn recording hook (once per run). The closure routes each civ's turn to
     *  its game's recorder; games with no registered recorder are ignored. */
    fun install(policy: PolicyProvider) {
        NextTurnAutomation.onCivTurn = { civ -> recorders[civ.gameInfo]?.recordCivTurn(civ, policy) }
    }

    fun register(gameInfo: GameInfo, recorder: ShardRecorder) { recorders[gameInfo] = recorder }
    fun unregister(gameInfo: GameInfo) { recorders.remove(gameInfo) }

    /** Clear the hook + registry at run end (restores byte-identical interactive behavior). */
    fun uninstall() { NextTurnAutomation.onCivTurn = null; recorders.clear() }

    private fun esc(s: String) = s.replace("\\", "\\\\").replace("\"", "\\\"")

    /** Provenance + caps + layout header JSON (also written verbatim as `schema.json`). */
    fun buildHeaderJson(
        gameInfo: GameInfo, fingerprint: String, gameId: String, seed: Long,
        caps: SampleCaps, blocks: List<Observation.Block>,
    ): String {
        val v = UncivGame.VERSION
        val layoutJson = blocks.joinToString(",") { b ->
            val kind = if (b.kind == BlockKind.VARIABLE) "var" else "fixed"
            """{"name":"${esc(b.name)}","dtype":"${b.dtype}","kind":"$kind","perItem":${b.perItem},"len":${b.values.size}}"""
        }
        val channels = SampleSchema.SPATIAL_CHANNELS.joinToString(",") { "\"${esc(it)}\"" }
        return "{" +
            """"schemaVersion":${SampleSchema.VERSION},""" +
            """"uncivVersionText":"${esc(v.text)}","uncivVersionNumber":${v.number},""" +
            """"compatibilityNumber":${com.unciv.logic.CompatibilityVersion.CURRENT_COMPATIBILITY_NUMBER},""" +
            """"gitSha":${gitSha()?.let { "\"${esc(it)}\"" } ?: "null"},""" +
            """"rulesetFingerprint":"${esc(fingerprint)}",""" +
            """"gameId":"${esc(gameId)}","seed":$seed,""" +
            """"nTiles":${gameInfo.tileMap.tileList.size},""" +
            """"caps":{"maxMajorCivs":${caps.maxMajorCivs},"maxCityStates":${caps.maxCityStates},""" +
            """"maxOwnCities":${caps.maxOwnCities},"maxVisOppCities":${caps.maxVisOppCities},""" +
            """"maxOwnUnits":${caps.maxOwnUnits},"maxVisOppUnits":${caps.maxVisOppUnits}},""" +
            """"spatialChannels":[$channels],""" +
            """"layout":[$layoutJson]""" +
            "}"
    }

    /** Best-effort git SHA at generation time (no build constant exists). */
    private fun gitSha(): String? = try {
        val p = ProcessBuilder("git", "rev-parse", "HEAD").redirectErrorStream(true).start()
        val sha = p.inputStream.bufferedReader().readLine()?.trim()
        if (p.waitFor() == 0 && !sha.isNullOrEmpty()) sha else null
    } catch (_: Exception) { null }
}

/**
 * One recorder per worker → one shard file. Featurizes the deciding civ, samples the policy's
 * action per factored head, and writes a step record. NOT thread-shared.
 */
class ShardRecorder(
    private val gameInfo: GameInfo,
    private val vocab: Vocab,
    private val config: SampleConfig,
    private val fingerprint: String,
    private val gameId: String,
    private val seed: Long,
    baseName: String,
) {
    private val featurizer = Featurizer(gameInfo, vocab, config)
    private val emitter = TrajectoryEmitter(File(config.outputDir ?: "."), baseName)
    private var opened = false
    private val seenCivs = HashSet<String>()

    private val actionHeads = listOf("mask_tech", "mask_policy", "mask_greatPerson", "mask_diplomaticVote")

    /** Record one trajectory step for civ [x]. `isFirst` is computed from first-occurrence; the
     *  terminal flag is left false in v1 (the game-end step is not separately buffered). */
    fun recordCivTurn(x: Civilization, policy: PolicyProvider) {
        val obs = featurizer.observe(x)
        val turn = gameInfo.turns
        val isFirst = seenCivs.add(x.civID)

        // sample the chosen action per civ-level head (recorded as the step's action labels)
        val actions = FloatArray(actionHeads.size) { i ->
            val head = actionHeads[i].removePrefix("mask_")
            val mask = obs.block(actionHeads[i]).let { BooleanArray(it.size) { k -> it[k] != 0f } }
            policy.chooseIndex(head, x, mask, turn).toFloat()
        }

        val blocks = obs.blocks +
            Observation.Block("actions", SampleSchema.DT_F32, BlockKind.FIXED, 0, actions)
        if (!opened) {
            val header = DataPlaneHooks.buildHeaderJson(gameInfo, fingerprint, gameId, seed, config.caps, blocks)
            emitter.open(header)
            writeSchemaSidecar(header)
            opened = true
        }

        val civSlot = gameInfo.civilizations.indexOf(x)
        val payload = LeBuffer(blocks.sumOf { it.values.size } + 64)
            .i32(turn).i32(civSlot)
            .u8(if (isFirst) 1 else 0).u8(0).u8(0) // isFirst | isLast | isTerminal (last/terminal v1=0)
            .u8(if (obs.overflow) 1 else 0).f32(0f) // overflow flag | reward placeholder
        for (b in blocks) Observation.writeBlock(payload, b)
        emitter.record(payload.toByteArray())
    }

    private fun writeSchemaSidecar(headerJson: String) {
        try {
            val dir = File(config.outputDir ?: ".")
            dir.mkdirs()
            File(dir, "schema.json").writeText(headerJson)
        } catch (_: Exception) { /* sidecar is best-effort; the shard header is authoritative */ }
    }

    fun close(): File? = if (opened) emitter.finalizeShard() else null
    fun abort() = emitter.abort()
    fun checksum(): Long = emitter.calculateChecksum()
}
