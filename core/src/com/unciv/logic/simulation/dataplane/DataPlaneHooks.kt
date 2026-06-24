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
 * Glue between the engine and the data plane. Two responsibilities, deliberately separated so they
 * compose for both GENERATE and EVAL self-play:
 *
 *  - **CONTROL** (always-on when a policy is installed): at each deciding-civ turn the installed
 *    [PolicyProvider] DRIVES that civ's tech + policy — it chooses an action from the legal mask and
 *    the choice is APPLIED to the game (not just labelled). Pre-filling `tech.techsToResearch` makes
 *    `NextTurnAutomation.chooseTechToResearch` respect it; the chosen policy is adopted and
 *    `adoptPolicy` is guarded to skip the heuristic for controlled civs. Decision-gated: a TECH
 *    action is taken only when the research queue is empty; a POLICY action only when a slot is free
 *    (otherwise the head records −1 = "no decision this turn"). This is what makes REINFORCE valid
 *    and "OnnxPolicy vs RandomPolicy" a real comparison rather than heuristic-vs-heuristic.
 *  - **EMIT** (only when a [ShardRecorder] is registered, i.e. GENERATE): featurize + write the
 *    trajectory step, recording the SAME action that control just applied (recorded == applied).
 *
 * `onCivTurn` is `null` in normal play ⇒ ZERO behavior change. Set/cleared by the self-play runner.
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

    // ---- per-game state: a Featurizer (always, for control masks + emit obs) + an optional recorder ----
    private class GameState(val featurizer: Featurizer, val vocab: Vocab, val recorder: ShardRecorder?)
    private val games = ConcurrentHashMap<GameInfo, GameState>()
    @Volatile private var installedPolicy: PolicyProvider? = null

    /** Install the civ-turn hook once per run: control every registered game's deciding civ, and
     *  emit if that game has a recorder. */
    fun install(policy: PolicyProvider) {
        installedPolicy = policy
        NextTurnAutomation.onCivTurn = { civ -> games[civ.gameInfo]?.let { handleCivTurn(civ, it) } }
    }

    /** Register a game for control (+ emission if [emit]). Both GENERATE and EVAL register; only
     *  GENERATE passes emit=true (writes shards). */
    fun registerGame(
        gameInfo: GameInfo, vocab: Vocab, config: SampleConfig, fingerprint: String,
        gameId: String, seed: Long, baseName: String, emit: Boolean,
    ) {
        val recorder = if (emit) ShardRecorder(gameInfo, vocab, config, fingerprint, gameId, seed, baseName) else null
        games[gameInfo] = GameState(Featurizer(gameInfo, vocab, config), vocab, recorder)
    }

    /** End-of-game: emit the per-civ terminal reward record (if emitting), publish the shard, and
     *  unregister. [winnerCivId] is the winning civ's id, or null for a draw. */
    fun finalizeGame(gameInfo: GameInfo, winnerCivId: String?): File? {
        val state = games.remove(gameInfo) ?: return null
        val rec = state.recorder ?: return null
        rec.recordTerminal(winnerCivId)
        return rec.close()
    }

    fun abortGame(gameInfo: GameInfo) { games.remove(gameInfo)?.recorder?.abort() }

    /** Clear the hook + registry at run end (restores byte-identical interactive behavior). */
    fun uninstall() { NextTurnAutomation.onCivTurn = null; installedPolicy = null; games.clear() }

    /** True iff a controlling policy is installed for [civ] (a registered, non-spectator major civ).
     *  Used by `NextTurnAutomation.adoptPolicy` to skip the heuristic for controlled civs. */
    fun controls(civ: Civilization): Boolean =
        installedPolicy != null && civ.isMajorCiv() && !civ.isSpectator() && games.containsKey(civ.gameInfo)

    private fun handleCivTurn(civ: Civilization, state: GameState) {
        val policy = installedPolicy ?: return
        if (!civ.isMajorCiv() || civ.isSpectator()) return
        val obs = state.featurizer.observe(civ)
        val turn = civ.gameInfo.turns
        val actions = chooseAndApply(civ, policy, state.vocab, obs, turn)
        state.recorder?.recordStep(civ, obs, actions, turn)
    }

    /** Choose (and APPLY) the controlled civ-level actions. Returns the per-head action vector in
     *  [SampleSchema.MASK_HEADS] order; −1 = "no decision this head this turn". v1 controls tech +
     *  policy only; greatPerson/diplomaticVote stay heuristic (recorded as −1). */
    private fun chooseAndApply(civ: Civilization, policy: PolicyProvider, vocab: Vocab, obs: Observation, turn: Int): FloatArray {
        val actions = FloatArray(SampleSchema.MASK_HEADS.size) { -1f }

        // TECH — decision only when no current research target (queue empty); pre-fill ⇒ heuristic respects it.
        if (civ.tech.techsToResearch.isEmpty()) {
            val mask = boolMask(obs, "mask_tech")
            val idx = policy.chooseIndex("tech", civ, mask, turn)
            actions[0] = idx.toFloat()
            if (idx >= 0) vocab.techId(idx)?.let { if (civ.tech.canBeResearched(it)) civ.tech.techsToResearch.add(it) }
        }

        // POLICY — decision only when a free policy slot is available; adopt the chosen policy.
        if (civ.policies.canAdoptPolicy()) {
            val mask = boolMask(obs, "mask_policy")
            val idx = policy.chooseIndex("policy", civ, mask, turn)
            actions[1] = idx.toFloat()
            if (idx >= 0) vocab.policyId(idx)?.let { name ->
                val pol = civ.gameInfo.ruleset.policies[name]
                if (pol != null && civ.policies.isAdoptable(pol)) civ.policies.adopt(pol)
            }
        }
        return actions
    }

    private fun boolMask(obs: Observation, blockName: String): BooleanArray =
        obs.block(blockName).let { BooleanArray(it.size) { k -> it[k] != 0f } }

    private fun esc(s: String) = s.replace("\\", "\\\\").replace("\"", "\\\"")

    /** Provenance + caps + layout header JSON (also written verbatim as `schema.json`). */
    fun buildHeaderJson(
        gameInfo: GameInfo, fingerprint: String, gameId: String, seed: Long,
        caps: SampleCaps, blocks: List<Observation.Block>, vocab: Vocab,
    ): String {
        val v = UncivGame.VERSION
        val layoutJson = blocks.joinToString(",") { b ->
            val kind = if (b.kind == BlockKind.VARIABLE) "var" else "fixed"
            """{"name":"${esc(b.name)}","dtype":"${b.dtype}","kind":"$kind","perItem":${b.perItem},"len":${b.values.size}}"""
        }
        val channels = SampleSchema.SPATIAL_CHANNELS.joinToString(",") { "\"${esc(it)}\"" }
        // v3: vocab counts so the Python model sizes its nn.Embedding tables (num_embeddings=count+1)
        // from the schema — NEVER hardcoded. Names match the keys the Python embedding builder reads.
        val vocabCounts = """{""" +
            """"terrains":${vocab.size(Vocab.TERRAINS)},"resources":${vocab.resourceCount},""" +
            """"improvements":${vocab.size(Vocab.IMPROVEMENTS)},"buildings":${vocab.buildingCount},""" +
            """"units":${vocab.unitCount},"religions":${vocab.size(Vocab.RELIGIONS)},""" +
            """"eras":${vocab.size(Vocab.ERAS)},"policies":${vocab.policyCount},""" +
            """"policyBranches":${vocab.policyBranchCount},"promotions":${vocab.promotionCount}""" +
            """}"""
        // slot↔civId for the major civs (agnostic provenance) — lets the trainer filter to its
        // learner's steps even though turn-order shuffle varies civ_slot per game.
        val majorSlots = gameInfo.civilizations.withIndex()
            .filter { (_, c) -> c.isMajorCiv() && !c.isSpectator() }
            .joinToString(",") { (i, c) -> """{"slot":$i,"civId":"${esc(c.civID)}"}""" }
        return "{" +
            """"schemaVersion":${SampleSchema.VERSION},""" +
            """"uncivVersionText":"${esc(v.text)}","uncivVersionNumber":${v.number},""" +
            """"compatibilityNumber":${com.unciv.logic.CompatibilityVersion.CURRENT_COMPATIBILITY_NUMBER},""" +
            """"gitSha":${gitSha()?.let { "\"${esc(it)}\"" } ?: "null"},""" +
            """"rulesetFingerprint":"${esc(fingerprint)}",""" +
            """"gameId":"${esc(gameId)}","seed":$seed,""" +
            """"majorCivSlots":[$majorSlots],""" +
            """"nTiles":${gameInfo.tileMap.tileList.size},""" +
            """"caps":{"maxMajorCivs":${caps.maxMajorCivs},"maxCityStates":${caps.maxCityStates},""" +
            """"maxOwnCities":${caps.maxOwnCities},"maxVisOppCities":${caps.maxVisOppCities},""" +
            """"maxOwnUnits":${caps.maxOwnUnits},"maxVisOppUnits":${caps.maxVisOppUnits}},""" +
            """"spatialChannels":[$channels],""" +
            """"vocabCounts":$vocabCounts,""" +
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
 * One recorder per worker → one shard file. Writes one step per controlled civ-turn (obs + the
 * already-applied action labels), then one TERMINAL record per civ at game end carrying the
 * ±1/0 reward. NOT thread-shared.
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
    private val emitter = TrajectoryEmitter(File(config.outputDir ?: "."), baseName)
    private var opened = false
    private val seenCivs = HashSet<String>()
    private val civSlotById = LinkedHashMap<String, Int>()   // preserves recording order for terminal pass
    private var layout: List<Observation.Block> = emptyList() // captured at open for terminal zero-fill

    /** Emit one non-terminal step for [x] with the already-chosen-and-applied [actions]. */
    fun recordStep(x: Civilization, obs: Observation, actions: FloatArray, turn: Int) {
        val isFirst = seenCivs.add(x.civID)
        val civSlot = gameInfo.civilizations.indexOf(x)
        civSlotById[x.civID] = civSlot

        val blocks = obs.blocks + Observation.Block("actions", SampleSchema.DT_F32, BlockKind.FIXED, 0, actions)
        if (!opened) {
            layout = blocks
            val header = DataPlaneHooks.buildHeaderJson(gameInfo, fingerprint, gameId, seed, config.caps, blocks, vocab)
            emitter.open(header)
            writeSchemaSidecar(header)
            opened = true
        }
        emitter.record(framePayload(turn, civSlot, isFirst = isFirst, isLast = false, isTerminal = false,
            overflow = obs.overflow, reward = 0f, blocks = blocks))
    }

    /** Emit one terminal reward-carrier per recorded civ: isTerminal=1, reward=±1 (winner/loser) or
     *  0 (draw). Obs blocks are zero-filled (terminal obs is unused for training — dataset.py reads
     *  only the reward). No-op if no steps were recorded. */
    fun recordTerminal(winnerCivId: String?) {
        if (!opened) return
        val zero = zeroBlocks()
        val turn = gameInfo.turns
        for ((civId, civSlot) in civSlotById) {
            val reward = when {
                winnerCivId == null -> 0f
                civId == winnerCivId -> 1f
                else -> -1f
            }
            emitter.record(framePayload(turn, civSlot, isFirst = false, isLast = true, isTerminal = true,
                overflow = false, reward = reward, blocks = zero))
        }
    }

    private fun zeroBlocks(): List<Observation.Block> = layout.map { b ->
        val len = if (b.kind == BlockKind.VARIABLE) 0 else b.values.size
        Observation.Block(b.name, b.dtype, b.kind, b.perItem, FloatArray(len))
    }

    private fun framePayload(
        turn: Int, civSlot: Int, isFirst: Boolean, isLast: Boolean, isTerminal: Boolean,
        overflow: Boolean, reward: Float, blocks: List<Observation.Block>,
    ): ByteArray {
        val payload = LeBuffer(blocks.sumOf { it.values.size } + 64)
            .i32(turn).i32(civSlot)
            .u8(if (isFirst) 1 else 0).u8(if (isLast) 1 else 0).u8(if (isTerminal) 1 else 0)
            .u8(if (overflow) 1 else 0).f32(reward)
        for (b in blocks) Observation.writeBlock(payload, b)
        return payload.toByteArray()
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
