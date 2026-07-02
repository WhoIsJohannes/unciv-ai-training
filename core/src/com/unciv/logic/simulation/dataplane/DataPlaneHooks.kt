package com.unciv.logic.simulation.dataplane

import com.unciv.UncivGame
import com.unciv.logic.GameInfo
import com.unciv.logic.automation.civilization.NextTurnAutomation
import com.unciv.logic.city.City
import com.unciv.logic.civilization.Civilization
import com.unciv.models.ruleset.PerpetualConstruction
import com.unciv.models.ruleset.unique.GameContext
import com.unciv.utils.Log
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.ln
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

    /** v7.1 — true iff per-city construction control is ACTIVE for [civ] (a controlled civ in a game
     *  whose config has `controlConstruction`). `ConstructionAutomation.chooseNextConstruction` returns
     *  early for these civs, so the heuristic NEVER refills construction — a controlled city goes idle
     *  (PerpetualConstruction) when its current item COMPLETES, and the policy picks the next at the
     *  following civ-turn (commit-until-done cadence: ONE decision per construction, no per-turn churn). */
    fun controlsConstruction(civ: Civilization): Boolean =
        controls(civ) && games[civ.gameInfo]?.featurizer?.config?.controlConstruction == true

    private fun handleCivTurn(civ: Civilization, state: GameState) {
        val policy = installedPolicy ?: return
        if (!civ.isMajorCiv() || civ.isSpectator()) return
        val config = state.featurizer.config
        val obs = state.featurizer.observe(civ)
        val turn = civ.gameInfo.turns
        val d = chooseAndApply(civ, policy, state.vocab, config, obs, turn)
        state.recorder?.recordStep(civ, obs, d.actions, d.behaviorLogp, d.constructionActions, d.constructionLogp,
            d.constructionEcon, d.constructionCurrent, turn)
    }

    /** The per-civ-turn decision: fixed civ-head actions + per-head behavior logp ([SampleSchema.MASK_HEADS]
     *  order), PLUS the v7 per-city construction action / logp aligned to [Featurizer.orderedOwnCities]. */
    private class CivTurnDecision(
        val actions: FloatArray, val behaviorLogp: FloatArray,
        val constructionActions: FloatArray, val constructionLogp: FloatArray,
        val constructionEcon: FloatArray,   // v7.3: per-city raw log-economy, aligned to orderedOwnCities
        val constructionCurrent: FloatArray, // v7.4: per-city current construction mask idx (BC target; heuristic pick when control off)
    )

    /** v7.4: 0-indexed construction-mask idx of a city's currently-building item, or −1 if it's idle /
     *  on a PerpetualConstruction / not a building-or-unit. Inverse of [Vocab.constructionId]. */
    private fun constructionMaskIdx(vocab: Vocab, name: String): Float {
        if (name.isEmpty()) return -1f
        val b = vocab.building(name); if (b >= 0) return b.toFloat()
        val u = vocab.unit(name); if (u >= 0) return (vocab.buildingCount + u).toFloat()
        return -1f
    }

    /** v7.3: a city's raw log-economy `ln(1+max(0,prod)+max(0,food)+max(0,science))` (clamped ≥0 so ln is
     *  finite for a starving city). The trainer's per-city value baseline + per-city GAE credit each
     *  city's construction by its OWN economy return. */
    private fun perCityEcon(city: City): Float {
        val s = city.cityStats.currentCityStats
        val e = s.production.toDouble().coerceAtLeast(0.0) + s.food.toDouble().coerceAtLeast(0.0) +
            s.science.toDouble().coerceAtLeast(0.0)
        return ln(1.0 + e).toFloat()
    }

    /** Choose (and APPLY) the controlled actions. v1 controls tech + policy (greatPerson/diplomaticVote stay
     *  heuristic, recorded −1/0f). v7 ADDS per-city construction when [SampleConfig.controlConstruction]: for
     *  each city that would otherwise choose this turn (idle on a PerpetualConstruction), the policy picks a
     *  legal construction and we pre-fill `constructionQueue[0]` (the heuristic then skips it). Recorded
     *  construction action == applied, per city; cities not decided record −1/0f. Logp is recorded for v6
     *  off-policy replay (summed across heads + cities into the per-step old_logp). */
    private fun chooseAndApply(
        civ: Civilization, policy: PolicyProvider, vocab: Vocab, config: SampleConfig,
        obs: Observation, turn: Int,
    ): CivTurnDecision {
        val actions = FloatArray(SampleSchema.MASK_HEADS.size) { -1f }
        val behaviorLogp = FloatArray(SampleSchema.MASK_HEADS.size) { 0f }

        // TECH — decision only when no current research target (queue empty); pre-fill ⇒ heuristic respects it.
        if (civ.tech.techsToResearch.isEmpty()) {
            val mask = boolMask(obs, "mask_tech")
            val (idx, logp) = policy.chooseIndexWithLogp("tech", civ, mask, turn)
            actions[0] = idx.toFloat(); behaviorLogp[0] = logp
            if (idx >= 0) vocab.techId(idx)?.let { if (civ.tech.canBeResearched(it)) civ.tech.techsToResearch.add(it) }
        }

        // POLICY — decision only when a free policy slot is available; adopt the chosen policy.
        if (civ.policies.canAdoptPolicy()) {
            val mask = boolMask(obs, "mask_policy")
            val (idx, logp) = policy.chooseIndexWithLogp("policy", civ, mask, turn)
            actions[1] = idx.toFloat(); behaviorLogp[1] = logp
            if (idx >= 0) vocab.policyId(idx)?.let { name ->
                val pol = civ.gameInfo.ruleset.policies[name]
                if (pol != null && civ.policies.isAdoptable(pol)) civ.policies.adopt(pol)
            }
        }

        // CONSTRUCTION (v7) — per own city in the SAME orderedOwnCities order as obs `mask_construction`.
        val cities = Featurizer.orderedOwnCities(civ, config.caps.maxOwnCities)
        val cActions = FloatArray(cities.size) { -1f }
        val cLogp = FloatArray(cities.size) { 0f }
        val cEcon = FloatArray(cities.size) { perCityEcon(cities[it]) }   // v7.3: per-city economy, all cities
        // v7.4 BC target: each city's CURRENT construction (mask idx) BEFORE this turn's control runs → with
        // construction OFF this is the heuristic's standing pick; the supervised label for behavior-cloning.
        val cCurrent = FloatArray(cities.size) { constructionMaskIdx(vocab, cities[it].cityConstructions.currentConstructionName()) }
        if (config.controlConstruction) {
            val constrW = vocab.buildingCount + vocab.unitCount
            val maskFlat = obs.block("mask_construction")
            if (maskFlat.size == cities.size * constrW) {  // alignment guard (must always hold)
                cities.forEachIndexed { i, city ->
                    // v7.1 COMMIT-UNTIL-DONE cadence: decide ONLY when the city is idle (on a
                    // PerpetualConstruction). Because the heuristic chooseNextConstruction is disabled for
                    // controlled-construction civs (see ConstructionAutomation / [controlsConstruction]),
                    // the city goes idle exactly when its current item COMPLETES — so the policy picks the
                    // NEXT construction once per item, NOT every turn. This avoids the per-turn churn that
                    // made the construction logp/entropy summand dominate the joint PPO objective (~6
                    // decisions/step → instability + the whole policy fails to learn; v7 small-rung 47%→14%).
                    if (city.isPuppet) return@forEachIndexed
                    if (city.cityConstructions.getCurrentConstruction() !is PerpetualConstruction) return@forEachIndexed
                    val row = BooleanArray(constrW) { k -> maskFlat[i * constrW + k] != 0f }
                    if (row.none { it }) return@forEachIndexed   // no legal construction → stays idle (−1)
                    val (idx, logp) = policy.chooseConstructionWithLogp(civ, city, i, row, turn)
                    if (idx < 0) return@forEachIndexed
                    val name = vocab.constructionId(idx) ?: return@forEachIndexed   // PR4 null-guard
                    if (!city.cityConstructions.getConstruction(name).isBuildable(city.cityConstructions)) return@forEachIndexed
                    city.cityConstructions.setCurrentConstruction(name)   // commit queue[0]
                    cActions[i] = idx.toFloat(); cLogp[i] = logp
                }
            }
        }
        return CivTurnDecision(actions, behaviorLogp, cActions, cLogp, cEcon, cCurrent)
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
            """"policyBranches":${vocab.policyBranchCount},"promotions":${vocab.promotionCount},""" +
            """"nations":${vocab.nationCount}""" +
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

    /** Emit one non-terminal step for [x] with the already-chosen-and-applied [actions] and the
     *  per-head behavior log-prob [behaviorLogp] (v6 off-policy replay; same width as actions), PLUS the
     *  v7 per-city construction action / behavior logp ([constructionActions]/[constructionLogp],
     *  VARIABLE, one row per own city in `own_cities` order; −1/0f where the city did not decide). */
    fun recordStep(
        x: Civilization, obs: Observation, actions: FloatArray, behaviorLogp: FloatArray,
        constructionActions: FloatArray, constructionLogp: FloatArray, constructionEcon: FloatArray,
        constructionCurrent: FloatArray, turn: Int,
    ) {
        val isFirst = seenCivs.add(x.civID)
        val civSlot = gameInfo.civilizations.indexOf(x)
        civSlotById[x.civID] = civSlot

        val blocks = obs.blocks +
            Observation.Block("actions", SampleSchema.DT_F32, BlockKind.FIXED, 0, actions) +
            Observation.Block(SampleSchema.BLOCK_BEHAVIOR_LOGP, SampleSchema.DT_F32, BlockKind.FIXED, 0, behaviorLogp) +
            Observation.Block(SampleSchema.BLOCK_CONSTRUCTION_ACTION, SampleSchema.DT_F32, BlockKind.VARIABLE, 1, constructionActions) +
            Observation.Block(SampleSchema.BLOCK_CONSTRUCTION_LOGP, SampleSchema.DT_F32, BlockKind.VARIABLE, 1, constructionLogp) +
            Observation.Block(SampleSchema.BLOCK_ECON_CITY, SampleSchema.DT_F32, BlockKind.VARIABLE, 1, constructionEcon) +
            Observation.Block(SampleSchema.BLOCK_CONSTRUCTION_CURRENT, SampleSchema.DT_F32, BlockKind.VARIABLE, 1, constructionCurrent) +
            Observation.Block(SampleSchema.BLOCK_PHI, SampleSchema.DT_F32, BlockKind.FIXED, 0, floatArrayOf(economyPotential(x)))
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

    /** v7.2: the civ's log-stabilized ECONOMY POTENTIAL Φ(s) — ln(1+Σprod)+ln(1+Σfood)+ln(1+Σscience)
     *  over its cities + ln(1+#techs). The ln keeps Φ bounded so the potential-based shaping reward
     *  F = γ·Φ(s')−Φ(s) the trainer forms cannot drift-dominate the terminal ±1. Reads the cached
     *  per-city currentCityStats (good enough — PBRS is policy-invariant for any Φ that tracks strength). */
    private fun economyPotential(x: Civilization): Float {
        var prod = 0.0; var food = 0.0; var sci = 0.0
        for (c in x.cities) {
            val s = c.cityStats.currentCityStats
            prod += s.production.toDouble(); food += s.food.toDouble(); sci += s.science.toDouble()
        }
        // Clamp the yield sums to ≥0 before ln: a starving city has NEGATIVE food, so a net-negative sum
        // would make ln(1+Σ) = NaN → NaN shaping reward → non-finite loss → the divergence guard skips the
        // round. coerceAtLeast(0) keeps Φ finite (low = weak economy, the correct potential semantics).
        return (ln(1.0 + prod.coerceAtLeast(0.0)) + ln(1.0 + food.coerceAtLeast(0.0)) + ln(1.0 + sci.coerceAtLeast(0.0))
            + ln(1.0 + x.tech.getNumberOfTechsResearched())).toFloat()
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
