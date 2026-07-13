package com.unciv.app.desktop

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.unciv.logic.automation.Timers
import com.unciv.logic.automation.unit.UnitAutomation
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.map.TileMap
import com.unciv.logic.map.mapunit.MapUnit
import com.unciv.logic.simulation.dataplane.Featurizer
import com.unciv.logic.simulation.dataplane.Observation
import com.unciv.logic.simulation.dataplane.PolicyProvider
import com.unciv.logic.simulation.dataplane.SampleConfig
import com.unciv.logic.simulation.dataplane.SampleSchema
import com.unciv.logic.simulation.dataplane.Vocab
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.random.Random

/**
 * [PolicyProvider] that runs a trained policy net INSIDE the JVM via onnxruntime to choose TECH and
 * POLICY from the legal mask. The net input is `concat(global, acting_civ)` built by the SAME
 * [Featurizer] the emitter uses (the one observation code-path — golden-tested for cross-boundary
 * parity). One shared read-only [OrtSession] is used by all worker threads (ORT `run` is
 * thread-safe; intra-op threads pinned to 1 to avoid oversubscribing the simulation workers).
 *
 * Heads other than tech/policy → −1 (heuristic fallback, like RandomPolicy). Empty legal set → −1.
 * `actUnit` delegates to `UnitAutomation.automateUnitMoves`, unchanged.
 *
 * @param eval deterministic mode: argmax over the legal-masked logits. Otherwise SAMPLE from the
 *   legal-masked softmax via the per-(civ,turn) RNG so generation runs replay.
 */
class OnnxPolicy(
    modelPath: String,
    private val vocab: Vocab,
    private val config: SampleConfig,
    private val rngFor: (Civilization, Int) -> Random,
    private val eval: Boolean,
    expectedSchemaVersion: Int,
    expectedRulesetFingerprint: String,
) : PolicyProvider, AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    /** True when the loaded model uses the multi-tensor input (contract v2 rich OR v3 structured). */
    private val rich: Boolean
    /** True when the loaded model is contract v3 (structured: rich + hex-GNN neighbor inputs). */
    private val structured: Boolean
    /** v7: true when the model exposes the per-city `construction_logits` output. */
    private val hasConstructionOutput: Boolean
    /** v7: the construction-mask / per-city-head width (buildings+units), for the PR3 dim cross-check. */
    private val constrW: Int = vocab.buildingCount + vocab.unitCount
    /** v8: true when the model exposes the per-unit `unit_intent_logits` output. */
    private val hasUnitIntentOutput: Boolean
    /** v8: the per-unit intent-head width ([Vocab.unitIntentCount] = UnitIntent.COUNT), dim cross-check. */
    private val intentW: Int = vocab.unitIntentCount

    init {
        val f = java.io.File(modelPath)
        require(f.isFile) { "OnnxPolicy: model not found at '$modelPath'" }
        val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(1) }
        session = env.createSession(f.absolutePath, opts)
        // PROVENANCE gate (criterion 6): refuse a model whose contract/schema/ruleset mismatches.
        // Accepts contract v1 (blind) OR v2 (rich) — the build path is selected by `rich`.
        val meta = session.metadata.customMetadata
        fun req(key: String) = meta[key] ?: error("OnnxPolicy: model missing ONNX metadata '$key'")
        val mSchema = req(SampleSchema.OnnxContract.META_SCHEMA_VERSION).toInt()
        val mContract = req(SampleSchema.OnnxContract.META_CONTRACT_VERSION).toInt()
        val mFingerprint = req(SampleSchema.OnnxContract.META_RULESET_FINGERPRINT)
        check(mSchema == expectedSchemaVersion) { "OnnxPolicy: model schema_version=$mSchema != live $expectedSchemaVersion (regenerate)" }
        structured = mContract == SampleSchema.OnnxContract.CONTRACT_VERSION_STRUCTURED
        rich = structured || mContract == SampleSchema.OnnxContract.CONTRACT_VERSION_RICH
        check(mContract == SampleSchema.OnnxContract.CONTRACT_VERSION || rich) {
            "OnnxPolicy: model contract_version=$mContract not in {1, 2, 3}"
        }
        check(mFingerprint == expectedRulesetFingerprint) { "OnnxPolicy: model ruleset_fingerprint mismatch (regenerate against the live ruleset)" }
        // PROVENANCE (council): a multi-tensor model must expose the full input inventory; a v3
        // structured model ALSO requires the hex-GNN neighbor inputs — a malformed export that drops
        // them fails LOUD here, not with an opaque runtime missing-input at the first decision.
        if (rich) {
            val want = mutableListOf(SampleSchema.OnnxContract.INPUT_GLOBAL, SampleSchema.OnnxContract.INPUT_ACTING)
            for (n in SampleSchema.OnnxContract.RICH_TOKEN_NAMES) {
                want += n; want += n + SampleSchema.OnnxContract.MASK_SUFFIX
            }
            if (structured) want += SampleSchema.OnnxContract.NEIGHBOR_INPUT_NAMES
            val have = session.inputNames
            val missing = want.filter { it !in have }
            check(missing.isEmpty()) { "OnnxPolicy: model missing expected inputs $missing (model has $have)" }
        }
        // v7: discover the per-city construction output. PR3/PR2 — if construction control is requested
        // but the model lacks the head, FAIL LOUD at load (an "ON" arm silently falling back to the
        // heuristic would contaminate the ON-vs-OFF comparison).
        hasConstructionOutput = SampleSchema.OnnxContract.OUTPUT_CONSTRUCTION in session.outputNames
        check(!config.controlConstruction || hasConstructionOutput) {
            "OnnxPolicy: controlConstruction=ON but model lacks output '${SampleSchema.OnnxContract.OUTPUT_CONSTRUCTION}' " +
                "(export the net with the v7 per-city construction head)"
        }
        // v8: discover the per-unit intent output — same fail-loud discipline as construction (an "ON" arm
        // silently falling back to the heuristic would contaminate the ON-vs-OFF comparison).
        hasUnitIntentOutput = SampleSchema.OnnxContract.OUTPUT_UNIT_INTENT in session.outputNames
        check(!config.controlUnitIntent || hasUnitIntentOutput) {
            "OnnxPolicy: controlUnitIntent=ON but model lacks output '${SampleSchema.OnnxContract.OUTPUT_UNIT_INTENT}' " +
                "(export the net with the v8 per-unit intent head)"
        }
    }

    /** One forward pass per (game, civ, turn), reused across the two head calls of that turn.
     *  ThreadLocal ⇒ no cross-thread / cross-game bleed when [com.unciv.logic.simulation.Simulation]
     *  runs games concurrently. */
    private class Memo(
        val key: String, val techLogits: FloatArray, val policyLogits: FloatArray,
        /** v7: per-city construction logits [ncities][constrW], or null if the model has no construction head. */
        val constructionLogits: Array<FloatArray>?,
        /** v8: per-unit intent logits [nunits][intentW], or null if the model has no unit-intent head. */
        val unitIntentLogits: Array<FloatArray>?,
    )
    private val memo = ThreadLocal<Memo?>()

    /** perf — the observation [DataPlaneHooks.handleCivTurn] just built for this (game, civ, turn),
     *  adopted by [forwardCached] instead of re-featurizing an identical one (observe() ran twice
     *  per controlled civ-turn). ThreadLocal: games run one-per-thread, like [memo]. */
    private val providedObs = ThreadLocal<Pair<String, Observation>?>()

    override fun provideObservation(civ: Civilization, turn: Int, obs: Observation) {
        providedObs.set(memoKey(civ, turn) to obs)
    }

    private fun memoKey(civ: Civilization, turn: Int) = "${civ.gameInfo.gameId}|${civ.civID}|$turn"

    /** Net decisions actually made (index ≥ 0). EVAL asserts this is > 0 to confirm the learner civ
     *  was really routed to the net (routing-held check, FND-0028). */
    private val decisions = java.util.concurrent.atomic.AtomicLong(0)
    fun decisionCount(): Long = decisions.get()

    /** v7 (PR2): per-city construction fallbacks (empty support / out-of-range row). A healthy ON arm
     *  reports ≈0; a large count means the net rarely controlled construction (comparison contaminated). */
    private val constructionFallbacks = java.util.concurrent.atomic.AtomicLong(0)
    fun constructionFallbackCount(): Long = constructionFallbacks.get()

    /** v8: per-unit intent fallbacks (missing head / out-of-range row). A healthy ON arm reports ≈0. */
    private val unitIntentFallbacks = java.util.concurrent.atomic.AtomicLong(0)
    fun unitIntentFallbackCount(): Long = unitIntentFallbacks.get()

    /** Public single-input inference for the blind PARITY test (no masking/sampling). */
    fun infer(input: FloatArray): Pair<FloatArray, FloatArray> = forward("parity", input).let { it.techLogits to it.policyLogits }

    /** True when the loaded model uses the rich multi-tensor contract (v2). */
    fun isRich(): Boolean = rich

    override fun chooseIndex(head: String, civ: Civilization, legalMask: BooleanArray, turn: Int): Int =
        chooseIndexWithLogp(head, civ, legalMask, turn).first

    /** v6 — net action + its masked-softmax behavior log-prob (recorded for off-policy replay).
     *  Reuses the per-(game,civ,turn) memoized logits and the SAME single rng draw as [chooseIndex]. */
    override fun chooseIndexWithLogp(head: String, civ: Civilization, legalMask: BooleanArray, turn: Int): Pair<Int, Float> {
        if (head !in SampleSchema.OnnxContract.MODELED_HEADS) return -1 to 0f
        val logits = logitsFor(head, civ, turn)
        val (idx, logp) = com.unciv.logic.simulation.dataplane.MaskedChoice.chooseWithLogp(logits, legalMask, eval, rngFor(civ, turn))
        if (idx >= 0) decisions.incrementAndGet()
        return idx to logp
    }

    /** v7 — per-city construction from the SAME memoized forward (ONE inference per civ-turn). Index the
     *  net's per-city construction logits at [cityRow] (the city's position in [Featurizer.orderedOwnCities],
     *  which is the own_cities token order), then masked-softmax-sample exactly like the civ heads. Missing
     *  output / out-of-range row → abstain (−1) and count a fallback (PR2). */
    override fun chooseConstructionWithLogp(
        civ: Civilization, city: com.unciv.logic.city.City, cityRow: Int, legalMask: BooleanArray, turn: Int,
    ): Pair<Int, Float> {
        val cons = forwardCached(civ, turn).constructionLogits
        if (cons == null || cityRow < 0 || cityRow >= cons.size) { constructionFallbacks.incrementAndGet(); return -1 to 0f }
        val (idx, logp) = com.unciv.logic.simulation.dataplane.MaskedChoice.chooseWithLogp(cons[cityRow], legalMask, eval, rngFor(civ, turn))
        if (idx >= 0) decisions.incrementAndGet() else constructionFallbacks.incrementAndGet()
        return idx to logp
    }

    /** v8 — per-unit intent from the SAME memoized forward (ONE inference per civ-turn). Index the net's
     *  per-unit intent logits at [unitRow] (the unit's position in [Featurizer.orderedOwnUnits] = the
     *  own_units token order), masked-softmax-SAMPLE for the dispatched intent (single rng draw, same stream
     *  as the other heads), and return the full masked log-softmax VECTOR so the recorder can score
     *  `log π_b(realized)` even when dispatch falls back to a different rung. Missing head / out-of-range row
     *  → abstain (−1) + a fallback (the unit stays heuristic). */
    override fun chooseUnitIntentWithLogp(
        civ: Civilization, unit: MapUnit, unitRow: Int, legalMask: BooleanArray, turn: Int,
    ): Pair<Int, FloatArray> {
        val logits = forwardCached(civ, turn).unitIntentLogits
        if (logits == null || unitRow < 0 || unitRow >= logits.size) { unitIntentFallbacks.incrementAndGet(); return -1 to FloatArray(0) }
        val row = logits[unitRow]
        val (idx, _) = com.unciv.logic.simulation.dataplane.MaskedChoice.chooseWithLogp(row, legalMask, eval, rngFor(civ, turn))
        if (idx < 0) { unitIntentFallbacks.incrementAndGet(); return -1 to FloatArray(0) }
        decisions.incrementAndGet()
        return idx to com.unciv.logic.simulation.dataplane.MaskedChoice.maskedLogSoftmax(row, legalMask)
    }

    /** The ONE memoized forward per (game, civ, turn), reused by tech/policy/construction calls. */
    private fun forwardCached(civ: Civilization, turn: Int): Memo {
        val key = memoKey(civ, turn)
        val cached = memo.get()
        if (cached != null && cached.key == key) return cached
        // perf: adopt the hooks-provided observation for this exact key (identical Featurizer inputs
        // — the same game, civ, turn, vocab, config); re-featurize only on a cold/foreign key.
        val provided = providedObs.get()
        val obs = if (provided != null && provided.first == key) provided.second
                  else Featurizer(civ.gameInfo, vocab, config).observe(civ)
        val m = if (rich) forwardRich(key, obs, if (structured) civ.gameInfo.tileMap else null)
                else forward(key, obs.block("global") + obs.block("acting_civ"))
        memo.set(m)
        return m
    }

    private fun logitsFor(head: String, civ: Civilization, turn: Int): FloatArray {
        val m = forwardCached(civ, turn)
        return if (head == "tech") m.techLogits else m.policyLogits
    }

    private fun forward(key: String, input: FloatArray): Memo {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, input.size.toLong())).use { t ->
            session.run(mapOf(SampleSchema.OnnxContract.INPUT_NAME to t)).use { res ->
                val tech = row(res.get(SampleSchema.OnnxContract.OUTPUT_TECH).get() as OnnxTensor)
                val policy = row(res.get(SampleSchema.OnnxContract.OUTPUT_POLICY).get() as OnnxTensor)
                return Memo(key, tech, policy, null, null)   // blind models never expose construction / unit-intent
            }
        }
    }

    /** perf — hex adjacency is static per map: cache the computed (index, mask) arrays per TileMap
     *  and only wrap them into fresh OnnxTensors per forward (createTensor copies the buffer; the
     *  caller still closes the tensors). WeakHashMap so entries die with their game; synchronized —
     *  the policy is shared by all worker threads (a lost race just recomputes a pure function). */
    private val neighborArrays: MutableMap<TileMap, Pair<LongArray, FloatArray>> =
        java.util.Collections.synchronizedMap(java.util.WeakHashMap())

    private fun neighborTensorsCached(tileMap: TileMap): Pair<OnnxTensor, OnnxTensor> {
        val (idx, mask) = neighborArrays.getOrPut(tileMap) { buildNeighborArrays(tileMap) }
        val n = tileMap.tileList.size.toLong()
        val deg = SampleSchema.OnnxContract.HEX_DEGREE.toLong()
        val idxT = OnnxTensor.createTensor(env, LongBuffer.wrap(idx), longArrayOf(1, n, deg))
        val mskT = OnnxTensor.createTensor(env, FloatBuffer.wrap(mask), longArrayOf(1, n, deg))
        return idxT to mskT
    }

    /** Rich (contract v2/v3) forward: build the SAME multi-tensor input the trainer pads, feed
     *  onnxruntime, read tech/policy (+ v7 per-city construction when present). EVERY created
     *  [OnnxTensor] is closed (no native leak — council R7). */
    private fun forwardRich(key: String, obs: Observation, tileMap: TileMap?): Memo {
        val inputs = buildRichTensors(env, obs)
        if (tileMap != null) {  // v3 structured: add the hex-GNN neighbor inputs from the LIVE map
            val (idx, mask) = neighborTensorsCached(tileMap)
            inputs[SampleSchema.OnnxContract.INPUT_NEIGHBOR_INDEX] = idx
            inputs[SampleSchema.OnnxContract.INPUT_NEIGHBOR_MASK] = mask
        }
        try {
            return Timers.timeThis("onnxForward") {   // D8: ms/decision span (Log-gated, zero-overhead off)
                session.run(inputs).use { res ->
                    val tech = row(res.get(SampleSchema.OnnxContract.OUTPUT_TECH).get() as OnnxTensor)
                    val policy = row(res.get(SampleSchema.OnnxContract.OUTPUT_POLICY).get() as OnnxTensor)
                    val construction = if (hasConstructionOutput)
                        rows2d(res.get(SampleSchema.OnnxContract.OUTPUT_CONSTRUCTION).get() as OnnxTensor) else null
                    if (construction != null && construction.isNotEmpty() && construction[0].size != constrW)
                        error("OnnxPolicy: construction_logits width=${construction[0].size} != live constrW=$constrW (regenerate vs the live ruleset)")  // PR3
                    val unitIntent = if (hasUnitIntentOutput)
                        rows2d(res.get(SampleSchema.OnnxContract.OUTPUT_UNIT_INTENT).get() as OnnxTensor) else null
                    if (unitIntent != null && unitIntent.isNotEmpty() && unitIntent[0].size != intentW)
                        error("OnnxPolicy: unit_intent_logits width=${unitIntent[0].size} != live intentW=$intentW (regenerate vs the live intent enum)")
                    Memo(key, tech, policy, construction, unitIntent)
                }
            }
        } finally {
            for (t in inputs.values) try { t.close() } catch (_: Exception) {}
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun row(t: OnnxTensor): FloatArray = (t.value as Array<FloatArray>)[0]

    /** v7: a [1, N, W] tensor → its [N][W] rows (the per-city construction logits). */
    @Suppress("UNCHECKED_CAST")
    private fun rows2d(t: OnnxTensor): Array<FloatArray> = (t.value as Array<Array<FloatArray>>)[0]

    companion object {
        /** Build the contract-v2 multi-tensor input from a live [Observation]. Shared by inference
         *  and the parity harness so JVM tensor construction is identical in both. Caller closes the
         *  returned tensors. u8 blocks (spatial) are fed as float32 (matches the training dtype). */
        // Fallback per-token widths if a block is absent from the Observation (a real shard always
        // emits all six, even empty ones — this just avoids a decision-time crash if a future schema
        // change drops one; the empty token set is handled by the masked pool).
        private val FALLBACK_WIDTH = mapOf("own_units" to 9, "opp_units" to 9,
                                           "own_cities" to 17, "opp_cities" to 17, "civ_tokens" to 84)

        /** v3 clock-direction offsets (dirs 12,2,4,6,8,10) — MUST match Python hexgraph.OFFSETS.
         *  internal so the adjacency-dump fidelity harness reuses the SAME offsets (no drift). */
        internal val HEX_OFFSETS = arrayOf(
            intArrayOf(1, 1), intArrayOf(0, 1), intArrayOf(-1, 0),
            intArrayOf(-1, -1), intArrayOf(0, -1), intArrayOf(1, 0),
        )

        /** The raw hex-GNN adjacency arrays for [buildNeighborTensorsLive]/the per-map cache — a pure
         *  function of the (frozen-after-load) map geometry. */
        fun buildNeighborArrays(tileMap: TileMap): Pair<LongArray, FloatArray> {
            val tiles = tileMap.tileList
            val n = tiles.size
            val deg = SampleSchema.OnnxContract.HEX_DEGREE
            val idx = LongArray(n * deg) { n.toLong() }   // sentinel = N
            val mask = FloatArray(n * deg)
            for (tile in tiles) {
                val r = tile.zeroBasedIndex
                if (r < 0 || r >= n) continue
                val x = tile.position.x.toInt(); val y = tile.position.y.toInt()
                for (d in 0 until deg) {
                    val nb = tileMap.getIfTileExistsOrNull(x + HEX_OFFSETS[d][0], y + HEX_OFFSETS[d][1])
                    val nr = nb?.zeroBasedIndex ?: -1
                    if (nr in 0 until n) { idx[r * deg + d] = nr.toLong(); mask[r * deg + d] = 1f }
                }
            }
            return idx to mask
        }

        /** Build the live hex-GNN adjacency [1,N,6] int64 + [1,N,6] f32 from the real [TileMap] using
         *  the engine's own world-wrap-correct [TileMap.getIfTileExistsOrNull]. Rows are in
         *  zeroBasedIndex order (matches the spatial token set); missing neighbor → sentinel index N
         *  (the model's zero pad row) + mask 0. Caller closes the returned tensors. */
        fun buildNeighborTensorsLive(env: OrtEnvironment, tileMap: TileMap): Pair<OnnxTensor, OnnxTensor> {
            val n = tileMap.tileList.size
            val deg = SampleSchema.OnnxContract.HEX_DEGREE
            val (idx, mask) = buildNeighborArrays(tileMap)
            val idxT = OnnxTensor.createTensor(env, LongBuffer.wrap(idx), longArrayOf(1, n.toLong(), deg.toLong()))
            val mskT = OnnxTensor.createTensor(env, FloatBuffer.wrap(mask), longArrayOf(1, n.toLong(), deg.toLong()))
            return idxT to mskT
        }

        fun buildRichTensors(env: OrtEnvironment, obs: Observation): LinkedHashMap<String, OnnxTensor> {
            val tokens = SampleSchema.OnnxContract.RICH_TOKEN_NAMES.map { name ->
                val blk = obs.blocks.firstOrNull { it.name == name }   // defensive: absent → empty set
                val width = when {
                    name == "spatial" -> SampleSchema.NUM_SPATIAL_CHANNELS
                    blk != null && blk.perItem > 0 -> blk.perItem
                    else -> FALLBACK_WIDTH.getValue(name)
                }
                Triple(name, blk?.values ?: FloatArray(0), width)
            }
            val g = obs.blocks.firstOrNull { it.name == "global" }?.values ?: FloatArray(0)
            val a = obs.blocks.firstOrNull { it.name == "acting_civ" }?.values ?: FloatArray(0)
            return richTensorsFromArrays(env, g, a, tokens)
        }

        /** Single source of truth for v2 tensor construction (used by live inference AND the parity
         *  harness, so JVM↔Python parity tests exactly what runs in play). `tokens` = ordered
         *  (name, flat-values, perItem-width). */
        fun richTensorsFromArrays(
            env: OrtEnvironment, global: FloatArray, acting: FloatArray,
            tokens: List<Triple<String, FloatArray, Int>>,
        ): LinkedHashMap<String, OnnxTensor> {
            val out = LinkedHashMap<String, OnnxTensor>()
            try {  // leak-safe (council 🔴): close any partially-built tensors if createTensor throws
                out[SampleSchema.OnnxContract.INPUT_GLOBAL] = vecTensor(env, global)
                out[SampleSchema.OnnxContract.INPUT_ACTING] = vecTensor(env, acting)
                for ((name, values, width) in tokens) {
                    val (tok, mask) = tokenTensors(env, values, width)
                    out[name] = tok
                    out[name + SampleSchema.OnnxContract.MASK_SUFFIX] = mask
                }
                return out
            } catch (e: Throwable) {
                for (t in out.values) try { t.close() } catch (_: Exception) {}
                throw e
            }
        }

        private fun vecTensor(env: OrtEnvironment, v: FloatArray): OnnxTensor =
            OnnxTensor.createTensor(env, FloatBuffer.wrap(v), longArrayOf(1, v.size.toLong()))

        /** [1, N, width] token tensor + [1, N] presence mask. Empty set ⇒ N=1 zero-token, mask=[0]
         *  (so the model's masked pool sees N≥1 — matches the trainer's pad-to-max(1,count)). */
        private fun tokenTensors(env: OrtEnvironment, values: FloatArray, width: Int): Pair<OnnxTensor, OnnxTensor> {
            val count = if (width > 0) values.size / width else 0
            val n = maxOf(1, count)
            val data = if (count > 0) values else FloatArray(width)
            val mask = FloatArray(n) { if (it < count) 1f else 0f }
            val tok = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), longArrayOf(1, n.toLong(), width.toLong()))
            val msk = OnnxTensor.createTensor(env, FloatBuffer.wrap(mask), longArrayOf(1, n.toLong()))
            return tok to msk
        }
    }

    override fun actUnit(unit: MapUnit) = UnitAutomation.automateUnitMoves(unit)

    override fun close() {
        val fb = constructionFallbacks.get()
        if (config.controlConstruction && fb > 0)
            com.unciv.utils.Log.error("WARN: OnnxPolicy construction fallbacks=%d (ON arm should be ~0; net rarely controlled construction)", fb)
        val ufb = unitIntentFallbacks.get()
        if (config.controlUnitIntent && ufb > 0)
            com.unciv.utils.Log.error("WARN: OnnxPolicy unit-intent fallbacks=%d (ON arm should be ~0; net rarely controlled unit intent)", ufb)
        try { session.close() } catch (_: Exception) {}
    }
}
