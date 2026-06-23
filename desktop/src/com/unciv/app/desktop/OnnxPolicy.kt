package com.unciv.app.desktop

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.unciv.logic.automation.unit.UnitAutomation
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.map.mapunit.MapUnit
import com.unciv.logic.simulation.dataplane.Featurizer
import com.unciv.logic.simulation.dataplane.Observation
import com.unciv.logic.simulation.dataplane.PolicyProvider
import com.unciv.logic.simulation.dataplane.SampleConfig
import com.unciv.logic.simulation.dataplane.SampleSchema
import com.unciv.logic.simulation.dataplane.Vocab
import java.nio.FloatBuffer
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
    /** True when the loaded model is contract v2 (rich multi-tensor input). */
    private val rich: Boolean

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
        rich = mContract == SampleSchema.OnnxContract.CONTRACT_VERSION_RICH
        check(mContract == SampleSchema.OnnxContract.CONTRACT_VERSION || rich) {
            "OnnxPolicy: model contract_version=$mContract not in {${SampleSchema.OnnxContract.CONTRACT_VERSION}, ${SampleSchema.OnnxContract.CONTRACT_VERSION_RICH}}"
        }
        check(mFingerprint == expectedRulesetFingerprint) { "OnnxPolicy: model ruleset_fingerprint mismatch (regenerate against the live ruleset)" }
    }

    /** One forward pass per (game, civ, turn), reused across the two head calls of that turn.
     *  ThreadLocal ⇒ no cross-thread / cross-game bleed when [com.unciv.logic.simulation.Simulation]
     *  runs games concurrently. */
    private class Memo(val key: String, val techLogits: FloatArray, val policyLogits: FloatArray)
    private val memo = ThreadLocal<Memo?>()

    /** Net decisions actually made (index ≥ 0). EVAL asserts this is > 0 to confirm the learner civ
     *  was really routed to the net (routing-held check, FND-0028). */
    private val decisions = java.util.concurrent.atomic.AtomicLong(0)
    fun decisionCount(): Long = decisions.get()

    /** Public single-input inference for the blind PARITY test (no masking/sampling). */
    fun infer(input: FloatArray): Pair<FloatArray, FloatArray> = forward(input)

    /** True when the loaded model uses the rich multi-tensor contract (v2). */
    fun isRich(): Boolean = rich

    override fun chooseIndex(head: String, civ: Civilization, legalMask: BooleanArray, turn: Int): Int {
        if (head !in SampleSchema.OnnxContract.MODELED_HEADS) return -1
        val logits = logitsFor(head, civ, turn)
        val idx = com.unciv.logic.simulation.dataplane.MaskedChoice.choose(logits, legalMask, eval, rngFor(civ, turn))
        if (idx >= 0) decisions.incrementAndGet()
        return idx
    }

    private fun logitsFor(head: String, civ: Civilization, turn: Int): FloatArray {
        val key = "${civ.gameInfo.gameId}|${civ.civID}|$turn"
        val cached = memo.get()
        val m = if (cached != null && cached.key == key) cached else run {
            val obs = Featurizer(civ.gameInfo, vocab, config).observe(civ)
            val (tech, policy) = if (rich) forwardRich(obs)
                                 else forward(obs.block("global") + obs.block("acting_civ"))
            Memo(key, tech, policy).also { memo.set(it) }
        }
        return if (head == "tech") m.techLogits else m.policyLogits
    }

    private fun forward(input: FloatArray): Pair<FloatArray, FloatArray> {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, input.size.toLong())).use { t ->
            session.run(mapOf(SampleSchema.OnnxContract.INPUT_NAME to t)).use { res ->
                val tech = row(res.get(SampleSchema.OnnxContract.OUTPUT_TECH).get() as OnnxTensor)
                val policy = row(res.get(SampleSchema.OnnxContract.OUTPUT_POLICY).get() as OnnxTensor)
                return tech to policy
            }
        }
    }

    /** Rich (contract v2) forward: build the SAME multi-tensor input the trainer pads (per-tile
     *  spatial token set + per-type entity token sets, each with a presence mask), feed onnxruntime,
     *  read the policy logits. EVERY created [OnnxTensor] is closed (no native leak — council R7). */
    private fun forwardRich(obs: Observation): Pair<FloatArray, FloatArray> {
        val inputs = buildRichTensors(env, obs)
        try {
            session.run(inputs).use { res ->
                val tech = row(res.get(SampleSchema.OnnxContract.OUTPUT_TECH).get() as OnnxTensor)
                val policy = row(res.get(SampleSchema.OnnxContract.OUTPUT_POLICY).get() as OnnxTensor)
                return tech to policy
            }
        } finally {
            for (t in inputs.values) try { t.close() } catch (_: Exception) {}
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun row(t: OnnxTensor): FloatArray = (t.value as Array<FloatArray>)[0]

    companion object {
        /** Build the contract-v2 multi-tensor input from a live [Observation]. Shared by inference
         *  and the parity harness so JVM tensor construction is identical in both. Caller closes the
         *  returned tensors. u8 blocks (spatial) are fed as float32 (matches the training dtype). */
        fun buildRichTensors(env: OrtEnvironment, obs: Observation): LinkedHashMap<String, OnnxTensor> {
            val tokens = SampleSchema.OnnxContract.RICH_TOKEN_NAMES.map { name ->
                val width = if (name == "spatial") SampleSchema.NUM_SPATIAL_CHANNELS
                            else obs.blocks.first { it.name == name }.perItem
                Triple(name, obs.block(name), width)
            }
            return richTensorsFromArrays(env, obs.block("global"), obs.block("acting_civ"), tokens)
        }

        /** Single source of truth for v2 tensor construction (used by live inference AND the parity
         *  harness, so JVM↔Python parity tests exactly what runs in play). `tokens` = ordered
         *  (name, flat-values, perItem-width). */
        fun richTensorsFromArrays(
            env: OrtEnvironment, global: FloatArray, acting: FloatArray,
            tokens: List<Triple<String, FloatArray, Int>>,
        ): LinkedHashMap<String, OnnxTensor> {
            val out = LinkedHashMap<String, OnnxTensor>()
            out[SampleSchema.OnnxContract.INPUT_GLOBAL] = vecTensor(env, global)
            out[SampleSchema.OnnxContract.INPUT_ACTING] = vecTensor(env, acting)
            for ((name, values, width) in tokens) {
                val (tok, mask) = tokenTensors(env, values, width)
                out[name] = tok
                out[name + SampleSchema.OnnxContract.MASK_SUFFIX] = mask
            }
            return out
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
        try { session.close() } catch (_: Exception) {}
    }
}
