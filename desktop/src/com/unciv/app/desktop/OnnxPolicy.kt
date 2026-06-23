package com.unciv.app.desktop

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.unciv.logic.automation.unit.UnitAutomation
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.map.mapunit.MapUnit
import com.unciv.logic.simulation.dataplane.Featurizer
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

    init {
        val f = java.io.File(modelPath)
        require(f.isFile) { "OnnxPolicy: model not found at '$modelPath'" }
        val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(1) }
        session = env.createSession(f.absolutePath, opts)
        // PROVENANCE gate (criterion 6): refuse a model whose contract/schema/ruleset mismatches.
        val meta = session.metadata.customMetadata
        fun req(key: String) = meta[key] ?: error("OnnxPolicy: model missing ONNX metadata '$key'")
        val mSchema = req(SampleSchema.OnnxContract.META_SCHEMA_VERSION).toInt()
        val mContract = req(SampleSchema.OnnxContract.META_CONTRACT_VERSION).toInt()
        val mFingerprint = req(SampleSchema.OnnxContract.META_RULESET_FINGERPRINT)
        check(mSchema == expectedSchemaVersion) { "OnnxPolicy: model schema_version=$mSchema != live $expectedSchemaVersion (regenerate)" }
        check(mContract == SampleSchema.OnnxContract.CONTRACT_VERSION) { "OnnxPolicy: model contract_version=$mContract != ${SampleSchema.OnnxContract.CONTRACT_VERSION}" }
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

    /** Public single-input inference for the PARITY test (no masking/sampling). */
    fun infer(input: FloatArray): Pair<FloatArray, FloatArray> = forward(input)

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
            val input = obs.block("global") + obs.block("acting_civ")
            val (tech, policy) = forward(input)
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

    @Suppress("UNCHECKED_CAST")
    private fun row(t: OnnxTensor): FloatArray = (t.value as Array<FloatArray>)[0]

    override fun actUnit(unit: MapUnit) = UnitAutomation.automateUnitMoves(unit)

    override fun close() {
        try { session.close() } catch (_: Exception) {}
    }
}
