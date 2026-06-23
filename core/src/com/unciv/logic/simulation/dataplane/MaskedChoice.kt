package com.unciv.logic.simulation.dataplane

import kotlin.math.exp
import kotlin.random.Random

/**
 * Pure masked action selection used by the in-JVM `OnnxPolicy`: argmax ([eval]) or softmax-SAMPLE
 * over the LEGAL support only. Returns a legal index, or −1 for an empty support — and NEVER an
 * illegal index. This is the AC2 legality invariant: it is weight-independent and unit-testable
 * without an ONNX session/model, and it mirrors the training-time masked log-prob (only legal
 * actions receive probability). Lives in `core` so the `tests` module (core-only) can verify it.
 */
object MaskedChoice {
    fun choose(logits: FloatArray, mask: BooleanArray, eval: Boolean, rng: Random): Int {
        val legal = mask.indices.filter { it < logits.size && mask[it] }
        if (legal.isEmpty()) return -1
        if (eval) return legal.maxByOrNull { logits[it] }!!
        val maxL = legal.maxOf { logits[it] }
        val exps = legal.map { exp((logits[it] - maxL).toDouble()) }
        val sum = exps.sum()
        var r = rng.nextDouble() * sum
        for (i in legal.indices) { r -= exps[i]; if (r <= 0.0) return legal[i] }
        return legal.last()
    }
}
