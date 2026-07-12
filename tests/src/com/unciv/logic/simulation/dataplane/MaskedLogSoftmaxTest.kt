package com.unciv.logic.simulation.dataplane

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

/**
 * v8 — [MaskedChoice.maskedLogSoftmax] returns the FULL masked log-softmax vector `log π(k)`: for every
 * LEGAL k, result[k] == log_softmax(masked logits)[k] (== the Python trainer's per-action `log π`), every
 * ILLEGAL k is exactly 0f, and an empty support → an all-zero vector. Its result[chosen] must equal
 * [MaskedChoice.chooseWithLogp]'s returned logp for the SAME chosen index (single numerically-stable form
 * ⇒ recorded `log π_b(realized)` is consistent whether the executed intent is the sampled one or a
 * dispatch fallback). Mirrors [MaskedChoiceLogpTest]'s `refLogp`-based fuzz style.
 */
class MaskedLogSoftmaxTest {

    /** Reference: log of the legal-masked softmax probability of [chosen] (independent reimpl). */
    private fun refLogp(logits: FloatArray, mask: BooleanArray, chosen: Int): Double {
        val legal = mask.indices.filter { it < logits.size && mask[it] }
        val maxL = legal.maxOf { logits[it].toDouble() }
        val exps = legal.map { exp(logits[it].toDouble() - maxL) }
        val sum = exps.sum()
        val pos = legal.indexOf(chosen)
        return ln(exps[pos] / sum)
    }

    @Test
    fun maskedLogSoftmaxMatchesChooseWithLogpAndZeroesIllegal() {
        val gen = Random(23)
        repeat(1000) { trial ->
            val n = 1 + gen.nextInt(60)
            val logits = FloatArray(n) { (gen.nextDouble() * 20 - 10).toFloat() }
            val mask = BooleanArray(n) { gen.nextBoolean() }
            val ls = MaskedChoice.maskedLogSoftmax(logits, mask)
            assertEquals("maskedLogSoftmax width must equal the mask width", n, ls.size)

            if (!mask.any { it }) {
                // (c) empty support → all-zero array.
                assertTrue("empty-support mask must yield an all-zero log-softmax", ls.all { it == 0f })
                return@repeat
            }

            // (a) every LEGAL k equals the reference masked log-softmax; (b) every ILLEGAL k is exactly 0f.
            for (k in 0 until n) {
                if (mask[k]) {
                    val ref = refLogp(logits, mask, k)
                    assertTrue("legal logSoftmax[$k]=${ls[k]} != masked-softmax ref $ref",
                        abs(ls[k].toDouble() - ref) < 1e-4)
                    assertTrue("a legal log-prob must be <= 0", ls[k] <= 1e-6f)
                } else {
                    assertEquals("illegal index $k must be exactly 0f", 0f, ls[k])
                }
            }

            // result[chosen] must equal chooseWithLogp(...).second for the SAME chosen (within fp).
            val (chosen, logp) = MaskedChoice.chooseWithLogp(logits, mask, eval = false, rng = Random(trial.toLong()))
            assertTrue("chooseWithLogp returned an ILLEGAL index $chosen", mask[chosen])
            assertEquals("maskedLogSoftmax[chosen] must match chooseWithLogp's logp",
                logp.toDouble(), ls[chosen].toDouble(), 1e-4)
        }
    }
}
