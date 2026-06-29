package com.unciv.logic.simulation.dataplane

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

/**
 * v6 — [MaskedChoice.chooseWithLogp] returns the chosen legal index AND its log-prob over the SAME
 * legal-masked softmax that `choose` samples from: logp == ln(softmax_over_legal)[chosen] ==
 * log_softmax(masked logits)[chosen] (the exact quantity the Python trainer's `_masked_logp` scores).
 *
 * Two invariants this pins:
 *  - VALUE: the returned logp equals the masked-softmax log-prob of the chosen index (within fp), and
 *    empty support / no decision → (-1, 0f).
 *  - SINGLE SOURCE / SINGLE DRAW: `choose` is exactly `chooseWithLogp(...).first` — same index for the
 *    same RNG seed, i.e. routing through chooseWithLogp does NOT perturb the rng.nextDouble() stream
 *    (generation replay must stay byte-identical).
 */
class MaskedChoiceLogpTest {

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
    fun logpMatchesMaskedSoftmaxOfChosen() {
        val gen = Random(7)
        repeat(3000) { trial ->
            val n = 1 + gen.nextInt(60)
            val logits = FloatArray(n) { (gen.nextDouble() * 20 - 10).toFloat() }
            val mask = BooleanArray(n) { gen.nextBoolean() }
            val rng = Random(trial.toLong())
            val (idx, logp) = MaskedChoice.chooseWithLogp(logits, mask, eval = false, rng = rng)
            if (!mask.any { it }) {
                assertEquals(-1, idx); assertEquals(0f, logp)
            } else {
                assertTrue("index $idx is ILLEGAL", mask[idx])
                val ref = refLogp(logits, mask, idx)
                assertTrue("logp $logp != masked-softmax ref $ref",
                    abs(logp.toDouble() - ref) < 1e-4)
                assertTrue("logp must be <= 0 (log of a probability)", logp <= 1e-6f)
            }
        }
    }

    @Test
    fun emptySupportReturnsMinusOneZero() {
        val none = BooleanArray(5) { false }
        val (i1, lp1) = MaskedChoice.chooseWithLogp(FloatArray(5) { it.toFloat() }, none, true, Random(1))
        val (i2, lp2) = MaskedChoice.chooseWithLogp(FloatArray(5) { it.toFloat() }, none, false, Random(1))
        assertEquals(-1, i1); assertEquals(0f, lp1)
        assertEquals(-1, i2); assertEquals(0f, lp2)
    }

    @Test
    fun chooseDelegatesToChooseWithLogpFirstSameDraw() {
        // Same logits/mask/seed must yield the same index from both entry points (one rng draw,
        // identical order) — the determinism / byte-identical-replay invariant.
        val gen = Random(99)
        repeat(2000) { trial ->
            val n = 1 + gen.nextInt(40)
            val logits = FloatArray(n) { (gen.nextDouble() * 20 - 10).toFloat() }
            val mask = BooleanArray(n) { gen.nextBoolean() }
            val a = MaskedChoice.choose(logits, mask, eval = false, rng = Random(trial.toLong()))
            val b = MaskedChoice.chooseWithLogp(logits, mask, eval = false, rng = Random(trial.toLong())).first
            assertEquals("choose != chooseWithLogp.first for trial $trial", a, b)
        }
    }
}
