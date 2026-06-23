package com.unciv.logic.simulation.dataplane

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

/**
 * AC2 LEGALITY — the masked-choice invariant that OnnxPolicy relies on: `chooseIndex` must NEVER
 * return an illegal index. The selection logic ([MaskedChoice.choose]) is weight-independent, so we
 * fuzz it across random logits/masks (and both argmax + sample modes) rather than needing a model.
 * Mirrors the spirit of the RandomPolicy mask-parity legality checks in FairnessAndDeterminismTests.
 */
class OnnxPolicyLegalityTest {

    @Test
    fun choiceIsAlwaysLegalOrMinusOne() {
        val gen = Random(42)
        repeat(3000) { trial ->
            val n = 1 + gen.nextInt(90)
            val logits = FloatArray(n) { (gen.nextDouble() * 20 - 10).toFloat() }
            val mask = BooleanArray(n) { gen.nextBoolean() }
            val eval = gen.nextBoolean()
            val idx = MaskedChoice.choose(logits, mask, eval, Random(trial.toLong()))
            if (idx == -1) {
                assertFalse("returned -1 although legal actions existed", mask.any { it })
            } else {
                assertTrue("index $idx out of range [0,$n)", idx in 0 until n)
                assertTrue("index $idx is ILLEGAL", mask[idx])
            }
        }
    }

    @Test
    fun emptyLegalSetReturnsMinusOne() {
        val none = BooleanArray(5) { false }
        assertEquals(-1, MaskedChoice.choose(FloatArray(5) { it.toFloat() }, none, true, Random(1)))
        assertEquals(-1, MaskedChoice.choose(FloatArray(5) { it.toFloat() }, none, false, Random(1)))
    }

    @Test
    fun evalPicksHighestLegalLogit() {
        val logits = floatArrayOf(5f, 9f, 2f, 8f)            // global max is index 1 (9f) but it's ILLEGAL
        val mask = booleanArrayOf(false, false, true, true)  // legal: {2, 3}; among these 8f@3 wins
        assertEquals(3, MaskedChoice.choose(logits, mask, eval = true, rng = Random(0)))
    }

    @Test
    fun sampleStaysInLegalSupportUnderExtremeSkew() {
        val logits = floatArrayOf(100f, -100f, -100f)        // huge logit on the ILLEGAL index 0
        val mask = booleanArrayOf(false, true, true)
        repeat(1000) { trial ->
            val i = MaskedChoice.choose(logits, mask, eval = false, rng = Random(trial.toLong()))
            assertTrue("sampled illegal index $i", mask[i])
        }
    }
}
