package com.unciv.logic.simulation

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Regression guard for the binomial win-rate p-value extracted from Simulation into SimStats
 *  (the EVAL entrypoint reuses it). Normal-approximation, one-tail. */
class SimStatsTest {

    @Test
    fun atExpectedRateOneTailPIsHalf() {
        // successes == n·p ⇒ z = 0 ⇒ one-tail "greater" p ≈ 0.5
        assertEquals(0.5, SimStats.binomialTest(50.0, 100.0, 0.5, "greater"), 1e-6)
    }

    @Test
    fun sixtyOfHundredBeatsFiftyPercentSignificantly() {
        val p = SimStats.binomialTest(60.0, 100.0, 0.5, "greater")
        assertTrue("60/100 vs 50% should be significant (p=$p)", p < 0.05) // ≈ 0.0228
    }

    @Test
    fun atRateIsNotSignificant() {
        assertTrue(SimStats.binomialTest(50.0, 100.0, 0.5, "greater") > 0.05)
    }

    @Test
    fun greaterAndLessAreComplementaryAroundMean() {
        val g = SimStats.binomialTest(60.0, 100.0, 0.5, "greater")
        val l = SimStats.binomialTest(60.0, 100.0, 0.5, "less")
        assertEquals(1.0, g + l, 1e-9)
    }
}
