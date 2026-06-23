package com.unciv.logic.simulation

import com.unciv.logic.GameInfo
import com.unciv.logic.civilization.Civilization
import kotlin.math.sqrt

/**
 * Headless-simulation statistics, extracted from [Simulation] so the self-play EVAL entrypoint
 * can reuse the SAME win-rate p-value as the interactive simulation report (and so it is unit
 * testable). Kept deliberately minimal: a one-tail binomial test (normal approximation) + the
 * shared "who won at the turn cap" tiebreaker. No ONNX/ML dependency — lives in `core`.
 */
object SimStats {

    /** One-tail binomial test via the normal approximation (unchanged from the prior in-`Simulation`
     *  implementation). `alternative` ∈ {"greater","less"}. Valid when n·p ≥ 10 and n·(1−p) ≥ 10. */
    fun binomialTest(successes: Double, trials: Double, p: Double, alternative: String): Double {
        val q = 1 - p
        val mean = trials * p
        val variance = trials * p * q
        val stdDev = sqrt(variance)
        val z = (successes - mean) / stdDev
        val pValue = 1 - normalCdf(z)
        return when (alternative) {
            "greater" -> pValue
            "less" -> 1 - pValue
            else -> throw IllegalArgumentException("Alternative must be 'greater' or 'less'")
        }
    }

    fun normalCdf(z: Double): Double = 0.5 * (1 + erf(z / sqrt(2.0)))

    /** Abramowitz & Stegun 7.1.26 approximation of the error function. */
    fun erf(x: Double): Double {
        val t = 1.0 / (1.0 + 0.5 * kotlin.math.abs(x))
        val tau = t * kotlin.math.exp(
            -x * x - 1.26551223 +
                t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
                t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
                t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))
        )
        return if (x >= 0) 1 - tau else tau - 1
    }

    /**
     * Turn-cap tiebreaker (self-play win rule, D7): when a game reaches the turn cap with NO formal
     * victory, the winner is the alive major civ with the strictly-highest total score. Returns
     * `null` on an exact tie (a draw) or when there are no alive majors. Used identically for the
     * training terminal reward AND the EVAL win-rate so the two never diverge.
     */
    fun scoreLeader(gameInfo: GameInfo): Civilization? {
        val aliveMajors = gameInfo.civilizations.filter { it.isMajorCiv() && it.isAlive() && !it.isSpectator() }
        if (aliveMajors.isEmpty()) return null
        val scored = aliveMajors.map { it to it.calculateTotalScore() }
        val best = scored.maxByOrNull { it.second } ?: return null
        // exact tie at the top ⇒ draw (no winner)
        if (scored.count { it.second == best.second } > 1) return null
        return best.first
    }
}
