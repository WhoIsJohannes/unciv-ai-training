package com.unciv.logic.simulation.dataplane

import com.unciv.logic.automation.unit.UnitAutomation
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.map.mapunit.MapUnit
import kotlin.math.ln
import kotlin.random.Random

/**
 * Policy-agnostic seam: chooses an action index per factored head from the legal mask, and acts on
 * a unit. Injected at `automateCivMoves`; a real RL policy implements this later. The data plane
 * records each [chooseIndex] result as the step's action label.
 */
interface PolicyProvider {
    /** Pick a legal index for [head] given its boolean legality mask, or -1 for none/abstain. */
    fun chooseIndex(head: String, civ: Civilization, legalMask: BooleanArray, turn: Int): Int

    /**
     * v6 — [chooseIndex] PLUS the behavior-policy log-prob log π_b(chosen | state) of the picked
     * action, recorded into the shard for off-policy replay. Default: delegate to [chooseIndex] and
     * return a UNIFORM-over-legal log-prob `ln(1/nLegal)` (idx<0 → 0f).
     *
     * WARNING: the uniform default is correct ONLY for a uniform policy (the [RandomPolicy] stub),
     * AND its value is harmless for the learner because round-0 RandomPolicy data is EXCLUDED from
     * the replay window (the learner-slot filter discards it). ANY non-uniform PolicyProvider (e.g.
     * a softmax-sampling net) MUST override this to record its TRUE sampling log-prob — otherwise the
     * importance ratio for replayed steps would be anchored to the wrong behavior policy.
     */
    fun chooseIndexWithLogp(head: String, civ: Civilization, legalMask: BooleanArray, turn: Int): Pair<Int, Float> {
        val idx = chooseIndex(head, civ, legalMask, turn)
        if (idx < 0) return -1 to 0f
        val nLegal = legalMask.count { it }
        return idx to (if (nLegal > 0) ln(1.0 / nLegal).toFloat() else 0f)
    }

    /** Act on a unit. Default policies route this into the existing UnitAutomation sub-routines. */
    fun actUnit(unit: MapUnit)
}

/**
 * Default plumbing stub (NOT an agent): uniform-random over the legal candidates per head, with
 * unit intents delegated to `UnitAutomation`. RNG is derived deterministically per (civ, turn) via
 * [rngFor] so recorded action labels replay byte-identically.
 */
class RandomPolicy(private val rngFor: (Civilization, Int) -> Random) : PolicyProvider {

    override fun chooseIndex(head: String, civ: Civilization, legalMask: BooleanArray, turn: Int): Int {
        val legal = legalMask.indices.filter { legalMask[it] }
        if (legal.isEmpty()) return -1
        return legal[rngFor(civ, turn).nextInt(legal.size)]
    }

    override fun actUnit(unit: MapUnit) {
        UnitAutomation.automateUnitMoves(unit)
    }
}
