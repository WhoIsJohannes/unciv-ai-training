package com.unciv.logic.simulation.dataplane

import com.unciv.logic.automation.unit.UnitAutomation
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.map.mapunit.MapUnit
import kotlin.random.Random

/**
 * Policy-agnostic seam: chooses an action index per factored head from the legal mask, and acts on
 * a unit. Injected at `automateCivMoves`; a real RL policy implements this later. The data plane
 * records each [chooseIndex] result as the step's action label.
 */
interface PolicyProvider {
    /** Pick a legal index for [head] given its boolean legality mask, or -1 for none/abstain. */
    fun chooseIndex(head: String, civ: Civilization, legalMask: BooleanArray, turn: Int): Int

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
