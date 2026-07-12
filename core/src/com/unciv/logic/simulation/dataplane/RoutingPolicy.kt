package com.unciv.logic.simulation.dataplane

import com.unciv.logic.city.City
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.map.mapunit.MapUnit

/**
 * Composite [PolicyProvider] that routes by civ identity: the single LEARNER civ goes to
 * [learner], every other civ to [opponent]. This is how "learner vs RandomPolicy" is expressed
 * over the data plane's one-policy-per-run injection seam ([DataPlaneHooks.install]) WITHOUT
 * modifying the data plane — the learner is identified by `civID` (a pinned nation in the Tiny
 * self-play config, stable across games so win-rate aggregates cleanly).
 */
class RoutingPolicy(
    private val learnerCivId: String,
    private val learner: PolicyProvider,
    private val opponent: PolicyProvider,
) : PolicyProvider {

    private fun forCiv(civ: Civilization): PolicyProvider =
        if (civ.civID == learnerCivId) learner else opponent

    override fun chooseIndex(head: String, civ: Civilization, legalMask: BooleanArray, turn: Int): Int =
        forCiv(civ).chooseIndex(head, civ, legalMask, turn)

    /** v6 — delegate the behavior log-prob to the routed policy too (do NOT inherit the uniform
     *  default, which would replace a routed net's true sampling logp with ln(1/nLegal)). */
    override fun chooseIndexWithLogp(head: String, civ: Civilization, legalMask: BooleanArray, turn: Int): Pair<Int, Float> =
        forCiv(civ).chooseIndexWithLogp(head, civ, legalMask, turn)

    /** v7 — route per-city construction to the civ's policy (learner net vs opponent uniform), so the
     *  recorded construction log-prob is the routed policy's TRUE sampling logp, not the abstain default. */
    override fun chooseConstructionWithLogp(civ: Civilization, city: City, cityRow: Int, legalMask: BooleanArray, turn: Int): Pair<Int, Float> =
        forCiv(civ).chooseConstructionWithLogp(civ, city, cityRow, legalMask, turn)

    /** v8 — route per-unit intent to the civ's policy (learner net vs opponent uniform), so the recorded
     *  intent log-prob is the routed policy's TRUE sampling logp, not the abstain default. */
    override fun chooseUnitIntentWithLogp(civ: Civilization, unit: MapUnit, unitRow: Int, legalMask: BooleanArray, turn: Int): Pair<Int, FloatArray> =
        forCiv(civ).chooseUnitIntentWithLogp(civ, unit, unitRow, legalMask, turn)

    override fun actUnit(unit: MapUnit) = forCiv(unit.civ).actUnit(unit)
}
