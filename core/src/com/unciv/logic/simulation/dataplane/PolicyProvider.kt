package com.unciv.logic.simulation.dataplane

import com.unciv.logic.automation.unit.UnitAutomation
import com.unciv.logic.city.City
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

    /**
     * v7 — per-ENTITY construction decision: pick a legal construction-mask index for [city] (row
     * [cityRow] in the civ's [Featurizer.orderedOwnCities]) PLUS its behavior log-prob log π_b, for
     * off-policy replay. The index is in the 0-indexed construction-mask space (width
     * buildingCount+unitCount), decoded back to a building/unit id via [Vocab.constructionId].
     *
     * Default: ABSTAIN (`-1 to 0f`) ⇒ the heuristic `ConstructionAutomation` builds. This is the
     * safe fallback for any policy that doesn't model construction (and the OFF / no-op path).
     * [RandomPolicy] overrides with uniform-over-legal (the opponent is "random" on construction the
     * same way it is on tech/policy); [OnnxPolicy] overrides with the net's masked-softmax sample.
     */
    fun chooseConstructionWithLogp(civ: Civilization, city: City, cityRow: Int, legalMask: BooleanArray, turn: Int): Pair<Int, Float> = -1 to 0f

    /**
     * v8 — per-ENTITY unit-intent decision: sample a legal INTENT for [unit] (row [unitRow] in the civ's
     * [Featurizer.orderedOwnUnits]) from its [legalMask] (over the [UnitIntent] ordinal space), returning
     * `(sampledIntentIdx, logpVector)`. The [sampledIntentIdx] (−1 = abstain) is DISPATCHED to that intent's
     * `UnitAutomation.tryX`. The [logpVector] is the FULL masked log-softmax over the intent space (width
     * [UnitIntent.COUNT], 0f for illegal) so the recorder can look up `log π_b(realized)` at turn-end — the
     * EXECUTED intent may differ from the sampled one when dispatch falls back to the heuristic ladder.
     *
     * Default: ABSTAIN (`-1 to empty`) ⇒ the unit stays fully heuristic (the OFF / no-op path). [RandomPolicy]
     * overrides uniform-over-legal; [OnnxPolicy] overrides with the net's masked-softmax sample + log-softmax.
     */
    fun chooseUnitIntentWithLogp(civ: Civilization, unit: MapUnit, unitRow: Int, legalMask: BooleanArray, turn: Int): Pair<Int, FloatArray> = -1 to FloatArray(0)

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

    /** Uniform-over-legal construction with its uniform log-prob `ln(1/nLegal)` — the opponent is
     *  "random" on construction the same way it is on tech/policy. Deterministic via [rngFor]. */
    override fun chooseConstructionWithLogp(civ: Civilization, city: City, cityRow: Int, legalMask: BooleanArray, turn: Int): Pair<Int, Float> {
        val legal = legalMask.indices.filter { legalMask[it] }
        if (legal.isEmpty()) return -1 to 0f
        val idx = legal[rngFor(civ, turn).nextInt(legal.size)]
        return idx to ln(1.0 / legal.size).toFloat()
    }

    /** v8: uniform-over-legal unit intent + its uniform log-prob vector (`ln(1/nLegal)` for each legal
     *  intent, 0f elsewhere) — the opponent is "random" on unit intent the same way it is on construction. */
    override fun chooseUnitIntentWithLogp(civ: Civilization, unit: MapUnit, unitRow: Int, legalMask: BooleanArray, turn: Int): Pair<Int, FloatArray> {
        val legal = legalMask.indices.filter { legalMask[it] }
        if (legal.isEmpty()) return -1 to FloatArray(0)
        val idx = legal[rngFor(civ, turn).nextInt(legal.size)]
        val lp = ln(1.0 / legal.size).toFloat()
        val vec = FloatArray(legalMask.size); for (k in legal) vec[k] = lp
        return idx to vec
    }

    override fun actUnit(unit: MapUnit) {
        UnitAutomation.automateUnitMoves(unit)
    }
}
