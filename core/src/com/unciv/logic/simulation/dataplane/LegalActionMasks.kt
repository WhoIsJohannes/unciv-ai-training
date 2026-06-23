package com.unciv.logic.simulation.dataplane

import com.unciv.logic.city.City
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.map.mapunit.MapUnit
import com.unciv.models.ruleset.Ruleset

/**
 * Factored legal-action masks built STRAIGHT from the engine's existing candidate enumeration —
 * no re-derivation. Each returns a `BooleanArray` over the relevant vocab (bit i ⇔ vocab item i is
 * a legal candidate). The diplomatic-vote head is built in the [Featurizer] because it maps onto
 * civ SLOTS (which the Featurizer assigns). Unit-intent+target is delegated to `UnitAutomation`
 * by the RandomPolicy and is not a flat-enumerated head (the engine has no discrete candidate list
 * for it) — see build-output.md.
 */
object LegalActionMasks {

    fun techMask(civ: Civilization, vocab: Vocab): BooleanArray {
        val m = BooleanArray(vocab.techCount)
        for ((name, idx) in iterate(vocab, Vocab.TECHS))
            if (civ.tech.canBeResearched(name)) m[idx] = true
        return m
    }

    fun policyMask(civ: Civilization, vocab: Vocab, ruleset: Ruleset): BooleanArray {
        val m = BooleanArray(vocab.policyCount)
        for ((name, idx) in iterate(vocab, Vocab.POLICIES)) {
            val policy = ruleset.policies[name] ?: continue
            if (civ.policies.isAdoptable(policy)) m[idx] = true
        }
        return m
    }

    /** Great-person head: over the units vocab; bit set for each great-person unit available now. */
    fun greatPersonMask(civ: Civilization, vocab: Vocab): BooleanArray {
        val m = BooleanArray(vocab.unitCount)
        for (gp in civ.greatPeople.getGreatPeople()) {
            val idx = vocab.unit(gp.name)
            if (idx >= 0) m[idx] = true
        }
        return m
    }

    /** Per-city construction head: concat of [buildings | units] vocab; bit set if buildable now. */
    fun constructionMask(city: City, vocab: Vocab): BooleanArray {
        val m = BooleanArray(vocab.buildingCount + vocab.unitCount)
        for (b in city.cityConstructions.getBuildableBuildings()) {
            val idx = vocab.building(b.name)
            if (idx >= 0) m[idx] = true
        }
        for (u in city.cityConstructions.getConstructableUnits()) {
            val idx = vocab.unit(u.name)
            if (idx >= 0) m[vocab.buildingCount + idx] = true
        }
        return m
    }

    /** Per-unit promotion head: over the promotions vocab. */
    fun promotionMask(unit: MapUnit, vocab: Vocab): BooleanArray {
        val m = BooleanArray(vocab.promotionCount)
        for (p in unit.promotions.getAvailablePromotions()) {
            val idx = vocab.promotion(p.name)
            if (idx >= 0) m[idx] = true
        }
        return m
    }

    private fun iterate(vocab: Vocab, category: String): List<Pair<String, Int>> =
        vocab.sections.first { it.first == category }.second.mapIndexed { i, name -> name to i }
}
