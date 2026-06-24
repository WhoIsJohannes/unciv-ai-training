package com.unciv.logic.simulation.dataplane

import com.unciv.models.ruleset.Ruleset
import com.unciv.models.ruleset.tile.ResourceType
import com.unciv.models.ruleset.tile.TerrainType
import com.unciv.ui.screens.victoryscreen.RankingType

/**
 * Programmatic string-id → dense-index adapters, built from the loaded GnK ruleset in a
 * deterministic CANONICAL order at sim start. Ruleset collections are `LinkedHashMap` (insertion
 * order is stable); `religions` is the lone `ArrayList` → sorted. The SAME canonical-section list
 * feeds [RulesetFingerprint], so a content change is reflected in BOTH the vocab indices and the
 * fingerprint — never silently mismatched.
 */
class Vocab(ruleset: Ruleset) {

    /** Ordered (category-name → ordered ids). Single source of truth for vocab + fingerprint. */
    val sections: List<Pair<String, List<String>>> = canonicalSections(ruleset)

    private val maps: Map<String, Map<String, Int>> =
        sections.associate { (name, ids) -> name to ids.withIndex().associate { (i, id) -> id to i } }

    private fun map(category: String) = maps.getValue(category)

    fun size(category: String): Int = map(category).size

    /** Index of [id] within [category], or -1 if absent (an out-of-vocab content drift). */
    fun index(category: String, id: String): Int = map(category)[id] ?: -1

    /** Reverse: the id at dense [index] within [category], or null if out of range. Used by the
     *  self-play CONTROL path to map a chosen action index back to a tech/policy id to APPLY it. */
    fun id(category: String, index: Int): String? = sections.firstOrNull { it.first == category }?.second?.getOrNull(index)
    fun techId(index: Int) = id(TECHS, index)
    fun policyId(index: Int) = id(POLICIES, index)

    // Convenience accessors for the hot paths.
    fun tech(id: String) = index(TECHS, id)
    fun building(id: String) = index(BUILDINGS, id)
    fun unit(id: String) = index(UNITS, id)
    fun policy(id: String) = index(POLICIES, id)
    fun policyBranch(id: String) = index(POLICY_BRANCHES, id)
    fun resource(id: String) = index(RESOURCES, id)
    fun promotion(id: String) = index(PROMOTIONS, id)
    fun terrain(id: String) = index(TERRAINS, id)
    fun improvement(id: String) = index(IMPROVEMENTS, id)
    fun nation(id: String) = index(NATIONS, id)
    fun era(id: String) = index(ERAS, id)
    fun victory(id: String) = index(VICTORIES, id)
    fun religion(id: String) = index(RELIGIONS, id)

    /**
     * v3 collision-free construction code over the disjoint building∪unit namespace (field width is
     * reserved as buildingCount+unitCount): building#k → k+1, unit#k → buildingCount+k+1, none → 0.
     * The +1 keeps 0 = "no/empty construction". (Was a latent bug: the unit branch lacked the
     * buildingCount offset, so building#k and unit#k collided onto the same code.)
     */
    fun constructionCode(name: String): Int {
        val b = building(name)
        if (b >= 0) return b + 1
        val u = unit(name)
        return if (u >= 0) buildingCount + u + 1 else 0
    }

    val techCount get() = size(TECHS)
    val buildingCount get() = size(BUILDINGS)
    val unitCount get() = size(UNITS)
    val policyCount get() = size(POLICIES)
    val policyBranchCount get() = size(POLICY_BRANCHES)
    val resourceCount get() = size(RESOURCES)
    val promotionCount get() = size(PROMOTIONS)
    val nationCount get() = size(NATIONS)

    companion object {
        const val TECHS = "technologies"
        const val UNITS = "units"
        const val BUILDINGS = "buildings"
        const val POLICIES = "policies"
        const val POLICY_BRANCHES = "policyBranches"
        const val RESOURCES = "tileResources"
        const val PROMOTIONS = "unitPromotions"
        const val TERRAINS = "terrains"
        const val IMPROVEMENTS = "tileImprovements"
        const val NATIONS = "nations"
        const val ERAS = "eras"
        const val VICTORIES = "victories"
        const val RELIGIONS = "religions"

        /**
         * The canonical, deterministic ordering used for BOTH vocab indexing and the ruleset
         * fingerprint. Entity collections in `LinkedHashMap` iteration order; `religions` sorted;
         * the enum/structural sections in declaration order.
         */
        fun canonicalSections(ruleset: Ruleset): List<Pair<String, List<String>>> = listOf(
            TECHS to ruleset.technologies.keys.toList(),
            UNITS to ruleset.units.keys.toList(),
            BUILDINGS to ruleset.buildings.keys.toList(),
            POLICIES to ruleset.policies.keys.toList(),
            POLICY_BRANCHES to ruleset.policyBranches.keys.toList(),
            RESOURCES to ruleset.tileResources.keys.toList(),
            PROMOTIONS to ruleset.unitPromotions.keys.toList(),
            TERRAINS to ruleset.terrains.keys.toList(),
            IMPROVEMENTS to ruleset.tileImprovements.keys.toList(),
            NATIONS to ruleset.nations.keys.toList(),
            ERAS to ruleset.eras.keys.toList(),
            VICTORIES to ruleset.victories.keys.toList(),
            RELIGIONS to ruleset.religions.sorted(),
            "enum:RankingType" to RankingType.entries.map { it.name },
            "enum:ResourceType" to ResourceType.entries.map { it.name },
            "enum:TerrainType" to TerrainType.entries.map { it.name },
            "schema:spatialChannels" to SampleSchema.SPATIAL_CHANNELS,
            "schema:demographics" to SampleSchema.DEMOGRAPHIC_CATEGORIES,
            "schema:maskHeads" to SampleSchema.MASK_HEADS,
        )
    }
}
