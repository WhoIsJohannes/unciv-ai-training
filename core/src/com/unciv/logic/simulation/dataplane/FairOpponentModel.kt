package com.unciv.logic.simulation.dataplane

import com.unciv.logic.civilization.Civilization
import com.unciv.logic.civilization.diplomacy.DiplomacyFlags
import com.unciv.models.ruleset.Ruleset
import com.unciv.ui.screens.victoryscreen.RankingType

/** A growable fixed-width float token written by index cursor. */
class TokenWriter(val width: Int) {
    val data = FloatArray(width)
    private var cursor = 0
    fun put(v: Float) { if (cursor < width) data[cursor] = v; cursor++ }
    fun put(v: Int) = put(v.toFloat())
    fun put(b: Boolean) = put(if (b) 1f else 0f)
    fun putBits(set: Set<Int>, n: Int) { for (i in 0 until n) put(if (i in set) 1f else 0f) }
    /** Skip [n] slots (leave them zero) — used to zero out an unavailable attribute group. */
    fun skip(n: Int) { cursor += n }
    val written get() = cursor
}

/**
 * Per-category ranking context over the ALIVE MAJOR civs, computed once per observation. Provides
 * the DOWN-GATED rank + bucket for an opponent and the identity-free best/avg/worst aggregate —
 * never a raw per-civ float (the Victory-Screen/Demographics leak we deliberately do not replicate).
 */
class DemographicsContext(decidingCiv: Civilization, categories: List<RankingType>) {
    private val aliveMajors = decidingCiv.gameInfo.civilizations.filter { it.isMajorCiv() && it.isAlive() }
    // metForRank = civs whose value the deciding civ may rank against (itself + met majors).
    private val rankable = aliveMajors.filter { it === decidingCiv || decidingCiv.knows(it) }

    val perCategory: Map<RankingType, CategoryStats> = categories.associateWith { cat ->
        val values = aliveMajors.map { it.getStatForRanking(cat) }
        val rankableSorted = rankable
            .map { it to it.getStatForRanking(cat) }
            // deterministic: value desc, ties broken by civID (R9)
            .sortedWith(compareByDescending<Pair<Civilization, Int>> { it.second }.thenBy { it.first.civID })
        CategoryStats(
            best = (values.maxOrNull() ?: 0).toFloat(),
            avg = if (values.isEmpty()) 0f else values.sum().toFloat() / values.size,
            worst = (values.minOrNull() ?: 0).toFloat(),
            min = (rankableSorted.minOfOrNull { it.second } ?: 0),
            max = (rankableSorted.maxOfOrNull { it.second } ?: 0),
            rankByCivId = rankableSorted.withIndex().associate { (i, p) -> p.first.civID to (i + 1) },
        )
    }

    class CategoryStats(
        val best: Float, val avg: Float, val worst: Float,
        val min: Int, val max: Int, val rankByCivId: Map<String, Int>,
    ) {
        fun rankOf(civId: String): Int = rankByCivId[civId] ?: 0
        fun bucketOf(value: Int): Int {
            if (max <= min) return 0
            val frac = (value - min).toFloat() / (max - min).toFloat()
            return (frac * SampleSchema.NUM_BUCKETS).toInt().coerceIn(0, SampleSchema.NUM_BUCKETS - 1)
        }
    }
}

/**
 * Encodes one opponent civ O relative to deciding civ X into a fixed-width token per the fair
 * opponent-information model (the prompt's encoding table + decisions D4). The ONE place opponent
 * intel is read. `config.omniscientOpponents=true` bypasses every gate (ablation upper-bound).
 *
 * Invariants the leakage/unmet/down-gate tests assert:
 *  - UNMET ⇒ token is all-zero, every availability slot 0.
 *  - Gold/GPT/resources only when a trade screen is openable (met ∧ major ∧ not-CS) — never otherwise.
 *  - Tech is the COUNT only; the tech LIST never enters.
 *  - Demographics are RANK + BUCKET only — no raw per-civ float.
 *  - Spaceship is the COUNT only; part identities never enter.
 */
class FairOpponentModel(private val vocab: Vocab, ruleset: Ruleset, private val config: SampleConfig) {

    private val wonderIds: List<String> = ruleset.buildings.values.filter { it.isWonder }.map { it.name }
    private val wonderIndex: Map<String, Int> = wonderIds.withIndex().associate { (i, n) -> n to i }
    private val demographics: List<RankingType> = SampleSchema.DEMOGRAPHIC_CATEGORIES.map { RankingType.valueOf(it) }
    private val numBranches = vocab.policyBranchCount

    /** Fixed token width (same for every civ slot, met or not). */
    val tokenWidth: Int = run {
        var w = 0
        w += 1                                  // met
        w += 1                                  // era index
        w += 1                                  // adopted-policy count
        w += numBranches                        // policy-branch adopted bits
        w += wonderIds.size                     // wonders multi-hot (broadcast)
        w += 1                                  // total score
        w += 1                                  // tech COUNT (never the list)
        w += 4                                  // victory numerators: ss-parts, orig-capitals, branches-done, religion-conv
        w += 1                                  // ss-parts denom-mask
        w += 1                                  // opinion-of-X
        w += 4                                  // diplo flags vs X: war / friendship / defensive-pact / denounced
        w += demographics.size * 2              // rank + bucket per demographic (NO raw float)
        w += 3                                  // trade: gold, gpt, tradeable-resource-count
        w += 1                                  // trade-mask
        w += 1                                  // #cities of O seen
        w
    }

    fun encode(x: Civilization, o: Civilization, demoCtx: DemographicsContext): FloatArray {
        val t = TokenWriter(tokenWidth)
        val met = x.knows(o)
        val omni = config.omniscientOpponents
        if (!met && !omni) {
            return t.data // all-zero, every mask 0 (UNMET invariant)
        }

        t.put(if (met || omni) 1 else 0)                       // met
        t.put(o.tech.era.eraNumber)                            // era
        t.put(o.policies.adoptedPolicies.count { !com.unciv.models.ruleset.Policy.isBranchCompleteByName(it) })

        val adoptedBranchIdx = o.policies.adoptedPolicies
            .mapNotNull { name -> vocab.policyBranch(name).takeIf { it >= 0 } }.toSet()
        t.putBits(adoptedBranchIdx, numBranches)

        // wonders multi-hot (broadcast/known by name)
        val ownedWonders = o.cities.asSequence()
            .flatMap { it.cityConstructions.getBuiltBuildings() }
            .filter { it.isWonder }
            .mapNotNull { wonderIndex[it.name] }.toSet()
        t.putBits(ownedWonders, wonderIds.size)

        t.put(o.calculateTotalScore().toInt())                 // total score (EXACT, met)
        t.put(o.tech.researchedTechnologies.size)              // tech COUNT (never list)

        // victory numerators
        val ssParts = o.victoryManager.currentsSpaceshipParts.values.sum()
        t.put(ssParts)
        t.put(o.cities.count { it.isOriginalCapital })         // domination numerator (derived)
        t.put(o.policies.adoptedPolicies.count { com.unciv.models.ruleset.Policy.isBranchCompleteByName(it) })
        t.put(0)                                               // religion-conversion numerator: v1 stub (needs ReligionManager API) — denom-mask handles
        t.put(if (o.shouldHideCivCount()) 0 else 1)            // ss/denom availability proxy

        // opinion + diplo vs X
        val dm = o.getDiplomacyManager(x)
        t.put(((dm?.opinionOfOtherCiv() ?: 0f)).toInt())
        t.put(x.isAtWarWith(o))                                // war (via diplomacyFunctions; no flag enum)
        t.put(dm?.hasFlag(DiplomacyFlags.DeclarationOfFriendship) == true)
        t.put(dm?.hasFlag(DiplomacyFlags.DefensivePact) == true)
        t.put(dm?.hasFlag(DiplomacyFlags.Denunciation) == true)

        // demographics: rank + bucket ONLY (down-gate)
        for (cat in demographics) {
            val stats = demoCtx.perCategory.getValue(cat)
            val raw = o.getStatForRanking(cat)
            t.put(stats.rankOf(o.civID))
            t.put(stats.bucketOf(raw))
        }

        // trade slots (gated)
        val tradeAvailable = omni || (met && o.isMajorCiv() && !o.isCityState && x.isMajorCiv())
        if (tradeAvailable) {
            t.put(o.gold)
            t.put(o.stats.statsForNextTurn.gold.toInt())
            t.put(o.getPerTurnResourcesWithOriginsForTrade().size + o.getStockpiledResourcesForTrade().size)
        } else {
            t.skip(3)
        }
        t.put(if (tradeAvailable) 1 else 0)                    // trade-mask

        t.put(citiesOfSeen(x, o))                              // #cities of O seen (from tile vision)
        return t.data
    }

    private fun citiesOfSeen(x: Civilization, o: Civilization): Int =
        o.cities.count { x.viewableTiles.contains(it.getCenterTile()) }
}
