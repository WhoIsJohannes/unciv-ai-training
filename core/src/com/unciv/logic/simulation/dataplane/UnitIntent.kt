package com.unciv.logic.simulation.dataplane

/**
 * v8 — the LAND-MILITARY unit INTENT vocabulary. Each entry is a candidate behaviour the net's
 * per-unit intent head can select; the chosen intent dispatches to the matching `UnitAutomation.tryX`
 * sub-routine (pathfinding stays 100% heuristic — v8 adds NO movement/tile-target logic).
 *
 * The order is the STABLE order of the `UnitAutomation.automateUnitMoves` land-military ladder
 * (UnitAutomation.kt:74-116), first-firing rung = the executed intent. The repeated health-gated
 * `tryHealUnit`/`tryRetreat`/wait-heal rungs are DEDUPED into a single [HEAL]. `UPGRADE` is only
 * reachable for HUMAN civs in the ladder (line 47), so it is inert for the AI self-play civs (its
 * mask bit stays 0) — kept in the vocabulary for a stable, future-proof ordering.
 *
 * The ORDINAL is the head/mask index and the recorded action idx. This order is FROZEN: it feeds the
 * [Vocab] `enum:UnitIntent` canonical section (⇒ the [RulesetFingerprint]) and every recorded shard.
 * Appending a new intent at the END is safe (bumps the fingerprint); reordering/removing perishes
 * shards (regenerate). Civilian / air / nuke units are NOT modelled (recorded intent = −1).
 */
enum class UnitIntent {
    HEAL,                 // tryHealUnit / tryRetreat / wait-and-heal-in-place (rungs 38,78,81,100,105)
    UPGRADE,              // tryUpgradeUnit (human-only rung 47 ⇒ inert for AI civs)
    ACCOMPANY,            // tryAccompanySettlerOrGreatPerson (rung 74)
    GO_TO_RUIN,           // tryGoToRuin (rung 76)
    DEFEND_SIEGED_CITY,   // tryHeadTowardsOurSiegedCity (rung 83)
    ATTACK,               // tryDisembarkUnitToAttackPosition / tryAttacking (rungs 86,89)
    RETAKE_CITY,          // tryTakeBackCapturedCity (rung 91)
    ADVANCE_ENEMY_CITY,   // HeadTowardsEnemyCityAutomation.tryHeadTowardsEnemyCity (rung 94)
    ATTACK_ENCAMPMENT,    // tryHeadTowardsEncampment (rung 96)
    GARRISON,             // tryGarrisoningLandUnit (rung 98)
    ADVANCE_CLOSE_ENEMY,  // tryAdvanceTowardsCloseEnemy (rung 103)
    PREPARE,              // tryPrepare (rung 107)
    EXPLORE,              // tryExplore (rung 110)
    FOG_BUST;             // tryFogBust (rung 112)

    companion object {
        /** The head/mask width. Mirrors `vocab.buildingCount + vocab.unitCount` for construction. */
        val COUNT = entries.size

        /** The 0-indexed mask/head idx → intent, or null out of range (inverse of `ordinal`). */
        fun fromIndex(idx: Int): UnitIntent? = entries.getOrNull(idx)

        /** The canonical id strings for the [Vocab] `enum:UnitIntent` section (⇒ RulesetFingerprint). */
        val ID_NAMES: List<String> = entries.map { it.name }
    }
}
