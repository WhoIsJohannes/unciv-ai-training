package com.unciv.logic.simulation.dataplane

/**
 * Fixed-width caps for the padded entity-list token groups. Sized for the largest standard
 * GnK scenario (Huge map, r40, up to Unciv's max major civs) with headroom; configurable so a
 * campaign can shrink them (which bumps the RulesetFingerprint-independent layout — see overflow
 * policy). Overflow (entities exceed a cap) is clamped + flagged, never silently dropped.
 */
data class SampleCaps(
    val maxMajorCivs: Int = 16,
    val maxCityStates: Int = 24,
    val maxOwnCities: Int = 64,
    val maxVisOppCities: Int = 64,
    val maxOwnUnits: Int = 192,
    val maxVisOppUnits: Int = 192,
) {
    val maxCivTokens get() = maxMajorCivs + maxCityStates

    companion object {
        val DEFAULT = SampleCaps()
    }
}

/**
 * Opt-in configuration for the trajectory data plane. Default = disabled, fair, non-strict.
 * `omniscientOpponents` (ablation upper-bound) and `strictVersioning` MUST default off.
 */
data class SampleConfig(
    val enabled: Boolean = false,
    val outputDir: String? = null,
    /** Ablation ONLY: feed raw per-civ values + full opponent vision. The single switch that
     *  changes observations. MUST default off. */
    val omniscientOpponents: Boolean = false,
    /** When a startup fingerprint/version mismatch is detected: refuse (true) vs warn (false). */
    val strictVersioning: Boolean = false,
    val expectedRulesetFingerprint: String? = null,
    val expectedSchemaVersion: Int? = null,
    val caps: SampleCaps = SampleCaps.DEFAULT,
    /** Route GameStarter's player shuffle through the seeded RNG (sim path only). */
    val deterministicShuffle: Boolean = true,
    /** v7: when true, the installed policy DRIVES each deciding city's production (per-city construction
     *  head); off ⇒ construction stays heuristic (the no-op / v6 path). MUST default off. */
    val controlConstruction: Boolean = false,
    /** v7.1b (default OFF — rejected as a standalone fix): restrict the policy's construction choice to
     *  BUILDINGS (mask out units in `mask_construction` when [controlConstruction]); units fall back to
     *  the heuristic. Dodges the credit problem rather than solving it — superseded by v7.2 PBRS, which
     *  lets the net learn the FULL unit/building balance. Kept as an option; default false = full control. */
    val constructionBuildingsOnly: Boolean = false,
)
