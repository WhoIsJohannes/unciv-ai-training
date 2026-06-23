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
)
