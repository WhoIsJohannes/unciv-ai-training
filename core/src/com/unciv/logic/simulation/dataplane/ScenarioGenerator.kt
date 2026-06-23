package com.unciv.logic.simulation.dataplane

import com.unciv.Constants
import com.unciv.logic.civilization.PlayerType
import com.unciv.logic.map.MapParameters
import com.unciv.logic.map.MapShape
import com.unciv.logic.map.MapSize
import com.unciv.logic.map.MapType
import com.unciv.models.metadata.BaseRuleset
import com.unciv.models.metadata.GameParameters
import com.unciv.models.metadata.GameSetupInfo
import com.unciv.models.metadata.Player
import kotlin.random.Random

/** A resolved, valid GnK scenario + its determinism keys + a per-episode provenance log line. */
class ScenarioSpec(
    val gameSetupInfo: GameSetupInfo,
    val gameId: String,
    val seed: Long,
    val mapSize: String,
    val numMajorCivs: Int,
    val numCityStates: Int,
    val episodeLogJson: String,
)

/**
 * Randomizes map + game params over the FULL standard GnK envelope (map sizes Tiny–Huge, 2–N major
 * civs, city-states) with guardrails: land-tiles/civ ≥ 80, seed ≠ 0, deterministic gameId, and the
 * MapRegions "too many players" retry (surfaced as an exception caught by the caller). Bounded by
 * [maxAttempts] (then throws) so generation can never loop forever.
 */
class ScenarioGenerator(
    private val caps: SampleCaps,
    private val maxAttempts: Int = 100,
    private val minLandTilesPerCiv: Int = 80,
    /** Cap the largest randomized map radius (default = full standard range incl. Huge r40).
     *  Useful for fast smoke/demo runs; the campaign default is unconstrained. */
    private val maxMapRadius: Int = Int.MAX_VALUE,
) {
    // (predefined map size name → MapSize) for the standard selectable range, within the cap.
    private val sizes: List<Pair<String, MapSize>> = listOf(
        "Tiny" to MapSize.Tiny, "Small" to MapSize.Small, "Medium" to MapSize.Medium,
        "Large" to MapSize.Large, "Huge" to MapSize.Huge,
    ).filter { it.second.radius <= maxMapRadius }.ifEmpty { listOf("Tiny" to MapSize.Tiny) }

    fun generate(seedBase: Long, episode: Int): ScenarioSpec {
        // Per-episode RNG: deterministic in (seedBase, episode); the map seed is forced non-zero.
        val rng = Random(seedBase * 1_000_003L + episode * 31L + 1L)

        repeat(maxAttempts) {
            val (sizeName, mapSize) = sizes[rng.nextInt(sizes.size)]
            val numMajor = 2 + rng.nextInt(caps.maxMajorCivs - 1)         // 2..maxMajorCivs
            val numCityStates = rng.nextInt(caps.maxCityStates + 1)        // 0..maxCityStates
            if (estimatedLandPerCiv(mapSize, numMajor) < minLandTilesPerCiv) return@repeat

            var seed = rng.nextLong()
            if (seed == 0L) seed = 1L                                      // enforce seed != 0

            val gp = GameParameters().apply {
                baseRuleset = BaseRuleset.Civ_V_GnK.fullName
                difficulty = "Prince"
                players = ArrayList<Player>().apply {
                    repeat(numMajor) { add(Player()) }                    // random-nation AI majors
                    add(Player(Constants.spectator, PlayerType.Human))    // headless human spectator
                    // (the engine's setTransients requires a human civ; the spectator never acts)
                }
                this.numberOfCityStates = numCityStates
                shufflePlayerOrder = true
                deterministicShuffle = true
            }
            val mp = MapParameters().apply {
                this.mapSize = mapSize
                shape = MapShape.hexagonal
                type = MapType.pangaea
                this.seed = seed
            }
            val gameId = "ep-$seedBase-$episode"
            val log = """{"gameId":"$gameId","seed":$seed,"mapSize":"$sizeName",""" +
                """"numMajorCivs":$numMajor,"numCityStates":$numCityStates,""" +
                """"baseRuleset":"${BaseRuleset.Civ_V_GnK.fullName}"}"""
            return ScenarioSpec(GameSetupInfo(gp, mp), gameId, seed, sizeName, numMajor, numCityStates, log)
        }
        throw IllegalStateException("ScenarioGenerator: unable to satisfy guardrails (land/civ >= " +
            "$minLandTilesPerCiv) after $maxAttempts attempts")
    }

    /** Hexagonal tile count ≈ 1 + 3r(r+1); ~55% land for a pangaea. */
    private fun estimatedLandPerCiv(mapSize: MapSize, numCivs: Int): Int {
        val r = mapSize.radius
        val tiles = 1 + 3 * r * (r + 1)
        return (tiles * 0.55 / numCivs).toInt()
    }
}
