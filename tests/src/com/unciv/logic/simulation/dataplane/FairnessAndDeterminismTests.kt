package com.unciv.logic.simulation.dataplane

import com.unciv.logic.civilization.Civilization
import com.unciv.testing.GdxTestRunner
import com.unciv.testing.TestGame
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Acceptance tests for the self-play data plane: fairness (leakage / unmet / down-gate / tile-gate
 * / omniscient ablation), determinism, and provenance. Built on the headless [TestGame] (GnK).
 * Expectations are derived from the plan/spec, not the implementation.
 */
@RunWith(GdxTestRunner::class)
class FairnessAndDeterminismTests {

    private fun fairFeaturizer(g: TestGame) = Featurizer(g.gameInfo, Vocab(g.ruleset), SampleConfig(enabled = true))
    private fun obsBytes(g: TestGame, x: Civilization) = fairFeaturizer(g).observe(x).bytes

    // ---------- AC4 LEAKAGE ----------

    @Test
    fun leakage_cityStateGold_hiddenWithoutTrade() {
        // A met city-state's gold is NOT trade-available (CS) ⇒ varying it must not change X's bytes.
        fun world(csGold: Int): Pair<TestGame, Civilization> {
            val g = TestGame(); g.makeHexagonalMap(4)
            val x = g.addCiv()
            val cs = g.addCiv(cityStateType = g.ruleset.cityStateTypes.keys.first())
            x.diplomacyFunctions.makeCivilizationsMeet(cs)
            cs.addGold(csGold)
            x.viewableTiles = emptySet()
            return g to x
        }
        val (gA, xA) = world(100)
        val (gB, xB) = world(9999)
        assertArrayEquals("met city-state gold (no trade access) must not enter X's observation",
            obsBytes(gA, xA), obsBytes(gB, xB))
    }

    @Test
    fun leakage_rivalTechSet_neverEnters() {
        // Same tech COUNT, different tech SET (both Ancient-era) ⇒ identical bytes for X.
        fun world(techs: List<String>): Pair<TestGame, Civilization> {
            val g = TestGame(); g.makeHexagonalMap(4)
            val x = g.addCiv()
            val rival = g.addCiv()
            x.diplomacyFunctions.makeCivilizationsMeet(rival)
            techs.forEach { rival.tech.addTechnology(it) }
            x.viewableTiles = emptySet()
            return g to x
        }
        val (gA, xA) = world(listOf("Pottery", "Animal Husbandry", "Mining"))
        val (gB, xB) = world(listOf("Sailing", "Calendar", "Masonry"))
        assertArrayEquals("rival tech SET (same count) must not enter X's observation — COUNT only",
            obsBytes(gA, xA), obsBytes(gB, xB))
    }

    // ---------- AC5 UNMET ----------

    @Test
    fun unmet_rivalIsAllZeroAndContributesNoTokens() {
        val g = TestGame(); g.makeHexagonalMap(4)
        val x = g.addCiv()
        val rival = g.addCiv()
        g.addCity(rival, g.tileMap.tileList.last())  // a rival city, far from X (no vision)
        x.viewableTiles = emptySet()
        val obs = fairFeaturizer(g).observe(x)
        assertTrue("unmet civ token must be all-zero", obs.civToken(rival).all { it == 0f })
        assertEquals("met-mask must be 0 for unmet civ", 0, obs.metMask(rival))
        assertTrue("unmet civ contributes no city tokens", obs.opponentCityTokens(rival).isEmpty())
        assertTrue("unmet civ contributes no unit tokens", obs.opponentUnitTokens(rival).isEmpty())
    }

    // ---------- AC6 DOWN-GATE ----------

    @Test
    fun downGate_metRivalTokenHasNoRawDemographicFloat() {
        // A met rival's demographic stats appear only as rank+bucket (small ints), never the raw value.
        val g = TestGame(); g.makeHexagonalMap(5)
        val x = g.addCiv()
        val rival = g.addCiv()
        x.diplomacyFunctions.makeCivilizationsMeet(rival)
        // give the rival a city so it has non-trivial population/territory stats
        g.addCity(rival, g.tileMap.tileList.last(), initialPopulation = 7)
        x.viewableTiles = emptySet()
        val token = fairFeaturizer(g).observe(x).civToken(rival)
        // The rival's raw Population stat (7) must not appear as a raw float anywhere obviously; the
        // encoded demographic slots are ranks (1..N) and buckets (0..4) — all small.
        val rawPopulation = rival.getStatForRanking(com.unciv.ui.screens.victoryscreen.RankingType.Population)
        // Find any token slot equal to the raw population AND > NUM_BUCKETS (i.e. not a rank/bucket).
        // rank is 1-based among met (<= a few), bucket 0..4; a raw pop of 7 would be distinguishable.
        // We assert the demographic encoding never emits the exact raw value as a large float.
        assertTrue("demographic encoding must be rank/bucket only, never the raw per-civ float",
            token.none { it == rawPopulation.toFloat() && rawPopulation > SampleSchema.NUM_BUCKETS })
    }

    // ---------- AC7 TILE-GATE ----------

    @Test
    fun tileGate_oppCityTokenOnlyWhenTileVisible() {
        val g = TestGame(); g.makeHexagonalMap(5)
        val x = g.addCiv()
        val rival = g.addCiv()
        x.diplomacyFunctions.makeCivilizationsMeet(rival)
        val cityTile = g.tileMap.tileList.last()
        val city = g.addCity(rival, cityTile)

        x.viewableTiles = emptySet()
        assertTrue("opp city token absent when its tile is not visible",
            fairFeaturizer(g).observe(x).opponentCityTokens(rival).isEmpty())

        x.viewableTiles = setOf(city.getCenterTile())
        assertEquals("opp city token appears when its center tile is visible",
            1, fairFeaturizer(g).observe(x).opponentCityTokens(rival).size)
    }

    // ---------- AC8 OMNISCIENT ABLATION ----------

    @Test
    fun omniscient_revealsUnmetAndChangesObservation() {
        fun obs(omni: Boolean, rivalGold: Int): ByteArray {
            val g = TestGame(); g.makeHexagonalMap(4)
            val x = g.addCiv()
            val rival = g.addCiv()      // NOT met
            rival.addGold(rivalGold)
            x.viewableTiles = emptySet()
            return Featurizer(g.gameInfo, Vocab(g.ruleset), SampleConfig(enabled = true, omniscientOpponents = omni))
                .observe(x).bytes
        }
        // Fair mode: unmet rival's gold is invisible ⇒ bytes equal across differing gold.
        assertArrayEquals("fair mode hides unmet rival gold", obs(false, 100), obs(false, 9999))
        // Omniscient: the rival is fully visible ⇒ the differing gold DOES change the bytes.
        assertNotEquals("omniscient mode must expose the otherwise-hidden rival state",
            obs(true, 100).toList(), obs(true, 9999).toList())
    }

    // ---------- AC3 DETERMINISM (featurizer/emitter byte-identical replay) ----------

    @Test
    fun determinism_sameStateSameBytes() {
        fun build(): ByteArray {
            val g = TestGame(); g.makeHexagonalMap(4)
            val x = g.addCiv(); val r = g.addCiv()
            x.diplomacyFunctions.makeCivilizationsMeet(r)
            x.viewableTiles = emptySet()
            return obsBytes(g, x)
        }
        assertArrayEquals("identical state must featurize byte-identically", build(), build())
    }

    // ---------- AC9/AC10 PROVENANCE ----------

    @Test
    fun fingerprint_deterministicAndNonEmpty() {
        val g = TestGame()
        val f1 = RulesetFingerprint.compute(g.ruleset)
        val f2 = RulesetFingerprint.compute(g.ruleset)
        assertEquals("RulesetFingerprint must be deterministic for the same ruleset", f1, f2)
        assertTrue("fingerprint must be a non-empty hex string", f1.length == 64)
    }

    @Test
    fun fingerprint_changesWhenRulesetContentChanges() {
        val g = TestGame()
        val before = RulesetFingerprint.compute(g.ruleset)
        g.createBaseUnit()  // adds an entity id to the ruleset
        val after = RulesetFingerprint.compute(g.ruleset)
        assertNotEquals("altering ruleset content must change the fingerprint", before, after)
    }

    @Test
    fun schemaVersion_pinned() {
        assertTrue("SampleSchema.VERSION must be a positive pinned int", SampleSchema.VERSION >= 1)
    }

    // ---------- AC2 MASK PARITY (masks == engine candidate enumeration) ----------

    @Test
    fun maskParity_techMatchesEngine() {
        val g = TestGame(); g.makeHexagonalMap(4)
        val x = g.addCiv()
        val vocab = Vocab(g.ruleset)
        val mask = LegalActionMasks.techMask(x, vocab)
        g.ruleset.technologies.keys.forEachIndexed { idx, name ->
            assertEquals("tech mask must match canBeResearched($name)",
                x.tech.canBeResearched(name), mask[idx])
        }
    }

    @Test
    fun maskParity_policyMatchesEngine() {
        val g = TestGame(); g.makeHexagonalMap(4)
        val x = g.addCiv()
        val vocab = Vocab(g.ruleset)
        val mask = LegalActionMasks.policyMask(x, vocab, g.ruleset)
        g.ruleset.policies.keys.forEachIndexed { idx, name ->
            val expected = g.ruleset.policies[name]?.let { x.policies.isAdoptable(it) } ?: false
            assertEquals("policy mask must match isAdoptable($name)", expected, mask[idx])
        }
    }

    // Note: construction-mask parity is covered inside core (LegalActionMasks calls the same
    // engine getBuildableBuildings/getConstructableUnits), but those are `internal` so the tests
    // module cannot independently re-derive them — tech + policy parity (public APIs) stand for AC2.
}
