package com.unciv.logic.simulation.dataplane

import com.unciv.testing.GdxTestRunner
import com.unciv.testing.TestGame
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * AC7: the v3 construction-namespace fix. The city-token `current_construction` field encodes a name
 * over the DISJOINT building∪unit space whose width is reserved as buildingCount+unitCount. The old
 * code wrote `(building(x) ?: unit(x)) + 1` with NO buildingCount offset, so building#k and unit#k
 * (same local index k) collided onto the same code. [Vocab.constructionCode] must give every building
 * and every unit a DISTINCT non-zero code.
 */
@RunWith(GdxTestRunner::class)
class ConstructionCodeTest {

    @Test
    fun constructionCodesAreInjectiveOverBuildingsAndUnits() {
        val g = TestGame()
        val vocab = Vocab(g.ruleset)
        val seen = HashMap<Int, String>()
        val collisions = ArrayList<String>()
        for (name in g.ruleset.buildings.keys + g.ruleset.units.keys) {
            val code = vocab.constructionCode(name)
            assertTrue("construction code for known '$name' must be > 0 (0 is reserved for none)", code > 0)
            val prev = seen.put(code, name)
            if (prev != null) collisions.add("'$name' and '$prev' both → $code")
        }
        assertTrue("construction codes must be collision-free across buildings∪units: $collisions",
            collisions.isEmpty())
    }

    @Test
    fun buildingAndUnitAtSameLocalIndexDoNotCollide() {
        // The exact original bug: a building and a unit at the SAME vocab index k both encoded to k+1.
        val g = TestGame()
        val vocab = Vocab(g.ruleset)
        val building = g.ruleset.buildings.keys.firstOrNull { vocab.building(it) >= 0 }
        val unit = g.ruleset.units.keys.firstOrNull { vocab.unit(it) == vocab.building(building ?: "") }
        if (building != null && unit != null) {
            assertNotEquals(
                "building#k and unit#k (same local index) must encode to different construction codes",
                vocab.constructionCode(building), vocab.constructionCode(unit),
            )
        }
        // unknown name → 0 (no/empty construction)
        assertEquals(0, vocab.constructionCode("___not_a_real_construction___"))
    }

    /**
     * v7: [Vocab.constructionId] MUST invert the 0-INDEXED construction-MASK space (the layout of
     * [LegalActionMasks.constructionMask] and the per-city net logits) — NOT the 1-indexed
     * [Vocab.constructionCode]. The control path samples an index into the mask and decodes it with
     * constructionId, so an off-by-one here would silently mislabel buildings as units (recorded != applied).
     */
    @Test
    fun constructionIdInvertsTheZeroIndexedMaskSpace() {
        val g = TestGame()
        val vocab = Vocab(g.ruleset)
        // buildings occupy mask indices [0, buildingCount): constructionId(building idx) == that building.
        for (name in g.ruleset.buildings.keys) {
            val idx = vocab.building(name)
            if (idx >= 0) assertEquals("constructionId must invert the building mask index", name, vocab.constructionId(idx))
        }
        // units occupy mask indices [buildingCount, buildingCount+unitCount): offset by buildingCount.
        for (name in g.ruleset.units.keys) {
            val idx = vocab.unit(name)
            if (idx >= 0) assertEquals("constructionId must invert the unit mask index (offset by buildingCount)",
                name, vocab.constructionId(vocab.buildingCount + idx))
        }
        // out-of-range / negative → null (the abstain / no-decision sentinel).
        assertEquals(null, vocab.constructionId(-1))
        assertEquals(null, vocab.constructionId(vocab.buildingCount + vocab.unitCount))
    }

    /**
     * v7: round-trip the mask itself — [LegalActionMasks.constructionMask] is exactly constrW wide and
     * every set bit decodes (via constructionId) to a real building/unit id. Guards the per-city head's
     * action-space width and the recorded-action → applied-construction mapping.
     */
    @Test
    fun constructionMaskWidthAndRoundTrip() {
        val g = TestGame(); g.makeHexagonalMap(4)
        val civ = g.addCiv()
        val city = g.addCity(civ, g.getTile(0, 0))
        val vocab = Vocab(g.ruleset)
        val mask = LegalActionMasks.constructionMask(city, vocab)
        assertEquals("mask width must be buildingCount+unitCount", vocab.buildingCount + vocab.unitCount, mask.size)
        var legal = 0
        for (k in mask.indices) if (mask[k]) {
            legal++
            assertTrue("legal mask slot $k must decode to a real construction id", vocab.constructionId(k) != null)
        }
        assertTrue("a founded city must have ≥1 legal construction", legal > 0)
    }
}
