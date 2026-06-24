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
}
