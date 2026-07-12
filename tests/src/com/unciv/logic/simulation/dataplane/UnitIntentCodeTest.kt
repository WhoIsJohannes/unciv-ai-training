package com.unciv.logic.simulation.dataplane

import com.unciv.testing.GdxTestRunner
import com.unciv.testing.TestGame
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith

/**
 * v8: the per-unit INTENT head. Mirrors [ConstructionCodeTest] for the construction head — pins the
 * decode inverse and the mask width/round-trip, but over the [UnitIntent] ordinal space (NO
 * building/unit-style offset: the intent MASK space IS the enum ordinal space directly).
 *
 * The control path samples an index into [LegalActionMasks.unitIntentMask] and decodes it with
 * [Vocab.unitIntentId] to dispatch/record the executed intent, so an off-by-one here would silently
 * mislabel one intent as another (recorded != applied).
 */
@RunWith(GdxTestRunner::class)
class UnitIntentCodeTest {

    /**
     * v8: [Vocab.unitIntentId] MUST invert the 0-INDEXED intent-MASK space, which is exactly the
     * [UnitIntent] ordinal space (the layout of [LegalActionMasks.unitIntentMask] and the per-unit net
     * logits). Out-of-range / negative → null (the abstain / no-decision sentinel).
     */
    @Test
    fun unitIntentIdInvertsTheZeroIndexedMaskSpace() {
        val g = TestGame()
        val vocab = Vocab(g.ruleset)
        for (idx in 0 until UnitIntent.COUNT)
            assertEquals("unitIntentId must invert the 0-indexed intent mask/ordinal space",
                UnitIntent.entries[idx].name, vocab.unitIntentId(idx))
        assertEquals("negative index → null (abstain sentinel)", null, vocab.unitIntentId(-1))
        assertEquals("index == COUNT is out of range → null", null, vocab.unitIntentId(UnitIntent.COUNT))
    }

    /**
     * v8: round-trip the mask itself — [LegalActionMasks.unitIntentMask] is exactly [UnitIntent.COUNT]
     * wide and every set bit decodes (via [Vocab.unitIntentId]) to a real intent name. A LAND-MILITARY
     * unit ALWAYS has ≥1 legal intent (the unconditional ACCOMPANY/ATTACK/ADVANCE_CLOSE_ENEMY/EXPLORE
     * rungs), so the head never abstains for a modeled unit.
     */
    @Test
    fun unitIntentMaskWidthAndDecodes() {
        val g = TestGame(); g.makeHexagonalMap(4)
        val civ = g.addCiv()
        g.addCity(civ, g.getTile(0, 0))
        val vocab = Vocab(g.ruleset)
        // GnK always has a Warrior (land-military). Place one on the capital tile.
        Assume.assumeTrue("GnK ruleset must provide a Warrior", g.ruleset.units.containsKey("Warrior"))
        val unit = g.addUnit("Warrior", civ, g.getTile(0, 0))
        assertTrue("Warrior must be a land-military unit", unit.baseUnit.isLandUnit && unit.isMilitary())

        val mask = LegalActionMasks.unitIntentMask(unit)
        assertEquals("intent mask width must be UnitIntent.COUNT", UnitIntent.COUNT, mask.size)
        var legal = 0
        for (k in mask.indices) if (mask[k]) {
            legal++
            assertTrue("legal intent slot $k must decode to a real intent id", vocab.unitIntentId(k) != null)
        }
        assertTrue("a land-military unit must have ≥1 legal intent (the unconditional rungs)", legal > 0)
    }

    /**
     * v8: a non-(land-military) unit is NOT modeled — [LegalActionMasks.unitIntentMask] returns an
     * all-false row (still exactly [UnitIntent.COUNT] wide) so the head abstains (recorded intent = −1).
     */
    @Test
    fun nonLandMilitaryUnitHasEmptyMask() {
        val g = TestGame(); g.makeHexagonalMap(4)
        val civ = g.addCiv()
        g.addCity(civ, g.getTile(0, 0))
        Assume.assumeTrue("GnK ruleset must provide a civilian Worker", g.ruleset.units.containsKey("Worker"))
        val worker = g.addUnit("Worker", civ, g.getTile(0, 0))
        assertTrue("Worker must NOT be a land-military unit",
            !(worker.baseUnit.isLandUnit && worker.isMilitary()))

        val mask = LegalActionMasks.unitIntentMask(worker)
        assertEquals("mask width is always UnitIntent.COUNT", UnitIntent.COUNT, mask.size)
        assertTrue("a non-(land-military) unit must have an all-false intent mask", mask.none { it })
    }
}
