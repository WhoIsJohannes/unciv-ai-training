package com.unciv.logic.simulation.dataplane

import com.unciv.logic.civilization.Civilization

/** FIXED = a constant-length array. VARIABLE = a count-prefixed list of [perItem]-wide rows
 *  (only the PRESENT entities are stored; padding-to-cap is the data loader's job, not storage). */
enum class BlockKind { FIXED, VARIABLE }

/**
 * A deciding-civ observation as a list of typed blocks (the shard record payload + header layout),
 * plus structured accessors used by the fairness tests. `bytes` is the canonical serialization the
 * leakage test compares for byte-equality. Storing only present entities (VARIABLE blocks) + u8
 * dtypes for masks/spatial keeps shards small — a Tiny game is hundreds of KB, not megabytes.
 */
class Observation(
    val blocks: List<Block>,
    private val presentCivIndex: Map<String, Int>,
    private val civTokenWidth: Int,
    private val oppCityOwners: List<String>,
    private val oppCityValues: List<FloatArray>,
    private val oppUnitOwners: List<String>,
    private val oppUnitValues: List<FloatArray>,
    /** True if any entity overflowed its cap (diagnostic only — never a feature fed to the policy). */
    val overflow: Boolean,
) {
    /** values are held as floats; serialized per [dtype] ("<f4" or "<u1"). For VARIABLE blocks the
     *  row count is `values.size / perItem`. */
    class Block(val name: String, val dtype: String, val kind: BlockKind, val perItem: Int, val values: FloatArray)

    val bytes: ByteArray by lazy {
        val b = LeBuffer(blocks.sumOf { it.values.size } + 64)
        for (blk in blocks) writeBlock(b, blk)
        b.toByteArray()
    }

    fun block(name: String): FloatArray = blocks.first { it.name == name }.values

    /** Per-block descriptor for the shard header / schema.json. */
    fun layout(): List<Map<String, Any>> = blocks.map {
        mapOf("name" to it.name, "dtype" to it.dtype,
            "kind" to if (it.kind == BlockKind.VARIABLE) "var" else "fixed",
            "perItem" to it.perItem, "len" to it.values.size)
    }

    fun civToken(civ: Civilization): FloatArray {
        val idx = presentCivIndex[civ.civID] ?: return FloatArray(0)
        val w = civTokenWidth
        return block("civ_tokens").copyOfRange(idx * w, idx * w + w)
    }

    fun metMask(civ: Civilization): Int {
        val idx = presentCivIndex[civ.civID] ?: return 0
        return block("civ_tokens")[idx * civTokenWidth].toInt()
    }

    fun opponentCityTokens(civ: Civilization): List<FloatArray> =
        oppCityOwners.indices.filter { oppCityOwners[it] == civ.civID }.map { oppCityValues[it] }

    fun opponentUnitTokens(civ: Civilization): List<FloatArray> =
        oppUnitOwners.indices.filter { oppUnitOwners[it] == civ.civID }.map { oppUnitValues[it] }

    companion object {
        /** Serialize one block into [b]: VARIABLE blocks are u16-count-prefixed; values per dtype. */
        fun writeBlock(b: LeBuffer, blk: Block) {
            if (blk.kind == BlockKind.VARIABLE) {
                val count = if (blk.perItem > 0) blk.values.size / blk.perItem else 0
                b.u16(count)
            }
            if (blk.dtype == SampleSchema.DT_U8) for (v in blk.values) b.u8(v.toInt().coerceIn(0, 255))
            else b.f32s(blk.values)
        }
    }
}
