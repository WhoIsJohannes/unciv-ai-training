package com.unciv.logic.simulation.dataplane

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32

/**
 * Growable LITTLE-ENDIAN byte buffer for building shard records. The JVM default is big-endian;
 * every multi-byte write here is explicitly little-endian so the pure-Python/numpy reader
 * (`np.frombuffer(..., '<f4')`) reads it correctly on any architecture.
 */
class LeBuffer(initialCapacity: Int = 1024) {
    private var buf = ByteArray(initialCapacity)
    private var len = 0

    private fun ensure(extra: Int) {
        if (len + extra > buf.size) buf = buf.copyOf(maxOf(buf.size * 2, len + extra))
    }

    fun u8(v: Int): LeBuffer { ensure(1); buf[len++] = (v and 0xFF).toByte(); return this }

    fun u16(v: Int): LeBuffer {
        ensure(2)
        ByteBuffer.wrap(buf, len, 2).order(ByteOrder.LITTLE_ENDIAN).putShort((v and 0xFFFF).toShort())
        len += 2
        return this
    }

    fun i32(v: Int): LeBuffer {
        ensure(4)
        ByteBuffer.wrap(buf, len, 4).order(ByteOrder.LITTLE_ENDIAN).putInt(v)
        len += 4
        return this
    }

    fun u32(v: Long): LeBuffer = i32((v and 0xFFFFFFFFL).toInt())

    fun f32(v: Float): LeBuffer {
        ensure(4)
        ByteBuffer.wrap(buf, len, 4).order(ByteOrder.LITTLE_ENDIAN).putFloat(v)
        len += 4
        return this
    }

    /** Append a fixed-length f32 array. */
    fun f32s(values: FloatArray): LeBuffer { for (v in values) f32(v); return this }

    fun rawBytes(src: ByteArray): LeBuffer { ensure(src.size); System.arraycopy(src, 0, buf, len, src.size); len += src.size; return this }

    val size get() = len
    fun toByteArray(): ByteArray = buf.copyOf(len)
}

/** Shard container constants + the checksum used by the determinism contract + Python reader. */
object ShardFormat {
    val MAGIC_BYTES: ByteArray = SampleSchema.MAGIC.toByteArray(Charsets.US_ASCII)

    /** Suffix for the in-progress shard; atomically renamed to `.bin` on clean finalize. */
    const val TMP_SUFFIX = ".tmp"
    const val SHARD_SUFFIX = ".bin"

    /**
     * `calculateChecksum` — CRC32 over the concatenated RECORD bytes ONLY (the header, which may
     * carry wall-clock / hostname / git-SHA, is excluded). Mirrored in Python by `zlib.crc32`.
     */
    fun calculateChecksum(recordBytes: ByteArray): Long {
        val crc = CRC32()
        crc.update(recordBytes)
        return crc.value
    }
}
