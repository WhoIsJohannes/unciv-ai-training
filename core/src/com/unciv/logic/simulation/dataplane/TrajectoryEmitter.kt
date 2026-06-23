package com.unciv.logic.simulation.dataplane

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32

/**
 * Thread-safe by construction: ONE emitter per worker writes ONE shard file. No shared writer ⇒ no
 * locks (avoids the @Synchronized-across-coroutine-suspension hazard). Writes a little-endian,
 * `.npy`-style self-describing container; the footer CRC32 covers the RECORD bytes only (header
 * excluded) so deterministic runs are byte-identical regardless of header timestamps.
 *
 * Crash safety: writes to `<name>.tmp`, atomically renames to `<name>.bin` on clean finalize, and
 * deletes the partial on [abort]. A reader tolerates a truncated trailing record (length-prefixed
 * framing), salvaging preceding steps.
 */
class TrajectoryEmitter(outputDir: File, baseName: String) {

    private val tmpFile = File(outputDir, baseName + ShardFormat.TMP_SUFFIX)
    private val finalFile = File(outputDir, baseName + ShardFormat.SHARD_SUFFIX)
    private val out: BufferedOutputStream
    private val crc = CRC32()
    private var recordCount = 0
    private var finalized = false

    init {
        outputDir.mkdirs()
        out = BufferedOutputStream(FileOutputStream(tmpFile))
    }

    /** Write magic + version + the provenance/layout header JSON. Call exactly once, before records. */
    fun open(headerJson: String) {
        out.write(ShardFormat.MAGIC_BYTES)
        writeU16(SampleSchema.VERSION)
        val hdr = headerJson.toByteArray(Charsets.UTF_8)
        writeU32(hdr.size.toLong())
        out.write(hdr)
    }

    /** Append one step record: `[u32 len | payload]`. The CRC covers the FRAMED bytes (length
     *  prefix + payload) so it equals `zlib.crc32` over the reader's records region. */
    fun record(payload: ByteArray) {
        val prefix = ByteArray(4)
        java.nio.ByteBuffer.wrap(prefix).order(ByteOrder.LITTLE_ENDIAN).putInt(payload.size)
        out.write(prefix)
        out.write(payload)
        crc.update(prefix)
        crc.update(payload)
        recordCount++
    }

    /** Write the footer, flush, close, and atomically publish `.tmp` → `.bin`. */
    fun finalizeShard(): File {
        if (finalized) return finalFile
        writeU32(recordCount.toLong())
        writeU32(crc.value)
        out.flush()
        out.close()
        tmpFile.renameTo(finalFile)
        finalized = true
        return finalFile
    }

    /** On an unrecoverable error: close and delete the partial shard (no orphan, no rename). */
    fun abort() {
        try { out.close() } catch (_: Exception) {}
        tmpFile.delete()
        finalized = true
    }

    /** The determinism contract's `calculateChecksum`: CRC32 over the record bytes written so far. */
    fun calculateChecksum(): Long = crc.value

    val records get() = recordCount

    private val scratch = ByteArray(4)
    private fun writeU16(v: Int) {
        ByteBuffer.wrap(scratch).order(ByteOrder.LITTLE_ENDIAN).putShort((v and 0xFFFF).toShort())
        out.write(scratch, 0, 2)
    }
    private fun writeU32(v: Long) {
        ByteBuffer.wrap(scratch).order(ByteOrder.LITTLE_ENDIAN).putInt((v and 0xFFFFFFFFL).toInt())
        out.write(scratch, 0, 4)
    }
}
