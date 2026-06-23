package com.unciv.logic.simulation.dataplane

import com.unciv.models.ruleset.Ruleset
import java.security.MessageDigest

/**
 * Deterministic content hash over the loaded ruleset — all entity ids + the enum/vocab order, in
 * the SAME canonical order [Vocab] uses to build indices. Exposed as the dataset's
 * `RulesetFingerprint`: it converts silent balance/content drift into a CAUGHT mismatch even when
 * [SampleSchema.VERSION] is unchanged. Stable across runs for identical ruleset content.
 *
 * Encoding is length-prefixed (no separator char) so it is unambiguous regardless of what
 * characters appear inside entity ids.
 */
object RulesetFingerprint {

    fun compute(ruleset: Ruleset): String {
        val sb = StringBuilder()
        sb.append("schemaVersion=").append(SampleSchema.VERSION).append('\n')
        for ((name, ids) in Vocab.canonicalSections(ruleset)) {
            sb.append(name).append('=').append(ids.size).append('\n')
            for (id in ids) sb.append(id.length).append(':').append(id)
            sb.append('\n')
        }
        val digest = MessageDigest.getInstance("SHA-256").digest(sb.toString().toByteArray(Charsets.UTF_8))
        return digest.joinToString("") { "%02x".format(it) }
    }
}
