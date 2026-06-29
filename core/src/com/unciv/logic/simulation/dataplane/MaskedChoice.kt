package com.unciv.logic.simulation.dataplane

import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

/**
 * Pure masked action selection used by the in-JVM `OnnxPolicy`: argmax ([eval]) or softmax-SAMPLE
 * over the LEGAL support only. Returns a legal index, or −1 for an empty support — and NEVER an
 * illegal index. This is the AC2 legality invariant: it is weight-independent and unit-testable
 * without an ONNX session/model, and it mirrors the training-time masked log-prob (only legal
 * actions receive probability). Lives in `core` so the `tests` module (core-only) can verify it.
 */
object MaskedChoice {
    /** Legacy entry point: the chosen legal index only. Delegates to [chooseWithLogp] so there is
     *  exactly ONE code path and ONE `rng.nextDouble()` draw in identical order (v6: generation
     *  replay must stay byte-identical). */
    fun choose(logits: FloatArray, mask: BooleanArray, eval: Boolean, rng: Random): Int =
        chooseWithLogp(logits, mask, eval, rng).first

    /**
     * v6 — returns the chosen legal index AND its log-prob over the SAME legal-masked softmax that
     * the sample draws from: `logp = ln(exps[pos] / sum)` == `log_softmax(masked logits)[chosen]`,
     * the exact quantity the Python trainer's `_masked_logp` scores. Empty support / no decision →
     * `(-1, 0f)` (matching `_masked_logp`'s acted-else-0).
     *
     * Exactly ONE `rng.nextDouble()` draw, only in the sample branch and in the same order as the
     * original `choose` — so routing `choose` through this does NOT perturb the per-(civ,turn) RNG
     * stream. For `eval = true` (argmax point mass) the returned logp is unused (eval games are
     * never recorded); it is still computed naturally over the same masked softmax — not special-cased.
     */
    fun chooseWithLogp(logits: FloatArray, mask: BooleanArray, eval: Boolean, rng: Random): Pair<Int, Float> {
        val legal = mask.indices.filter { it < logits.size && mask[it] }
        if (legal.isEmpty()) return -1 to 0f
        val maxL = legal.maxOf { logits[it] }
        val exps = legal.map { exp((logits[it] - maxL).toDouble()) }
        val sum = exps.sum()
        val pos = if (eval) {
            legal.indices.maxByOrNull { logits[legal[it]] }!!          // argmax over legal (no rng draw)
        } else {
            var r = rng.nextDouble() * sum                            // the single rng draw
            var chosen = legal.size - 1
            for (i in legal.indices) { r -= exps[i]; if (r <= 0.0) { chosen = i; break } }
            chosen
        }
        // Numerically-stable log-softmax form: logp = (logit[chosen] - maxL) - ln(sum). Identical to
        // ln(exps[pos]/sum) for normal values, but a LARGE FINITE negative number (never -Inf) when the
        // chosen action's exp underflowed to 0 — and exactly matches Python's F.log_softmax (parity).
        return legal[pos] to ((logits[legal[pos]] - maxL).toDouble() - ln(sum)).toFloat()
    }
}
