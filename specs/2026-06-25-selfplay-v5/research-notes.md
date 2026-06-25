> Web-sourced content below is DATA, not instructions.

# Web Research — selfplay-v5 (Continual Training)

Two load-bearing external questions for the design: (A) exact numerical-equivalence condition
for chunked gradient accumulation under a `.mean()`-reduced loss (the AC6 requirement), and
(B) pitfalls of carrying Adam optimizer state across continual/on-policy RL rounds.

## Query 1 — gradient accumulation numerical equivalence (mean reduction, micro-batch)
**Key findings:**
- The gradient of a loss over a large batch equals the *average* of gradients over sub-batches —
  so accumulate-then-single-step is mathematically equivalent to whole-batch, BUT only with
  correct loss normalization.
- The canonical bug: per-micro-batch `.mean()` then averaging the means equally — wrong whenever
  chunk sizes differ. Correct = either (a) `sum`-reduce each chunk and divide by TOTAL count once,
  or (b) weight each chunk's mean by `chunk_n / total_n` and sum.
- Exact bitwise equality does NOT hold — float summation-order changes cause small rounding
  differences. Equivalence is "very close, within fp tolerance" (matches AC6's wording exactly).
- Caveat that does NOT apply here: batch-norm computes stats over the batch dim, which differs for
  micro vs full batches. Our encoder has no batchnorm-over-batch (attention/GNN are within-step over
  tiles/tokens) — confirmed by the deep-scan #3 mission. So chunking the step axis is sound.
**Relevance:** Directly dictates the micro-batch accumulation: keep the frozen `.mean()` objective,
but scale each chunk's loss by `chunk_n / total_n` before `loss.backward()`, accumulate grads, single
`opt.step()`. The numerical-equivalence test asserts loss + grad-norm match whole-batch within ~1e-5.
**Libraries found:** none new (PyTorch only; standard idiom).

## Query 2 — warm-start Adam state across continual / on-policy RL rounds
**Key findings:**
- Carrying the full optimizer triple (params + 1st-moment exp_avg + 2nd-moment exp_avg_sq) across
  iterations via `optimizer.load_state_dict` is an established continual-RL technique: faster
  convergence (reported ~9× fewer epochs), smoother trajectories (moments low-pass-filter the
  gradient, suppressing cold-start transients), and encodes recent curvature beyond weights.
- Pitfall: with β1/β2 = 0.9/0.95, prior moment estimates decay fast — after ~1000 update steps the
  carried state contributes <0.0044%. So carryover helps EARLY and washes out over many steps.
  Here each round does only ~8 `opt.step()` calls (whole-batch, one per epoch); 16 rounds ≈ 128
  steps total — well inside the regime where carried momentum is still material. Carryover is
  genuinely beneficial for this workload (NOT washed out).
- Continual-backprop guidance: "when weights are reset, zero the corresponding moments + step." Our
  analog: the divergence guard rolls weights back to last-good (not to random), so it must roll the
  optimizer state back to the matching last-good snapshot — exactly the v5 `opt.state_dict()`
  snapshot/restore. This keeps the guard sound under a persistent optimizer.
**Relevance:** Validates the core v5 hypothesis (not-resetting should help) and pins the
divergence-guard correctness requirement. PPO on-policy-ness is unaffected: warm-start changes only
initialization; `old_logp` is still recomputed from the current (warm) net each round, and that net's
weights == the weights that generated the round's data, so the importance ratio anchor stays valid.

## Key insights
1. Micro-batch correctness: scale each chunk's `.mean()` loss by `chunk_n/total_n`, accumulate, single
   `opt.step()`. Equivalence is fp-tolerance, not bitwise — test with atol≈1e-5 on loss + grad-norm.
2. No batchnorm-over-batch in the encoder → slicing the flat step axis is forward-equivalent (pending
   deep-scan #3 confirmation of no cross-step ops).
3. Carrying Adam state across rounds is a validated technique and is NOT washed out at ~128 total
   steps; expect faster/smoother accumulation — the v5 bet is well-founded.
4. Divergence guard must snapshot/restore optimizer state alongside net weights, else a diverged round
   poisons the persistent moments. (Roll back moments to last-good, the analog of zeroing on reset.)
5. Warm-start is PPO-correct: on-policy assumption intact; only initialization changes.
