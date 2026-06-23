# Ship Council (Code Review) — onnx-selfplay-v2

37 findings (2 critical, 24 major, 11 minor) over the 1359-line diff. Verdict: APPROVE after fixes.

## Fixed before merge (2 critical + 3 high-value major)
- 🔴 **Native tensor leak on partial construction** → `OnnxPolicy.richTensorsFromArrays` now wraps the
  build in try/catch and closes any partially-created tensors on exception. `parityRunRich` builds
  tensors inside the try so the ORT session closes on a parse/build failure too.
- 🔴 **NaN model poisoning on divergence** → `_optimize_actor_critic` snapshots last-good weights each
  epoch and RESTORES them on a non-finite loss, so a diverged round can never export NaN weights.
- 🟡 **buildRichTensors crash on absent block** → `firstOrNull` + fallback widths; an absent block
  becomes an empty token set (masked-pooled to zero) instead of NoSuchElementException at decision time.
- 🟡 **Provenance gate didn't validate v2 input inventory** → `OnnxPolicy.init` now checks the model's
  `session.inputNames` contains the full expected multi-tensor set for a v2 model.
- 🟡 **Map-size CLI silently fell back to Tiny** → `resolveMapSize` now errors on an unknown name
  (a typo fails loud instead of silently corrupting a Medium experiment).

## Deferred / documented (minor + edge)
- Acceptance script continues after a failed variant: already `log FAILED` per variant; the real run had
  no failures. (driver script; not shipped code.)
- Dead `.pt` checkpoint load: `.pt` is written for observability; resume uses `.onnx` + curve.csv. Intentional.
- Zero-step training round still exports: edge case (a round with 0 learner steps — can't happen at 24 games).
- GAE V=0 bootstrap for turn-capped (truncated) games: score-leader-on-timeout assigns a real ±1 terminal,
  so the terminal-only reward is meaningful; true-truncation bootstrapping is a known RL subtlety, future work.
- Rich round materialized as one dense padded batch: worked on Medium (no OOM); micro-batching is future work.
- Observation-normalizer + hyperparam-magic-constants: filed in cleanup-opportunities.md.

All fixes re-verified: desktop compiles; parity (blind+rich) + unit + gae GREEN; rich-critic live smoke
(init gate accepts valid v2 model, 0 illegal actions, no divergence).
