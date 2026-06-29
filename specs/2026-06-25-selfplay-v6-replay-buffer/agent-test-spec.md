# Test spec (test_mode=integration) — selfplay-v6-replay-buffer

RED-first. Runner: `python3 -m pytest python/tests` (system python3 has torch 2.8.0) + `./gradlew test`.

## New test files (RED now → GREEN after Phase 3)
- `tests/.../MaskedChoiceLogpTest.kt` — chooseWithLogp logp == masked-softmax log-prob; empty→(-1,0f);
  `choose == chooseWithLogp.first` (single rng draw / byte-identical replay).
- `python/tests/test_replay_noop.py` — K=1 (stored==round-start recompute) is bit-identical to the
  recompute path (no-op, max|Δw|<1e-6).
- `python/tests/test_offpolicy_equivalence.py` — Kotlin `ln(exps/sum)` == Python `_masked_logp` within
  1e-5 (ratio≈1); stored==recompute ⇒ mean_ratio≈1, clip_frac==0.
- `python/tests/test_behavior_logp_shard.py` — v4 behavior_logp block round-trips to b_logp_*; v3 refuses.
- `python/tests/test_clip_eps_guard.py` — clip_eps=0 + replay raises; clip_eps=0 without replay allowed.
- `python/tests/test_replay_resume.py` — refill excludes round 0 (`max(1,start-(K-1))..start-1`).

## Updated fixtures (forced by VERSION 3→4)
- `python/tests/test_train_dataset.py`, `python/tests/test_v2_units.py`.

## RED verification (pre-implementation)
`pytest test_replay_noop test_offpolicy_equivalence test_behavior_logp_shard test_clip_eps_guard`
→ 6 failed / 1 passed (the pure formula doc test) — failing for the right reason (v6 API absent:
`b_logp_tech` kwarg, `behavior_logp` param, BLOCK, SCHEMA_VERSION still 3). Verified 2026-06-26.
