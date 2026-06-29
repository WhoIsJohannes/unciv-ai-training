# Discovery Output — selfplay-v6-replay-buffer

- **Mode**: BUILD
- **Size**: L (cross-system Kotlin↔Python, shard schema VERSION bump 3→4 in lockstep, PPO off-policy correctness, determinism-sensitive)
- **Feature (1-line)**: Add a PPO-correct cross-round recent-window replay buffer (v6) to the Unciv self-play loop — record per-head behavior-policy log-probs at the sampling point and use stored behavior logp as `old_logp` for replayed (off-policy) steps.
- **Context category**: backend/tooling (+ testing). Invariant docs are the feature-workflow's generic product invariants (Mentiora Next.js/FastAPI) — **mostly N/A** to this Kotlin(libGDX)+Python RL repo. Transferable principles applied: fail-loud provenance discipline, SSOT (Kotlin `SampleSchema.VERSION` ↔ Python `SCHEMA_VERSION` lockstep), Python type hints, "tests + build must pass". No UI/auth/PII → Stagehand agent-tests N/A; relevant tests = JUnit `tests/` + Python `pytest`.
- **Domain preset**: data-pipeline (lean) → conditional council lenses likely relevant in Phase 2: **practitioner/RL-correctness** (off-policy importance-sampling validity, determinism), **cost_efficiency** (multi-hour experiment compute). `data_privacy_legal` dropped (N/A).

## Invariant docs loaded
features.md, code-quality.md, architecture.md, testing.md (security.md skimmed — no surface).

## Light scan summary
See `codebase-scan-light.md`. Highlights:
- Greenfield: no existing `behavior_logp`/off-policy/replay-RL code. v5 is on-policy only.
- New `behavior_logp` FIXED f32 block round-trips through the layout-driven reader with **zero reader change**.
- Frozen ONNX contract (CONTRACT_VERSION_*/META_*/INPUT/OUTPUT names) locked by `test_parity.py` — untouched (logp is post-inference).
- Mirror templates: `OnnxPolicyLegalityTest.kt` (chooseWithLogp), `test_microbatch.py::test_microbatch_noop_when_K_ge_N` (K=1 no-op), `test_continual_resume.py` (warm-start/ONNX parity), `test_train_dataset.py::_build_v2_shard` (behavior_logp shard + v3→v4 refusal).
- Experiment driver = `run_v5.sh`; v6 primary arm mirrors v5 **ARM A**: structured/small-rung/Medium/16 rounds/`--micro-batch-steps 0`/`--continual`, ceiling 200 games @ `--eval-seed 4242424`. v5 numbers: 40.7% vs blind 28.9% (58/200), p=0.0069; medium rung 46.6%.

## Open questions → resolved
1. **Execution model** → Full /feature pipeline (worktree `../Unciv-selfplay-v6-replay-buffer`, branch off local master, council reviews, ship PR back to master). Note: `origin` = public yairm210/Unciv; the **local** master is the source of truth (no pull from origin).
2. **Experiment run** → I build + make `./gradlew test` + `pytest` green + verify all fast unit assertions + a Tiny end-to-end replay smoke, then **launch the full Medium re-run in the background this session** (`--replay-window 1` then `4`, resumable) and report AC1/AC2 z/p when it finishes.

## Round 1 Q&A
- Q: full pipeline vs direct-on-master? → **Full /feature pipeline**.
- Q: how to run the multi-hour Medium experiment? → **I launch it in the background now**.

## Experiment arms (derived, not asked — comparable to v5 ARM A)
- Baseline arm: `--rung small --map-size Medium --rounds 16 --gen-games 16 --eval-games 80 --turn-cap 250 --micro-batch-steps 0 --continual --replay-window 1` (must reproduce v5 40.7% within fp tolerance).
- Replay arm: identical but `--replay-window 4`.
- Same `--gen-seed 1000 --eval-seed 999000`; ceiling 200 games @ `--eval-seed 4242424`; z-tests vs blind 28.9%.
