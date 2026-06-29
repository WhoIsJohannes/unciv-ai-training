# Decisions — selfplay-v6-replay-buffer

## D0. Execution model (Phase 1 Step 0)
Full /feature pipeline (worktree + branch off **local** master + council + ship PR back to master).
`origin` is the public yairm210/Unciv; local master is the source of truth (no pull from origin).

## D1. Experiment run (Phase 1 Step 3)
Build + make `./gradlew test` + `pytest` green + verify all fast unit assertions + a Tiny
end-to-end replay smoke, THEN launch the multi-hour Medium re-run in the background this session
(`--replay-window 1` then `4`; resumable via `--resume`); report AC1/AC2 z/p when it completes.

## D2. Intake council roster (Phase 2 Step 5)
Core 6 (skeptic, architect, practitioner, product_manager, qa_testing, security_red_team) +
**domain_fidelity** (the load-bearing lens: PPO off-policy importance-sampling correctness is a
specialist domain where the wrong abstraction — anchoring the ratio to the wrong policy — is a
silent failure; this is precisely the v5-deferred risk v6 must get right) +
**cost_efficiency** (the multi-hour Medium experiment is non-trivial CPU/inference compute).
Dropped: user/business lenses (no UI/auth/PII), finance (local compute, not prod API spend),
ethics_responsible_ai (offline game-AI, no user-impacting harm surface).
Models vendor-diverse across gpt-5.5 / gemini-3.1-pro / deepseek-v4-pro / opus-4.8.

## D3. Replay-health diagnostics (Phase 2 JOIN, intake council critical #2)
ADD 3 read-only diagnostics — `mean_ratio`, `clip_frac` (from the already-computed `ratio` in the
clip branch), `frac_replayed` (threaded from the trainer) — into the per-round `stats` +
`metrics.jsonl`. Touches only `stats.update(...)` (NOT the verbatim clip/value/entropy/objective/
grad-clip math) ⇒ K=1 no-op stays bit-identical. Flagged ➕ ADDED deviation (origin: intake council).
Rationale: off-policy health must be observable during the multi-hour run; mean_ratio≈1 confirms
near-on-policy; at K=1 (stored unused) ratio≡1 trivially.

## D4. Experiment arms (Phase 2 JOIN)
BOTH rungs: structured small-rung AND medium-rung, each K=1 vs K=4 → 4 arms total. Small-rung is the
headline 40.7% comparison (AC1/AC2); medium-rung (v5 46.6%) tests whether replay helps at higher
capacity. Medium rung uses `--micro-batch-steps 256` (as v5 ARM B); small rung uses 0. All 16 rounds,
Medium map, seeds gen=1000/eval=999000, ceiling 200 games @ eval-seed 4242424. Heavier (overnight+),
resumable via --resume. NOT a merge blocker — reported as experimental evidence when complete.

## D5. AC reframing (Phase 2 JOIN, intake council)
- AC1: two-layer. (i) UNIT bit-identity: K=1 → behavior_logp=None → literal v5 recompute path AND
  train_data==data ⇒ training bit-identical given identical shards. (ii) GEN byte-identity: choose
  routes through chooseWithLogp, ONE rng.nextDouble() in identical order ⇒ generation replays
  byte-identical. ⇒ end-to-end K=1 reproduces v5's exact curve (determinism-gated); statistical
  indistinguishability from 40.7% is the fallback acceptance. "1e-4 fp tolerance" applies ONLY to the
  unit stored≈recomputed-warm-net-logp assertion.
- AC2 success: replay (K=4) WINS iff fewer GEN rounds to ≥40.7% (small) / ≥46.6% (medium), OR higher
  200-game ceiling at round 16 with z p<0.05; else NULL (valid result, not a blocker). Sample
  efficiency = fewer GENERATED games (gen dominates wall-clock); K=4 ≈4× training compute/round is the
  acknowledged cost of reuse. Report both framings with z/p.

## D6. Merge gate (Phase 2 JOIN, intake council majors)
PR ships on CODE + all unit assertions + Tiny smoke GREEN. The multi-hour Medium AC1/AC2 numbers are
experimental evidence reported when the background run finishes — NOT a merge blocker (matches D1).

## D7. --resume round-0 exclusion (Phase 2 Step 11 plan council, 🔴 bug)
The in-process deque excludes round 0 (RandomPolicy, maximally off-policy). The --resume refill MUST
too: refill rounds `max(1, start-(K-1)) .. start-1`, NEVER round 0. Globbing `[start-K..start-1]`
would re-admit round-0 data when start ≤ K. Covered by test_replay_resume.py.

## D8. micro-batch on K=4 arms (Phase 2 Step 11 plan council)
K=4 arms set `--micro-batch-steps 256` so the 4× assembled batch is chunked (avoids the whole-batch
OOM risk the small-rung K=4 arm would otherwise hit). Micro-batching is math-identical (v5 AC6) ⇒ does
NOT confound the K=1-vs-K=4 comparison. K=1 small-rung keeps mb=0 (exact v5 ARM A repro); medium rung
mb=256 both. Default `--replay-window` stays 4 per the task spec (council dissent noted; run_v6.sh sets
the window explicitly for every arm, so the default never affects the experiment).

## D9. AC5 determinism — pre-existing whole-shard nondeterminism (build evidence)
`test_same_seed_byte_identical_shards` FAILS on clean master (04f2e27fa, 0 dirty) AND on this branch —
two same-seed `gen random` runs diverge at byte 343 with DIFFERENT shard SIZES (567139 vs 570433), i.e.
the games play out to different lengths. This is deep ENGINE self-play nondeterminism in FROZEN code
v6 never touches (Featurizer/Observation/TrajectoryEmitter/engine). v6 does NOT regress it.
What v6's no-op actually requires is verified by construction: exactly ONE rng.nextDouble() per sample
in identical order, `choose == chooseWithLogp(...).first` (MaskedChoiceLogpTest, green). The unit
`test_replay_noop` (K=1 bit-identical training given identical shards) is the rigorous no-op gate;
AC1's end-to-end claim is the STATISTICAL one (D5) — exact whole-curve reproduction was never literally
possible given the pre-existing engine nondeterminism. Pre-existing engine determinism is OUT OF SCOPE.
