# Intake Council Triage — selfplay-v6-replay-buffer

8 reviewers, 34 findings (7 critical / 21 major / 6 minor), ~$0.57. Reviewers saw the
SUMMARY (`discovery-output.md`), not the full task spec — so many "missing X" findings are
already decided by the task. Triaged into ACCEPT (fold into plan) / ALREADY-HANDLED (make
explicit) / REJECT (out of scope or wrong).

## ACCEPT — genuine sharpenings to fold into plan.md
- **A1 (criticals #2, SLO/determinism majors) — add replay-health metrics.** Emit per-round
  `mean_ratio`, `clip_frac`, `frac_replayed` into stats + metrics.jsonl so off-policy health is
  observable in-flight during the multi-hour run (and doubles as a determinism sanity signal:
  at K=1 the stored logp is unused so ratio≡1; at K≥2 mean_ratio≈1 confirms near-on-policy).
  LOW cost, HIGH value. The ONLY code addition beyond the task's letter.
- **A2 (critical #6 — best finding) — reframe AC1 precisely.** "within fp tolerance" is wrong
  for a 200-game discrete win-rate. The RIGOROUS claim is two-layer:
  (i) UNIT: with `--replay-window 1` → behavior_logp=None → literal v5 recompute path AND
      train_data==data ⇒ training is **bit-identical** given identical shards (no-op test);
  (ii) GEN: `choose` routes through `chooseWithLogp` with exactly ONE rng.nextDouble() in
      identical order ⇒ generation replays **byte-identical** to v5 (the extra recorded block
      does not perturb the RNG) ⇒ v4 shards carry the same trajectories as v5's v3 shards.
  ⇒ End-to-end K=1 reproduces v5's **exact** games/curve if determinism holds; report it as
  exact-reproduction (determinism-gated), with statistical-indistinguishability from 40.7% as
  the fallback acceptance if any residual nondeterminism exists. Drop the "fp tolerance" phrase
  for the end-to-end; keep 1e-4 fp tolerance ONLY for the unit stored≈recomputed assertion.
- **A3 (criticals #4/#5 — AC2 threshold) — quantify success.** Replay (K=4) WINS iff
  (a) it reaches ≥40.7% in strictly FEWER generation rounds than the K=1 arm, OR
  (b) at round 16 its 200-game ceiling exceeds the K=1 arm's by a z-test p<0.05.
  Otherwise NULL (replay didn't help) — a valid scientific result, NOT a merge blocker.
- **A4 (compute-efficiency major) — name the metric honestly.** Sample efficiency = fewer
  GENERATED games (generation dominates wall-clock); K=4 trains 8 epochs over 4× data per round
  (≈4× training compute) — acknowledged as the expected cost of reuse. Report BOTH framings.
- **A5 (scope majors) — explicit out-of-scope + minimalism.** `deque(maxlen=K)`, ONE FIXED f32
  block, TWO known heads — NO sampler/priority/persistence framework, NO registry. Add a
  "What this is NOT" section. Schema 3→4: v3 shards intentionally refuse (perishable); v6 runs
  are fresh; rollback = revert branch / use v5 binary; resume only within a v6 run.
- **A6 (edge-case majors) — clarify the off-policy ratio edges in plan.md:**
  - masks are stored per-step (generation-time, frozen) ⇒ the action is ALWAYS legal under its
    own stored mask ⇒ gather is valid, no illegal-action case.
  - joint logp = `b_logp_tech + b_logp_policy` (SUM); non-acting head = 0 — exactly mirrors the
    current `old_logp` recompute `_masked_logp(tech)+_masked_logp(policy)`.
  - `logratio.clamp(±20)` bounds the ratio BOTH directions (near-zero-prob action → very negative
    logp → ratio→0, not exploding; clamp + clip cap the other direction).
  - generation sampling is PURE masked-softmax (no temperature/Dirichlet) ⇒ recorded logp == the
    exact sampling log-prob; no hidden exploration noise.
- **A7 (gate majors) — decouple merge from experiment.** PR ships on CODE + unit assertions +
  Tiny smoke GREEN; the multi-hour Medium AC1/AC2 numbers are experimental evidence reported when
  the background run finishes — NOT a merge blocker (matches the user's "background" choice).
- **A8 (Tiny smoke major) — define it.** Tiny smoke = `--map-size Tiny --rounds 2 --gen-games 2
  --eval-games 2 --replay-window 4 --variant structured --rung small`; PASS = runs to completion,
  round-1 trains on current∪round-0-excluded data (round 0 excluded), v4 shards written + reload,
  metrics show frac_replayed>0 at round≥2, no crash/NaN.

## ALREADY-HANDLED by the task (make explicit in plan; not new work)
- Criticals #1/#7 "GAE/value for off-policy" → the task ALREADY mandates val0 = current-net
  forward every round; GAE recomputed per-trajectory under the CURRENT critic. **Foreground this
  as the headline correctness invariant in plan.md** (2 reviewers raised it — it must be loud).
- "K=1 exact vs JVM/PyTorch float divergence" → at K=1 the stored Kotlin logp is UNUSED
  (window-gated → None ⇒ recompute path). Zero fp-divergence exposure at K=1. The stored logp
  only matters at K≥2, where small deviations are exactly what the clip handles + the 1e-4 unit
  test bounds.

## REJECT — out of scope / wrong for this repo
- "Shard-corruption attacker injects NaN/Inf/positive logp" → threat model N/A (shards generated
  locally by the trusted self-play process; not an external input). `logratio.clamp(±20)` already
  bounds the effect of a pathological stored logp (defense-in-depth). Note as explicit non-goal.
- "Compute budget / kill criteria / SLO / runbook" → research experiment, resumable via --resume,
  user-launched in background; the existing divergence-guard + safe_opt rollback is the in-flight
  safety. A formal SLO/runbook is overkill for a single研究 re-run. (cost_efficiency lens noted;
  no new infra.)
- "Memory eviction policy / OOM bounds" → `deque(maxlen=K)` IS the eviction policy; ≤K×16 games;
  micro-batching chunks the larger batch; gc.collect per round retained. No new mechanism.
