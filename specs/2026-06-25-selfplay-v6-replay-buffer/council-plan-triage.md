# Plan Council Triage — selfplay-v6-replay-buffer

8 reviewers, 33 findings (5 critical / 24 major / 4 minor), ~$0.60. Verdict: **APPROVE with
refinements** — no finding invalidates the plan; one is a genuine bug, several sharpen it.

## ACCEPT — folded into plan.md / decisions.md before the gate
- **🔴 BUG (critical + major, raised 2×): --resume refill violates round-0 exclusion.** Refilling
  the deque over `[start-K .. start-1]` includes round 0 (RandomPolicy data) when start ≤ K. FIX:
  refill over `max(1, start-(K-1)) .. start-1` — never round 0. + add a resume-refill test
  (round-0 exclusion + missing-dir warning). → D7.
- **K=4 arms use --micro-batch-steps 256** (OOM-safe, math-identical per v5 AC6) to avoid a 4×
  whole-batch blowup at small-rung; K=1 arms match v5 (small=0, medium=256). → D8.
- **AC1 sharpening:** determinism gate = the unit tests (bit-identity test_replay_noop + single-draw
  Kotlin test) GREEN; end-to-end acceptance = K=1 200-game ceiling within the 95% binomial CI of
  v5's 40.7% (n=200 ⇒ ≈±6.8%), else investigate. → D5 updated.
- **AC2 sharpening:** the headline win-claim uses the **200-game ceiling z-test** (p<0.05), NOT the
  noisy 80-game per-round eval; round-to-threshold is directional/secondary. → D5 updated.
- **PolicyProvider.chooseIndexWithLogp default:** strong KDoc — the uniform `ln(1/nLegal)` is correct
  ONLY for a uniform policy and is discarded by the learner-slot filter; any non-uniform provider
  MUST override. → plan Pillar A.
- **Add resume-refill test + non-acting-head 0f assertion**; put the Tiny-smoke PASS criteria into the
  merge-gate test plan (not just council notes). → plan Test plan.
- **Risk notes:** policy-collapse from 4× effective epochs and the loose `logratio.clamp(±20)` are
  bounded/observable by the existing clip + the new mean_ratio/clip_frac diagnostics + safe_opt
  rollback + small K; the lever is lower --replay-window/--clip-eps. → plan Risks.

## REJECT — out of scope for a resumable research re-run / pre-existing / over-engineering
- "Default K=4 ships experimental behavior" (critical + 1 major) — the task EXPLICITLY specs default
  4 with justification; the experiment passes --replay-window for every arm so results are unaffected;
  K=1 no-op is unit-proven. Keep default 4 per spec; run_v6.sh sets the window explicitly. Council
  dissent noted. (Aesthetic/scope call the user already made.)
- "No health-check/liveness/watchdog/timeouts" — existing `--gradle-timeout` (1800s/round, raises
  loudly) + per-round curve.csv/metrics.jsonl progress + the new diagnostics are adequate for a
  user-monitored research run. No new infra.
- "Model metadata coupled to dataset schema (leaky)" — PRE-EXISTING v3/v4 behavior (export stamps
  shard version into META_SCHEMA_VERSION); the perishable-provenance gate is intended, not a v6 leak.
- "clip_eps overloads clipping + IS toggle" — the hard guard makes the overload safe; a separate flag
  is marginal API surface vs the task's "use existing knobs."
- "Shard-tamper input validation" — local trusted pipeline (REJECT, as intake); logratio.clamp is
  defense-in-depth.
- "Compute budget / kill criteria / disk monitoring / runbook / total-runtime success dimension /
  NULL decision tree" — research re-run, resumable, user-launched; safe_opt rollback + divergence
  guard are the in-flight safety. AC2 already allows NULL as a valid result. No formal SLO.
