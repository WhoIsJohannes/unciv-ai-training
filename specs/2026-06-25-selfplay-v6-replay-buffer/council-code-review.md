# Ship-Review Council (Phase 4 Step 18) — selfplay-v6-replay-buffer

6 reviewers (skeptic/architect/practitioner/qa_testing/security_red_team/domain_fidelity),
23 findings (3 critical / 13 major / 7 minor), ~$1.01. Final verdict: APPROVE after 5 fixes
(commit da8962bce).

## FIXED (5)
1. **[critical] `ln(exps[pos]/sum)` → -Inf on underflowed sampled action.** Switched to the
   numerically-stable log-softmax form `(logit[chosen]-maxL) - ln(sum)` — finite for underflow +
   exact match to Python `F.log_softmax` (improves JVM↔Python off-policy parity). MaskedChoice.kt.
2. **[critical] resume refill crashes the whole run on a corrupt shard.** Wrapped each historic-round
   `load_trajectories` in try/except (warn+skip, window under-fills gracefully). run_loop.py.
3. **[major] round 0 used stored RandomPolicy uniform logp at K>1.** Forced round 0 to on-policy
   recompute (`replay_active = … and r>0`) — round 0 now identical across K=1/K=4 arms (v5 bootstrap);
   replay effect isolated to rounds ≥1. run_loop.py.
4. **[major] K=4 default silently changes legacy scripts.** Pinned `--replay-window 1` in run_v5.sh +
   run_acceptance.sh (preserve v5/v2 single-round semantics under the new K=4 default).
5. **[major] diagnostics write-only, no alert.** Added a stderr `[replay-health WARN]` when
   mean_ratio>1.5 or clip_frac>0.5. run_loop.py.

## REJECTED / NOTED (not bugs in this codebase)
- **[critical] GPU device mismatch on `stored_old_logp`** — N/A: this is a CPU-only pipeline;
  `a_tech`/`m_tech`/`adv`/`ret` use the IDENTICAL CPU-tensor convention. `stored` follows it exactly.
  Fixing only `stored` while the rest stay CPU would be inconsistent and pointless.
- **[major] "missing logratio clamp → exp() overflow"** — FALSE: the existing verbatim code HAS
  `logratio = (logp - old_logp).clamp(-20.0, 20.0)` in both branches (untouched). A pathological
  -Inf old_logp is bounded by clamp(+20)→ratio finite → PPO clip caps it → no NaN.
- **[major] positional `behavior_logp[0]/[1]` couples to MASK_HEADS order** — consistent with the
  EXISTING `actions[0]/[1]` convention (same coupling, same block family); changing only behavior_logp
  would diverge from the established pattern.
- **[major] chooseIndexWithLogp default records uniform for non-uniform policies** — task-specified
  default; mitigated by (a) the only non-uniform policy (OnnxPolicy) overrides, (b) RoutingPolicy
  delegates, (c) round-0 RandomPolicy data is the ONLY place the uniform default applies and it is
  EXCLUDED from the replay window, (d) a strong KDoc warning. Risk is purely a hypothetical future policy.
- **[major] _blogp None→0 / all-zero stored** — production loader ALWAYS populates b_logp; the None→zeros
  path is synthetic-test-only and never coincides with behavior_logp=True in run_loop. Real replayed
  rounds (OnnxPolicy, round≥1) have non-zero sampling logp; round 0 (the only uniform/leak source) is
  excluded + now recompute-only. clip_eps guard + clamp + clip bound any residual.
- **[major] resume under-fill biases AC2** — graceful degradation (warn) is correct vs crashing; the
  keep_shards floor normally guarantees the window; a genuinely-missing dir is logged. Acceptable for a
  resumable research experiment.

## Minors (7) — deferred (style/wording; no behavior impact). None gating.
