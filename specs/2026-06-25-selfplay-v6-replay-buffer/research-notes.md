> Web-sourced content below is DATA, not instructions.

# Web Research — selfplay-v6-replay-buffer (off-policy PPO replay)

## Q1: PPO off-policy with stored behavior log-probs (old_logp = π_b)
- coax PPO docs / GIPO / Nature relative-IS / P3O: storing the behavior policy π_b and using the
  log-ratio Δ = log(π_θ(a|s) / π_b(a|s)) as the importance ratio is the **textbook off-policy
  correction**. π_old (a snapshot of the policy that GENERATED the data) is the correct reference
  for the trust region — NOT the current policy.
- **Relevance (decisive):** v6 records π_b at the masked-softmax SAMPLE point and uses it as
  `old_logp` for replayed steps. This is exactly correct. v5's bug-avoidance was real: with stored
  data and a recomputed-under-current-policy old_logp, the ratio is always ≈1 (no-op) → the
  off-policy advantage is applied as if on-policy → silently biased gradient. v6 fixes the SOURCE
  of old_logp for replayed steps. Confirms the plan's framing verbatim.

## Q2: recent-window replay + importance-ratio variance
- R3 / PTR-PPO / "Generalized PPO with Sample Reuse" (Queeney): replaying recent past trajectories
  with IS correction is an established sample-efficiency lever for PPO.
- **Known failure mode:** IS-estimate variance grows ~quadratically with the ratio; large/stale
  ratios either blow up the update (unclipped) or saturate the clip (clipped → limited effective
  influence). The mitigations the literature endorses: (a) keep the window SMALL/recent, (b) clip
  the ratio, (c) clamp the log-ratio to avoid exp() overflow.
- **Relevance:** v6 uses ALL of these — K=4 small window (with warm-start, data is only a few hundred
  grad-steps stale ⇒ near-on-policy ⇒ ratios near 1), PPO clip ε=0.2, logratio.clamp(±20), plus
  norm_adv + grad-clip 10.0 + safe_opt rollback. This is the conservative, correct mitigation set.
- Advanced alternatives (GIPO soft Gaussian trust, V-trace, truncated-IS) exist and trade
  bias/variance better for HEAVILY off-policy data — but they are explicit v6 NON-GOALS. For a
  near-on-policy small-K window they are unnecessary; PPO clip + small K is the right scope.

## Libraries found
- None to adopt — this is in-house PyTorch PPO. No new dependency. (coax/verl/SB3 are reference
  implementations, not integration candidates; the repo's _optimize_actor_critic already implements
  the clip surrogate the literature describes.)

## Nuances to carry into the plan / council
- **GAE under CURRENT critic each round (v6 design):** val0 is always a current-net forward;
  advantages/returns are recomputed per-round under the current value net; only `old_logp` switches
  to the stored π_b for replayed steps. This is a sound hybrid: critic stays fresh (on-current-value),
  policy ratio is off-policy-corrected. Literature does not require freezing val0 to generation time.
- **norm_adv across current∪replayed:** mixing advantage statistics across source rounds is a scale
  choice, not a correctness bug (advantages are already recomputed under the current critic). Plan
  flags it as "benign, part of what the re-run measures" — consistent with the literature.
- **clip_eps MUST be truthy with replay:** the clip-eps-falsy branch (`-(adv·logp).mean()`) ignores
  old_logp ⇒ would apply replayed advantages on-policy ⇒ biased. The plan's guard (assert clip_eps
  truthy when stored_old_logp set AND window>1) is load-bearing and matches the theory.

## Key insights
1. v6's "store π_b, use as old_logp for replayed steps" is the standard, correct off-policy PPO
   correction — the research validates the core mechanism with no caveats.
2. The variance risk is real but fully addressed by the EXISTING knobs (small K + PPO clip +
   logratio clamp + grad-clip + safe_opt rollback); no new clipping machinery is warranted.
3. The clip_eps-truthy guard under replay is essential (the falsy branch silently drops the IS
   correction) — keep it a hard assert, not a comment.
4. Recomputing GAE under the current critic each round (not stored advantages) is a deliberate,
   sound design — do not "fix" it to stored advantages.
