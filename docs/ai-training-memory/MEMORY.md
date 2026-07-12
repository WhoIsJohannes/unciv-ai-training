# Memory index

- [Self-play roadmap bottleneck](selfplay-roadmap-bottleneck.md) — from-scratch-per-round training WAS the ceiling; v5 continual training RESOLVED it (clears blind 40.7% vs 28.9%, p=0.0069, on master). v6 replay-buffer; then action heads.
- [Construction control — WIN (v7.4, replicated)](selfplay-v7-construction-negative.md) — was a robust negative (mode collapse to ~0%); RESOLVED by BC-clone-heuristic (120ep) + KL-to-clone leash: **bckl 52.3% vs off 37.5%, +14.8pp, t=2.51 SIGNIFICANT (n=8), crosses 50%**. Recipe: clone the (only-~random-level) heuristic then leashed-finetune. Opt-in flags: --bc-pretrain-dir --bc-epochs 120 --construction-kl-coef 0.5.
- [Self-play baseline VARIANCE](selfplay-baseline-variance.md) — identical code swings 8.8%↔41.7% ceiling by gen-seed (nondeterministic gen); single-seed experiments UNRELIABLE. Climb happens rounds 8–15. Replicate ≥4 seeds + paired diffs. NO v6/v7.x regression exists. mb0≡mb256.
- [Unciv fork push credential](unciv-fork-push-credential.md) — pushing to fork (WhoIsJohannes/unciv-ai-training) needs the WhoIsJohannes token via inline credential helper; active gh account J-Mentiora is pull-only.
