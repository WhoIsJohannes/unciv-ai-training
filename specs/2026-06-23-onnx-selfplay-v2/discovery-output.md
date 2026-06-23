# Phase 1 Discovery — Output

- **Mode:** BUILD · **Size:** L
- **Feature:** v2 self-play upgrade — a model that consumes the FULL emitted observation (map spatial grid + units + cities + opponents) and an actor-critic trainer with a LEARNED VALUE CRITIC (GAE), built on the committed v1 infra. Two attributable ablation curves: `blind-critic` (Stage A) and `rich-critic` (Stage B).
- **Context category:** backend/tooling (cross-system ML pipeline: Python trainer + Kotlin/JVM bridge + ONNX contract). **Domain preset:** data-pipeline (bias: `practitioner`, `cost_efficiency`, `data_privacy_legal`-N/A → likely swap for an ML/RL-fidelity lens in Phase 2).
- **Invariant docs loaded:** code-quality.md, architecture.md, testing.md — these are generic Next.js/FastAPI product templates; only the *spirit* applies here (small typed files, behavior tests, clear naming). Real invariants = the task's acceptance criteria (parity, determinism, provenance, legality, terminal-reward-only).

## Setup decisions (user-answered)
- **Worktree/branch:** v1 was uncommitted WIP in `/Users/j/Unciv-onnx-selfplay-loop`. Per user ("put everything on main, work off it clean"): committed v1 as `8e0e4ba0a`, fast-forwarded `master` to it (now the clean base), branched `onnx-selfplay-v2` worktree at `/Users/j/Unciv-onnx-selfplay-v2`.
- **Training scope:** **Build + full run to acceptance** — implement everything, then run training long enough to satisfy all acceptance criteria (convergence stddev + Medium-map ceiling at p<0.05). Compute reality noted as a managed risk.

## Light scan summary
Full v1 map in `codebase-scan-light.md`. Highlights: contract v1 (input "obs"[199] → tech_logits[80]/policy_logits[70]); REINFORCE w/ mean baseline + masked-logp + -1/no-action handling; round driver `run_loop.py` (gradle selfPlay gen/eval → curve.csv/png); `OnnxPolicy.kt` single-tensor bridge w/ provenance gate + seeded MaskedChoice; `SelfPlayRunner` hardcodes `MapSize.Tiny`; full provenance/determinism/legality test suite already present. All emitted blocks (spatial nTiles×13, entity token sets, masks) already in every shard.

## Open questions / design forks (resolve in Phase 2)
1. **Spatial encoder:** grid CNN (needs per-tile col/row + map W/H — NOT emitted; "no new emitted data" is a hard non-goal) vs the task-sanctioned fallback (per-tile tokens + positional features + masked pooling/attention). Deep-scan `Tile.zeroBasedIndex`/`HexMath` to see if offset coords are reconstructable from shard data alone; pick + document.
2. **Contract v2 multi-tensor input** named-tensor set + dynamic axes; export-drops-value invariant; JVM bridge build of the same input from `Observation`.
3. **Map-size parametrization** (CLI → SelfPlayRunner) for the added Medium eval.
4. **PPO clip vs plain A2C** for Stage A/B (task says "optionally clip with a few inner epochs").

## Round 1 Q&A
- Q: worktree/v1-uncommitted handling → A: consolidate v1 onto master, work off it clean.
- Q: training execution scope → A: build + full run to acceptance.
- No further clarifying iterations: task spec is exhaustive; remaining forks are technical and delegated to design.
