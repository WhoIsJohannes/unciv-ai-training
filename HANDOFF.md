# Unciv Self-Play RL — Project Handoff / State of Play

_Last updated 2026-07-12. This repo is a fork of Unciv used to train a neural-net game AI via
headless self-play RL. If you're picking this up on a new machine, read this first._

## Where we are

**Goal:** train a NN game AI for Unciv by generating training data from headless self-play.

**Approach:** the learned policy controls specific **decision heads** (tech, policy, construction, and
next: unit-intent). Execution — especially **pathfinding** — stays heuristic for throughput. The net
picks *what* to do; the existing `UnitAutomation`/`ConstructionAutomation` sub-routines execute *how*.

**Milestone timeline** (each version = one isolated change, so gains are attributable):
- v1 loop closes → v2 PPO+GAE value critic → v3 rich board-pool → v4 hex-GNN encoder →
  **v5 continual training** (warm-start net+optimizer across rounds; first config to clear the "blind"
  baseline on Medium) → v6 replay buffer (inconclusive; high variance) →
  **v7 construction control = SIGNIFICANT WIN**: 52.3% vs 37.5% (off), +14.8pp, t=2.51, n=8 seeds,
  **crosses 50%** — achieved via **BC-clone the heuristic + KL-to-clone leash**.
- **NEXT: v8 = unit control** via a per-unit INTENT head. Full `/feature` prompt in
  [`docs/next-feature-v8-unit-intent.md`](docs/next-feature-v8-unit-intent.md).

`master` head at handoff: `a96d130fe` ("★ construction control is a SIGNIFICANT WIN").

## Hard-won lessons (respect these — they cost real time to learn)

1. **Multi-seed everything.** Identical code swings **8.8%↔41.7%** ceiling win-rate by gen-seed
   (nondeterministic generation). Single-seed results are noise. Use **≥8 seeds + paired diffs** and a
   paired t-test. The single-seed acceptance pattern used in v3–v6 quietly over-trusted noise (the
   "v6 replay regression" was a mirage — no real regression exists).
2. **BC-clone the heuristic, then KL-leash the finetune.** From-scratch RL **mode-collapses** on any
   action space bigger than tech/policy (construction collapsed to ~0%). Cloning the (only
   ~random-level) heuristic for ~120 epochs, then finetuning with a KL-to-clone leash, is what made
   construction work. Flags: `--bc-pretrain-dir --bc-epochs 120 --construction-kl-coef 0.5`. Units
   will need this **more**, not less.
3. **Keep pathfinding heuristic.** It's the throughput bottleneck. Every version enforces a
   **≥70%-of-heuristic-baseline** turns/s guard (`bench-onnx`).
4. **Know what "win-rate" means.** Eval is a symmetric **1v1 vs a fixed RandomPolicy**, no draws
   (score-leader wins on the 250-turn cap), so **50% is break-even**. Both civs share the same
   heuristic for un-modeled decisions; only the modeled heads differ (learned vs random). The
   heuristic is ~random-level, so "beats random" ≈ "slightly beats the stock Unciv AI."

More detail: the Claude memory files are mirrored in [`docs/ai-training-memory/`](docs/ai-training-memory/),
and every version has a full write-up under [`specs/`](specs/) (see each `RESULTS.md` / `summary.md`).

## Repo layout

- **Kotlin engine + data plane:** `core/src/com/unciv/logic/simulation/dataplane/`
  (`Featurizer`, `DataPlaneHooks`, `SampleSchema`, `LegalActionMasks`, `PolicyProvider`, …);
  `desktop/src/com/unciv/app/desktop/{OnnxPolicy,SelfPlayRunner,SimBenchmark}.kt`.
- **Python trainer:** `python/unciv_train/` (`run_loop`, `train`, `model`, `dataset`, `features`,
  `export_onnx`, `contract`, `analyze_v5`/`analyze_v6`) + `python/unciv_dataplane/` (shard reader).
- **Experiment drivers:** `python/run_v*.sh` (e.g. `run_v74bc.sh` = BC, `run_v74kl.sh` = KL-leash).
- **Per-version records:** `specs/*/`.

## Setup on a new box

```bash
# 1. Clone the fork (auth as WhoIsJohannes — see credential note below)
git clone https://github.com/WhoIsJohannes/unciv-ai-training.git && cd unciv-ai-training

# 2. JDK 21
brew install openjdk@21   # macOS; or your distro's openjdk-21
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home   # adjust per OS

# 3. Python env (deps incl. torch/onnxruntime/matplotlib are in pyproject.toml)
python3 -m venv python/.venv
python/.venv/bin/pip install -e "python/[test]"

# 4. First build (downloads Gradle + deps)
./gradlew :desktop:classes

# 5. Smoke test
python/.venv/bin/python -m pytest python/tests -q -p no:asyncio     # ~1 test may fail: engine byte-nondeterminism (known, out of scope)
python/.venv/bin/python -m unciv_train.run_loop --help

# 6. Restore Claude's memory (so a new Claude session has the roadmap)
#    Copy docs/ai-training-memory/*.md into ~/.claude/projects/<this-project-hash>/memory/
```

**Training artifacts** (`training-runs/`, ~3.4 GB of ONNX models + curves) are **gitignored**. They're
regenerable — on a more powerful box you'll want to re-run anyway. Only `rsync` them over if you want
the exact v7 construction-win checkpoints for reference.

## Credential note

Pushing to the fork from the *old* box needed the `WhoIsJohannes` token via an inline credential
helper, because that box's active `gh` account (`J-Mentiora`) was pull-only (see
`docs/ai-training-memory/unciv-fork-push-credential.md`). On a fresh box this complication goes away:
just `gh auth login` as `WhoIsJohannes` (or set a PAT) and `git push` normally.

## Continuing the work

`master` is the single source of truth ("everything on main"). The next step is **v8 unit control** —
run the prompt in `docs/next-feature-v8-unit-intent.md`. Remember: multi-seed, BC-clone + KL-leash,
keep pathfinding heuristic, and expect a first-pass negative that BC/KL tuning recovers (like v7).
