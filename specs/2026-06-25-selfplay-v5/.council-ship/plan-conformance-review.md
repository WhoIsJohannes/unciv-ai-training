# Plan-conformance review

> **Advisory only — this never blocks the merge.** An independent reviewer judged the shipped code against the plan; LLM conformance judges over-flag, so treat low-confidence deviations skeptically and confirm against the cited evidence.

## Summary

The implementation successfully delivers continual training, micro-batching, and the v5 experiment driver with high fidelity to the complex mathematical and operational requirements. However, it missed adding the planned invariant assertions for terminal rewards and modeled heads, and the architecture map reveals an unrecorded extraction of hex graph logic.

## At a glance

- Matches: **9**
- Deviations: 🔴 corner-cut **1** · ⚪ equivalent **0** · 🟢 improvement **0**
- Unrecorded architectural decisions: **1**
- Untested plan items: **1**

## Deviations

### 🔴 Corner-cut (weaker than planned)
- **Assert invariants (AC7): terminal-only ±1 reward unchanged; MODELED_HEADS == {tech,policy}** — Skipped adding assertions for terminal-only reward and MODELED_HEADS invariants in existing tests. (`python/tests/test_continual_resume.py:133`) _(confidence 0.95)_
  - ↳ Add the missing invariant assertions to test_gae.py or test_parity.py as planned.

## Unrecorded architectural decisions (ADR stubs)

### Extract hex graph adjacency logic to hexgraph.py
- **Context:** The MAP diff shows a new module `hexgraph.py` and new exports in `features.py` related to hex graphs.
- **Decision:** Extract fixed degree-six hex adjacency indices and masks into a dedicated `hexgraph.py` module.
- **Consequence:** Separates spatial graph construction from feature padding, improving cohesion, but was not recorded in decisions.md.
- **Evidence:** `python/unciv_train/MAP.md:16-17`

## Architecture fidelity (MAP diff vs decisions.md)

⚠️ **drift** — The MAP diff reveals a new `hexgraph.py` module that was not documented in `decisions.md`.

## Untested plan items

- **Assert invariants (AC7): terminal-only ±1 reward unchanged; MODELED_HEADS == {tech,policy}** — No test assertions were added for the terminal reward invariant or MODELED_HEADS, despite being explicitly requested in the plan.

## Builder self-report vs reality

progress.md claims 'No MISSING items' and checks off the tests item, but silently dropped the AC7 invariant assertions for terminal reward and MODELED_HEADS.

## Independent test re-run

❌ FAIL — `pytest tests/ --ignore=tests/smoke_llm.py -p no:asyncio -q` (exit None, 0.0s) — could not run test command: [Errno 2] No such file or directory: 'pytest'

## MAP drift (deterministic, LLM-free)

- 0 stale of 11 mapped files

## Matches

- **Lift the optimizer out of _optimize_actor_critic + inject it** — Optimizer injected as a parameter and used directly. (`python/unciv_train/train.py:119`)
- **Opt-aware divergence guard** — safe_opt deepcopied at init and restored on NaN. (`python/unciv_train/train.py:143`)
- **Micro-batch the dense traversal** — Chunked epoch pass with size-weighted accumulation implemented; whole-batch path kept byte-verbatim in else branch. (`python/unciv_train/train.py:182-211`)
- **Warm net+opt in the trainers** — Trainers reuse net and optimizer if provided, skipping manual_seed. (`python/unciv_train/train.py:388-392`)
- **Persist + carry across the loop** — warm_net and warm_opt carried to the next round when --continual is true. (`python/unciv_train/run_loop.py:284-287`)
- **Atomic saves** — _atomic_torch_save implemented using tmp file and os.replace. (`python/unciv_train/run_loop.py:70-77`)
- **Resume fail-fast** — _load_warm checks for both sidecars and raises FileNotFoundError with a clear message. (`python/unciv_train/run_loop.py:82-105`)
- **Experiment driver python/run_v5.sh** — Sequential arms A and B configured exactly as planned. (`python/run_v5.sh:53-55`)
- **analyze_v5.py ceiling eval + z-tests** — z-tests vs v4 23.0% and blind 28.9% implemented. (`python/unciv_train/analyze_v5.py:84-91`)

---
_Reviewer: google/gemini-3.1-pro-preview; reasoning budget 12000 tokens._
