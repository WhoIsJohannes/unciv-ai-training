# ONNX policy bridge + self-play training loop — Phase 3: Build

Progress tracker for Steps 12–16. Tick boxes and fill evidence after EVERY step.
Save this file to disk after each step.

> **Slug**: `onnx-selfplay-loop` | **Branch**: `onnx-selfplay-loop` | **Started**: 2026-06-23
> **Spec folder**: `specs/2026-06-23-onnx-selfplay-loop/` | **Mode**: _(from discovery-output.md)_
> **Size**: _(from discovery-output.md)_

**RULES:** (1) MANDATORY STOP = present output and WAIT for user response — **unless**
the step produced nothing decision-worthy or surprising (all gates green, no
regressions, no findings, no exceptions). In that case, print a one-line status and
continue. Error refusals and failure escalations (iter-3-fail, plan retreat) NEVER
skip. When in doubt, stop. (2) Questions via `AskUserQuestion` with 2-4 options.
(3) One step at a time. (4) BLOCKING = KILL, gate fail ×3, policy violation.
(5) Artifacts in `specs/2026-06-23-onnx-selfplay-loop/`. (6) Update `FEATURE_STATE.json` after each step.
(7) Log OpenRouter calls to `cost-log.jsonl`. (8) **Telemetry**: log step start/end to
`specs/2026-06-23-onnx-selfplay-loop/timings.jsonl` (`{"step": N, "phase": 3, "event": "start|end", "ts": "..."}`).
(8b) **Q&A capture**: after every `AskUserQuestion` gate, append one line to
`specs/2026-06-23-onnx-selfplay-loop/questions.jsonl`: `{"ts", "phase", "step", "header", "question",
"options": [...], "chosen", "source"}`. Consumed at ship by
`lib.observability.summary` → `user_question` events.
Sync events (from `sync-with-base.sh`, this phase) use
`{"event": "sync", "from_phase": N, ...}` — see `lib/sync-with-base.sh` for the
full sync-event schema.
(9) **Default-rigor.** When a decision is binary AND one option matches established
rigor (write tests / fix root cause / hit real DB / reuse over rebuild / retreat
over patch / address vs forget) and the other is a corner-cut, choose rigor
without asking. Only surface `AskUserQuestion` when the user must provide info
Claude cannot determine: state (resume, dev URL), intent (BUILD/FIX, merge
yes/no), repo fact (FE_DIR/BE_DIR), or genuine tradeoff (UI mockup time budget,
council roster scope). Does NOT cover aesthetic/scope tradeoffs where "more"
isn't strictly more rigorous — those remain user decisions.

### Pre-read

Read Phase 2 handoff artifacts: `plan.md`, `decisions.md`, `council-plan-review.md` (if exists),
`discovery-output.md`.

- [ ] Phase 2 artifacts read

---

## Step 12: Build — Initialize

Create `specs/2026-06-23-onnx-selfplay-loop/progress.md` with all plan items as checkboxes + Codebase Patterns section.

- [ ] `specs/2026-06-23-onnx-selfplay-loop/progress.md` created

---

## Step 13: Build — Iteration 1

**13a. Read progress & pick items**
- [ ] Read `specs/2026-06-23-onnx-selfplay-loop/progress.md`, pick highest-priority incomplete items

**13b. Implement**

Write code. Do NOT generate: abstractions for one-time ops, error handling for impossible
scenarios, docstrings restating names, utility files for single-use, comments saying WHAT,
wrappers adding no logic, leftover console.log, type assertions when guards exist.

**Different-context rule:** Derive test expectations from **plan/spec**, NOT your implementation.

- [ ] Code written
- [ ] Tests derived from plan/spec (not implementation)

**13c. Completion oracle**

> *Self-reported here. Independently backstopped at the Phase-4 plan-conformance gate
> (Step 18b), which re-judges fidelity from the diff WITHOUT trusting `progress.md` — `[x]`
> is a claim it will verify, not the source of truth. Mark honestly.*

- [ ] Each item in `progress.md`: `[x] verified in <file>:<line>` or `[ ] MISSING: <what>`
- [ ] All MISSING items fixed

**13d. Quality gates**

⚡ **PARALLEL:** Run frontend, backend, security gates simultaneously.

The gate block probes for the repo's layout (root, `frontend/`, `web/`,
`apps/web/`, `app/` for FE; root, `backend/`, `server/`, `services/api/`, `api/`
for BE). On first run in a new layout, if nothing matches, the orchestrator
`AskUserQuestion`s for the paths and persists the answer to
`.feature-workflow/layout.env` so subsequent runs auto-load it.

> **Subshell expectation.** The orchestrator runs the block below in a subshell
> (e.g. `bash -c '…'` or `(…)`) so the BLOCKING `exit 1` terminates only this gate
> block, not the orchestrator's shell. Same applies to every fenced `bash` block
> in this template that contains `exit` or `set -e`.

```bash
# Load persisted layout (if any) — set on first detect-miss.
[ -f .feature-workflow/layout.env ] && . .feature-workflow/layout.env

# Detect frontend (any JS/TS package marker counts).
if [ -z "${FE_DIR:-}" ]; then
  for c in . frontend web apps/web app; do
    [ -f "$c/package.json" ] && FE_DIR="$c" && break
  done
fi

# Detect backend (any Python project marker counts: poetry, setuptools, pip, pipenv).
if [ -z "${BE_DIR:-}" ]; then
  for c in . backend server services/api api; do
    for marker in pyproject.toml setup.py requirements.txt Pipfile; do
      if [ -f "$c/$marker" ]; then BE_DIR="$c"; break 2; fi
    done
  done
fi

# BLOCKING: if neither layout detected AND no explicit skip flag, refuse to
# silently "pass". The orchestrator must AskUserQuestion (see below), persist
# the answer, and re-run. Set FEATURE_SKIP_GATE_QUALITY=1 ONLY for docs-only /
# template-only changes where no lint/test/typecheck is meaningful.
if [ -z "${FE_DIR:-}" ] && [ -z "${BE_DIR:-}" ] && [ -z "${FEATURE_SKIP_GATE_QUALITY:-}" ]; then
  echo "BLOCKING: no FE or BE project markers detected in . / frontend / web / apps/web / app / backend / server / services/api / api." >&2
  echo "Orchestrator: AskUserQuestion to locate the repo's package roots, then persist to .feature-workflow/layout.env and re-run." >&2
  echo "Or set FEATURE_SKIP_GATE_QUALITY=1 if this is a genuine no-code change (e.g. doc-only)." >&2
  exit 1
fi

# Persistence (orchestrator runs this after the AskUserQuestion if it fired):
#   mkdir -p .feature-workflow
#   { echo "FE_DIR=$FE_DIR"; echo "BE_DIR=$BE_DIR"; } > .feature-workflow/layout.env
# Either FE_DIR or BE_DIR may legitimately be empty (single-language repo).

# Run gates against detected layout. Empty FE_DIR / BE_DIR is a no-op (not failure).
[ -n "${FE_DIR:-}" ] && (cd "$FE_DIR" && npm run lint && npm run type-check && \
  npm test && npm run build)
[ -n "${BE_DIR:-}" ] && (cd "$BE_DIR" && ruff check . && ruff format --check . && \
  pyright . && pytest .)

# Security gates: scope to detected layout if found, else repo root. Prefer
# $FE_DIR/src if it exists (typical monorepo), otherwise fall back to $FE_DIR
# itself (flat layout where sources live alongside package.json at the root).
SEMGREP_PATHS=""
if [ -n "${FE_DIR:-}" ]; then
  [ -d "$FE_DIR/src" ] && SEMGREP_PATHS="$FE_DIR/src" || SEMGREP_PATHS="$FE_DIR"
fi
[ -n "${BE_DIR:-}" ] && SEMGREP_PATHS="$SEMGREP_PATHS $BE_DIR"
[ -z "$(echo $SEMGREP_PATHS | tr -d ' ')" ] && SEMGREP_PATHS="."
gitleaks detect --source . --no-git --verbose 2>&1
semgrep scan --config auto --severity ERROR --severity WARNING $SEMGREP_PATHS 2>&1
[ -n "${FE_DIR:-}" ] && (cd "$FE_DIR" && npm audit --audit-level=high 2>&1)
```

If both `FE_DIR` and `BE_DIR` come back empty, the block above exits non-zero
(BLOCKING). The orchestrator MUST then `AskUserQuestion`: "Where should
lint/test/typecheck run? (e.g. `.`, `apps/web`, `packages/foo`)", persist the
answer to `.feature-workflow/layout.env`, and re-run. Genuine doc-only changes
get `FEATURE_SKIP_GATE_QUALITY=1` recorded in `decisions.md` instead.

- [ ] Layout: FE_DIR=___ BE_DIR=___ (or "skipped — FEATURE_SKIP_GATE_QUALITY=1: rationale in decisions.md")
- [ ] All applicable checks ran (or "no applicable checks for doc-only change")

**13d-bis. Shape-without-behavior + boundary gates (feature-workflow-internal IP helpers)**

These run AFTER 13d's lint/type/test gates and BEFORE mutation/lint. Each writes
a sidecar JSON to `specs/2026-06-23-onnx-selfplay-loop/<gate>-findings.json` (schema_version=1). Exit code
1 = findings → BLOCKING per the canonical "gate failure ×3" rule.

```bash
# Static gates — no API key, runs on every machine.
set -e

# src_self_contained: Rust-specific (blocks include!/#[path] from src/ to systems/).
# Skip cleanly when the repo isn't laid out that way.
if [ -d src/systems ]; then
  SPEC_DIR=specs/2026-06-23-onnx-selfplay-loop python3 "/Users/j/.claude/feature-workflow-internal/lib/static/src_self_contained.py" --src-root src
fi

# Allowlist override path (per repo): .feature-workflow/design-system-allowlist.txt
SPEC_DIR=specs/2026-06-23-onnx-selfplay-loop python3 "/Users/j/.claude/feature-workflow-internal/lib/static/visual_gate.py" \
  --repo "/Users/j/Unciv-onnx-selfplay-loop" --base "self-play-data-plane" --head HEAD --lib-dir "/Users/j/.claude/feature-workflow-internal/lib"

# Tier A (hidden ARIA seeds) + Tier C (defaultDeps throw-stubs).
SPEC_DIR=specs/2026-06-23-onnx-selfplay-loop python3 "/Users/j/.claude/feature-workflow-internal/lib/static/behavior_gate.py" \
  --repo "/Users/j/Unciv-onnx-selfplay-loop" --base "self-play-data-plane" --head HEAD --mode both
```

Per-gate skip flags (use only for genuine, justified reasons; record in
`specs/2026-06-23-onnx-selfplay-loop/decisions.md`):
- `FEATURE_SKIP_GATE_VISUAL=1 python3 …` — skip visual_gate (e.g. backend-only feature).
- `FEATURE_SKIP_GATE_BEHAVIOR=1 python3 …` — skip behavior_gate.
- `FEATURE_SKIP_GATE_SRC=1 python3 …` — skip src_self_contained.

- [ ] src_self_contained: clean / N findings fixed
- [ ] visual_gate Tier A: clean / N findings fixed / advisory (no allowlist)
- [ ] behavior_gate Tier A+C: clean / N findings fixed

**Mutation testing** (only if the repo already has a mutation runner — Stryker for
JS/TS in `package.json`, mutmut/cosmic-ray in `pyproject.toml`):
```bash
# Run whatever the repo configures. Discover source root for the Python case:
SRC=""; for d in src lib app .; do
  [ -d "$d" ] && [ "$(find "$d" -maxdepth 2 -name '*.py' -print -quit)" ] && SRC="$d" && break
done
npx stryker run                              # JS/TS
# Python: only run when $SRC was actually discovered — no silent fallback to `.`
if [ -n "$SRC" ]; then
  mutmut run --paths-to-mutate "$SRC"
else
  echo "mutmut skipped — no Python source detected"
fi
```
S: advisory only. M/L: <50% surviving mutants = warning, <30% = BLOCKING.
Skip if the repo has no mutation config — don't introduce one as part of a feature.

- [ ] Mutation testing ran (or "skipped — no mutation runner configured")
- [ ] Invariant compliance checked (Step 16 security checklist below applies; no
  external policy doc to cross-reference)

**13e.** Log iteration in `specs/2026-06-23-onnx-selfplay-loop/progress.md`.
**13f.** All gates PASS + verified → Step 14. FAIL → loop. Iteration 3 failing → **MANDATORY STOP**.

Evidence: items_done= | remaining= | gates= | mutation= | decision=

<!-- Copy Step 13 block for iterations 2-3 if needed -->

---

## Step 14: Context Checkpoint

### Cleanup-opportunity scan + Traceability check

⚡ **PARALLEL (M/L):** Launch both simultaneously. S: lite cleanup artifact only
(skip the LLM-driven cleanup scan and traceability — see below).

#### Cleanup-opportunity scan → `/Users/j/Unciv-onnx-selfplay-loop/.feature-workflow/cleanup-opportunities.md`

Produces a structured artifact of cleanup candidates noticed while the build context is
fresh. Consumed by Phase 4 Step 17 (per-item address-now / defer / drop prompt), by the
`/feature` Step 0.7 nudge, and by standalone `/cleanup --address-oldest` later. Schema
documented in `templates/cleanup-checklist.md` and tested by `tests/test_cleanup_skill.py`.

**Storage:** the artifact is a **central per-repo file** at
`/Users/j/Unciv-onnx-selfplay-loop/.feature-workflow/cleanup-opportunities.md` — NOT per-spec. Items
accumulate across all `/feature` runs in this repo. Done items (`[x]`/`[~]`) are
sorted to the bottom by the artifact's mutator helpers (F3 invariant).

**For S features (SIZE GATE):** write a **lite artifact**. If the central file does
not exist, create it with the schema header. Append any inline-noticed items via the
same dedup logic as M/L (below) but skip the standalone scan agent.

**For M/L features:** standalone scan pass. Read the changed files in the diff, look for:
- **Large file** (>300 lines, `kind:large-file`) — flag if a natural split exists.
- **Duplication** (`kind:duplication`) — flag blocks ≥5 lines that appear ≥2× across the
  changed area or against the surrounding modules.
- **Magic values / naming** (`kind:naming` / `kind:magic-value`) — flag inline.

The scan is an inline Claude prompt (no dedicated helper in v0.6) capped at ≤2000 tokens
of output. If the scan fails (e.g. no model context), write the lite artifact instead
and log "cleanup scan degraded to lite" to stderr — never blocks the build.

**In-session dedup (M/L and S share the same logic; no extra LLM call):** before
appending new findings, the same Claude turn that produces them MUST:

1. Read the existing central file via `cleanup_artifact.parse()`. **F6 failure
   semantics:** if `parse()` raises OR the file fails YAML front-matter validation,
   log warning `cleanup dedup: existing central file unreadable; appending without
   dedup` and proceed treating the existing item set as empty. NEVER block the build.
2. For each new finding, decide:
   - **Duplicate** of an existing item (same `file:` + `kind:`, or paraphrased text
     referring to the same problem) → SKIP the write. Optionally call
     `cleanup_artifact.append_originating_feature(text, existing.uuid, onnx-selfplay-loop)` to
     record provenance.
   - **Refines** an existing item (same target but new detail) → MERGE: update the
     existing item's text/meta in-place; append `onnx-selfplay-loop` to `originating_features:`.
   - **Genuinely new** → APPEND a fresh `- [ ] <desc>  <!-- ... -->` line.
3. The mutators (`mark_*`, `append_originating_feature`) auto-sort the file so
   `[ ]` items remain at top. No explicit `sort_unaddressed_first` call needed.

**Per-item metadata** (extended from v0.6):
- `id:<uuid>` (existing)
- `file:<path>` `line:<N>` (existing)
- `kind:<large-file|duplication|naming|magic-value>` (existing)
- `source_commit:<read from FEATURE_STATE.json's source_commit_at_start>` — the
  build's STARTING commit (captured at Phase 1 Step 0), NOT `git rev-parse HEAD`
  at scan time. Used by `is_stale_for_head()` later. See decisions.md D16.
- (provenance is NOT tracked in the artifact — council m-2 found no consumer;
  if you need to know which feature surfaced an item, use `git log -S "<id:uuid>"`
  against `.feature-workflow/cleanup-opportunities.md` instead.)

**Writing:** ALWAYS wrap the read-modify-write cycle in
`cleanup_artifact.locked_read_modify_write(path)` (round-2-C-1 fix — lost-update
race). The context manager yields a `commit(text)` callable that writes atomically
under an exclusive lock. Pattern:

```python
from lib import cleanup_artifact
path = f"{REPO_ROOT}/.feature-workflow/cleanup-opportunities.md"
with cleanup_artifact.locked_read_modify_write(path) as commit:
    try:
        existing = open(path).read()
    except FileNotFoundError:
        existing = ""
    # ... dedup logic against existing ... compose new_text ...
    commit(new_text)
```

Never call `atomic_write` directly on the central path from Phase 3 Step 14 —
go through the lock-aware helper so concurrent /feature invocations don't lose
each other's writes.

Artifact format:

```markdown
---
schema: cleanup-v1
generated_at: <ISO 8601>
---

## Cleanup opportunities

- [ ] <description>  <!-- id:<uuid> file:<path> line:<N> kind:<...> source_commit:<sha> -->
```

Note: the front-matter `source_commit:` and `feature_slug:` fields from v0.6 are
demoted to per-item metadata (since the central file aggregates across features).
For compatibility, the parser tolerates legacy front-matter fields silently.

- [ ] Central file written via `locked_read_modify_write` (full scan for M/L, lite for S)
- [ ] In-session dedup ran (or "no existing items, all new") — F6 fallback logged if applicable
- [ ] Per-item `source_commit` set from FEATURE_STATE.json's `source_commit_at_start`

### Traceability check (M/L only)

> **SIZE GATE:** Skip for **S**.

Launch a Claude sub-agent via the Agent tool (`subagent_type: "general-purpose"`)
with this mission: read `specs/2026-06-23-onnx-selfplay-loop/plan.md`, `specs/2026-06-23-onnx-selfplay-loop/progress.md`, and
`specs/2026-06-23-onnx-selfplay-loop/agent-test-spec.md`; produce a table mapping every plan item to its
implementing commits/files AND to a covering test. Flag rows where either column
is empty. Write the table to `specs/2026-06-23-onnx-selfplay-loop/traceability.md`.

- [ ] Traceability sub-agent ran (or "skipped — S")
- [ ] Gaps addressed (or "full coverage")

### Checkpoint

Write `specs/2026-06-23-onnx-selfplay-loop/checkpoint.md`: what was built, key files, open issues, gate status,
traceability, test spec path, council plan review path.

- [ ] `specs/2026-06-23-onnx-selfplay-loop/checkpoint.md` written
- [ ] Context compressed (if supported)

---

## Step 15: Test — Green + Affected

> **HARD GATE — NOT SKIPPABLE.** Phase 4 refuses to start without Step 15.

Read `test_mode` from `FEATURE_STATE.json` (set in Phase 2 Step 11). The path below
adapts: if `agentic`, prefer the repo's agentic runner; if `integration`, run the
integration test you wrote.

### Dev Server Pre-Flight

Most agentic and many integration tests need a live app. Ask user via `AskUserQuestion`:

A) Already running (give URL — the test command will use it)
B) Start it (or I'll start it myself — confirm URL once up)

If the test framework runs in-process (e.g. pure unit/integration with no HTTP), skip
this section.

- [ ] Dev server ready (or N/A), health-checked at: ___

### Green

Run the test command for the spec/test you wrote in Phase 2 Step 11.

```bash
# Use the repo's existing runner — typically the same command as the
# Verify-Red step in Phase 2, just expecting a PASS now:
<test runner command> <path to your new spec/test>
```

- [ ] Test **passed**
- [ ] If failing: fix feature code (not the test), re-run
- [ ] Any test artifacts (recordings, snapshots) committed if the framework produces them

### Affected Existing Tests

Run any existing tests in scopes touched by your change. If the repo has a way to
discover them (e.g. `<runner>:affected`, `jest --findRelatedTests`, a CI workflow that
selects tests by changed paths), use it. Otherwise scope manually by grepping for the
modules you touched.

- [ ] Affected tests identified and ran (or "none affected")
- [ ] Regressions fixed (pre-existing failures noted but don't block)

### Plan retreat check — gate (M/L only)

> **SIZE GATE:** Skip for **S**.

> *This builder-side gate catches plan flaws the builder notices. It is independently
> backstopped at the Phase-4 plan-conformance gate (Step 18b), which can surface plan↔code
> drift the builder did not self-report — they are complementary, not redundant.*

This is a hard escalation gate, NOT post-hoc analysis. First decide whether a **retreat
trigger** is present:

- **Trigger present** when the test reveals a plan flaw, OR traceability reveals
  pervasive problems.
- **No trigger** when tests are green and traceability is clean.

**If a trigger IS present:** write `specs/2026-06-23-onnx-selfplay-loop/retreat.md` and fire a
**MANDATORY STOP that NEVER auto-skips** (it is exempt from the green-path skip clause
below). **Default: A (Retreat to Phase 2)** — almost always correct; fixing a revealed
plan flaw beats papering over it. Surface the alternatives only with a concrete
justification: B) Patch (slip small enough to handle in-build) or C) Accept with
documented limitation (divergence is doc-only or a deferred follow-up is filed).
Present via `AskUserQuestion` with A marked **(Recommended)**.

- [ ] Retreat triggers found: **yes / no**
- [ ] _(if yes)_ **MANDATORY STOP** fired (never auto-skipped); `retreat.md` written;
  decision recorded: _(A retreat / B+justification / C+justification)_

### Combined MANDATORY STOP

- [ ] **MANDATORY STOP**: Green + affected tests presented.
  **Skip when**: test passed green AND no affected-test regressions. (Does NOT cover the
  retreat gate above — when a retreat trigger is present, that gate's MANDATORY STOP
  fires regardless of this skip clause.)

Evidence (Green): result= | fixes= | artifacts=
Evidence (Affected): tests= | result= | regressions=

---

## Step 15b: Find Siblings (FIX mode only)

> **MODE GATE:** FIX only.

Grep for the same pattern elsewhere. Fix all instances.

- [ ] Sibling search completed
- [ ] Siblings fixed (or "none found")

---

## Step 16: Unit Tests + Compliance Checks

### Unit tests
- [ ] Written (or "none needed")

### Property-based tests (if business logic/validation/transforms added)
- [ ] Written with the language-idiomatic framework — `fast-check` (TS/JS),
  `hypothesis` (Python), `proptest` (Rust). (Or "skipped — no business logic".)

### Contract tests (if API endpoints added/modified)
- [ ] Written (or "skipped — no API changes")

### Plan fidelity
Compare plan vs `progress.md`. Fidelity = (DONE + justified MODIFIED) / total.

> *This is the builder's self-assessed score; it is reconciled at the Phase-4
> plan-conformance gate (Step 18b) by an INDEPENDENT reviewer that re-derives fidelity from
> the diff + decisions + MAP, never trusting this number. A large gap between the two is the
> signal worth investigating.*

- [ ] Score calculated, spec intent assessed, deviations justified

### Security checklist
- [ ] No new secrets committed (the gitleaks gate in Step 13d covers this — confirm passed)
- [ ] All new external inputs validated at the boundary
- [ ] Data scoping — any new read returns only data the caller is authorized for
  (per-user / per-tenant / per-workspace / per-project — whatever the repo's
  isolation model is)
- [ ] No PII in new logging statements
- [ ] Auth on any new endpoints

---

## Phase 3 Complete — Handoff

### Step 15 Completion Gate

> **BLOCKING.** Check Green ticked + any test artifacts the framework produces
> are committed. Fail → STOP, go back to Step 15.

- [ ] Step 15 verified (or exception label from Step 11)

### Build output

Write `specs/2026-06-23-onnx-selfplay-loop/build-output.md`: summary, files changed, gate status, traceability,
test results, open issues, plan fidelity, security checklist status.

### Sync feature branch with origin/self-play-data-plane

Before advancing state, pull in any work that landed on `self-play-data-plane` while this
feature was in flight. Idempotent — safe on resume; no-op when already up to date.
Helper exports `LAST_SYNC_COMMIT` and `LAST_SYNC_EVENT` for the state-write below.

```bash
source /Users/j/.claude/feature-workflow-internal/lib/sync-with-base.sh 3 || exit 1
```

### Update state

```bash
cat > specs/2026-06-23-onnx-selfplay-loop/FEATURE_STATE.json << STATEEOF
{ "slug": "onnx-selfplay-loop", "started": "2026-06-23", "mode": "$MODE", "size": "$SIZE",
  "current_phase": 4, "current_step": 17, "last_completed_step": 16,
  "spec_dir": "specs/2026-06-23-onnx-selfplay-loop", "repo_root": "/Users/j/Unciv-onnx-selfplay-loop",
  "last_sync_commit": "$LAST_SYNC_COMMIT", "last_sync_event": "$LAST_SYNC_EVENT" }
STATEEOF
```

Note: heredoc delimiter is unquoted `STATEEOF` (not `'STATEEOF'`) so `$LAST_SYNC_*`
expand. All other `$VAR` in this heredoc were already substituted into the checklist
copy by the `/feature` orchestrator at the start of the run.

- [ ] `build-output.md` written
- [ ] `FEATURE_STATE.json` updated to phase 4
- [ ] Cost log: _(total spend)_

**Phase 3 complete.** The orchestrator will now load `phase4-ship.md`.
