# $NAME — Phase 4: Ship

Progress tracker for Steps 17–23. Tick boxes and fill evidence after EVERY step.
Save this file to disk after each step.

> **Slug**: `$SLUG` | **Branch**: `$SLUG` | **Started**: $DATE
> **Spec folder**: `$SPEC_DIR/` | **Mode**: _(from discovery-output.md)_
> **Size**: _(from discovery-output.md)_

**RULES:** (1) MANDATORY STOP = present output and WAIT for user response — **unless**
the step produced nothing decision-worthy or surprising (council=APPROVE clean, CI
green, no findings). In that case, print a one-line status and continue. Error
refusals and the Step 22 merge decision NEVER skip. When in doubt, stop.
(2) Questions via `AskUserQuestion` with 2-4 options. (3) One step at a time.
(4) BLOCKING = KILL, gate fail ×3, policy violation. (5) Artifacts in `$SPEC_DIR/`.
(6) Update `FEATURE_STATE.json` after each step. (7) Log OpenRouter calls to
`cost-log.jsonl`. (8) **Telemetry**: log step start/end to `$SPEC_DIR/timings.jsonl`
(`{"step": N, "phase": 4, "event": "start|end", "ts": "..."}`).
(8b) **Q&A capture**: after every `AskUserQuestion` gate, append one line to
`$SPEC_DIR/questions.jsonl`: `{"ts", "phase", "step", "header", "question",
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

Read: `build-output.md`, `plan.md`, `decisions.md`, `council-plan-review.md` (if exists).

- [ ] Phase 3 artifacts read

---

## Step 17: Create PR

**BLOCKING PRE-CHECK:**
1. Read `FEATURE_STATE.json` — confirm `last_completed_step >= 15`
2. If Step 11 used an exception label, it carries forward.

**If fails: REFUSE.** Print `BLOCKED: Step 15 not completed.` and STOP.

### Pre-PR sync with origin/$BASE_BRANCH

Phase 3 build can run long enough for `$BASE_BRANCH` to advance again between the
3→4 handoff and now. Pull in any new commits before opening the PR so the diff
reflects only this feature's work (and CI runs against an up-to-date base).

```bash
source $LIB_DIR/sync-with-base.sh 4 || exit 1

# Record the pre-PR sync result. This is the only mid-step state-write in the
# workflow — it touches ONLY the two sync-tracking fields; phase/step ownership
# remains with the Phase 3 handoff that just ran.
python3 - <<PYEOF
import json, pathlib
p = pathlib.Path("$SPEC_DIR/FEATURE_STATE.json")
state = json.loads(p.read_text())
state["last_sync_commit"] = "${LAST_SYNC_COMMIT}"
state["last_sync_event"] = "${LAST_SYNC_EVENT}"
p.write_text(json.dumps(state, indent=2) + "\n")
PYEOF
```

### Create PR

PR from `$SLUG` → `$BASE_BRANCH`. Description MUST include:
- [ ] **What & Why**
- [ ] **Decisions** (from `decisions.md`)
- [ ] **How this works** _(if touching auth, tenant isolation, billing, encryption, data export)_
- [ ] **Schema changes** _(if any)_
- [ ] **Security** _(auth changes, new endpoints, data exposure)_
- [ ] **Feature flag** _(if significant new feature)_
- [ ] **Tests** _(test files added/modified; results summary)_
- [ ] **Plan fidelity** _(score + deviations)_
- [ ] **Policy compliance**
- [ ] **Files** _(grouped by purpose)_
- [ ] **Cost summary** _(from cost log)_

### Cleanup opportunities — address-now / defer / drop

If `$REPO_ROOT/.feature-workflow/cleanup-opportunities.md` exists AND contains items in
`[ ]` (unaddressed) state, surface them to the user via `AskUserQuestion` *per item*
(or grouped if there are ≤4). Options:

- **address-now** — invoke `/cleanup --items <id1>,<id2> --on-current-branch` to apply
  the cleanups in a mini-build pass on this PR's branch. The mini-build runs the per-batch
  audit + gates from `templates/cleanup-checklist.md` scoped to the chosen IDs, commits
  the result on the current branch (amending the PR), and flips the artifact items to
  `[x]` (via `cleanup_artifact.mark_addressed`, which preserves the sort invariant).
  On gate failure: revert + `gh pr comment` with the log; items stay `[ ]`.
- **defer** — leave the item `[ ]`. Standalone `/cleanup --address-oldest` will pick
  it up later.
- **drop** — call `cleanup_artifact.mark_dropped(text, uuid, "user-dropped-at-phase4-YYYY-MM-DD")`
  which rewrites the line to `- [~] ... dropped:user-dropped-... -->` (schema-correct
  per `_STATE_FROM_MARK`).

**Stale-item handling (per-item, NOT artifact-level):** the central file aggregates
items from many `/feature` runs, each with its own `source_commit:` metadata. Before
each `AskUserQuestion`, check that specific item's `source_commit:` field:

- If the field is missing or empty, log "no source_commit on item <uuid> — treating
  as fresh" and present the prompt normally.
- If the field exists and `is_stale_for_head(item.meta['source_commit'])` returns
  True, prefix the prompt with **"(stale — source code has moved since this item
  was flagged)"** so the user can still choose address-now/defer/drop with context.
  Don't skip stale items at scan time — `/cleanup` discovers and auto-drops stale
  ones at action time anyway, but the user might still want to manually drop here.

This per-item approach replaces the v0.6 artifact-level stale check (which would
mark the WHOLE central file stale based on a single commit — meaningless now that
the file accumulates across features).

**Non-TTY fallback:** if the session has no interactive TTY (e.g. running in CI), default
all items to **defer** and log the artifact path in the session summary. The artifact
stays `[ ]` for `/cleanup` later.

- [ ] Central `cleanup-opportunities.md` checked: present / absent / lite (no items)
- [ ] Per-item AskUserQuestion presented (or "defer all" if non-TTY)
- [ ] Per-item stale-check ran for each `[ ]` (warning prefix added where applicable)
- [ ] Address-now items processed via mini-build invocation
- [ ] Drop items rewritten via `cleanup_artifact.mark_dropped` (writes `[~]` + reason)
- [ ] Defer items left `[ ]` (no change)
- [ ] If any artifact mutation occurred, written via `atomic_write` and committed
  alongside the address-now mini-build commit

Evidence: artifact_state= | items_total= | address-now= | defer= | drop= | mini_build_result=

### Self-Healing Retrospective

> Universal — runs once per feature, covers all prior phases. (S, M, L all run this.
> Phase 1-3 retrospectives were removed; this is the single retro.)

1. **Workarounds** — hardcoded values, `any` types, inline hacks?
2. **Silent failures** — ignored warnings, vacuous tests, suppressed lints?
3. **Skipped steps** — items skipped without SIZE/MODE GATE exception?
4. **Corner-cuts** — items marked done without verification?

- [ ] Scan 1 — Workarounds: _(finding or "clean")_
- [ ] Scan 2 — Silent failures: _(finding or "clean")_
- [ ] Scan 3 — Skipped steps: _(finding or "clean")_
- [ ] Scan 4 — Corner-cuts: _(finding or "clean")_
- [ ] All findings fixed

- [ ] Print one-line notification: `PR created: <url> — launching council review` and
  proceed to Steps 18/19 without waiting. The PR URL is already in `build-output.md`
  and council runs next; the old MANDATORY STOP here was pure ceremony.

Evidence: PR_URL=

---

## Steps 18 + 19: Council Review (Code) + CI Green Check

⚡ **PARALLEL:** Launch both simultaneously after PR creation.

### Step 18: Council Review (Code)

**ISOLATION RULE:** Each reviewer = independent sub-agent or external LLM call.

**COUNCIL CONTINUITY:** Pass `$SPEC_DIR/council-plan-review.md` (the Phase 2
Step 11 verdict) into `--context` so code reviewers verify plan concerns were
addressed.

### Pick roster (Phase 4)

Read `$LIB_DIR/llm/council/roles/lenses.md`. Start from the Phase 2 roster
(`$SPEC_DIR/roster-phase2.json`) as a baseline, then adjust based on what was
actually built: drop roles whose lens never fired, add roles whose lens became
relevant in the diff. Cap per size budget (S=6, M≤10, L≤14). Write
`$SPEC_DIR/roster-phase4.json`. Record changes vs Phase 2 in `decisions.md`.

First write a **source-scoped diff to a file** (the council target must be a file path —
`build_context` reads it; an inline `$(git diff)` blob fails, and an unscoped diff bloats
the context with spec artifacts):

```bash
mkdir -p $SPEC_DIR/.council-ship
# Exclude **/MAP.md — the codebase-map diff is a HUMAN artifact reviewed separately at
# Step 21, not by the code council (keeps code review focused; avoids map churn noise).
git diff "$BASE_BRANCH" -- lib/ services/ templates/ tests/ ':(exclude)**/MAP.md' > $SPEC_DIR/.council-ship/diff.patch
```

Then run the **council adaptive loop** (`$TEMPLATES_DIR/council-adaptive-loop.md`). Claude
drives ship-review rounds and consolidates in-session. `$OPENROUTER_API_KEY` is guaranteed
present. Phase 4 uses the ship-review prompt (regressions, edge cases, observability gaps in
the diff) instead of intake (design-gaps in the brief). Parameters:

- `ROSTER=$SPEC_DIR/roster-phase4.json`
- `REVIEWER_TEMPLATE=$LIB_DIR/prompts/ship/reviewer.md`
- `TARGET=$SPEC_DIR/.council-ship/diff.patch`
- `CONTEXT=$SPEC_DIR`
- `OUTPUT=$SPEC_DIR/.council-ship/`
- `CAP=2` — round-cap backstop (lowered from 3 in council-config-tune, 2026-06-23: round-3
  telemetry showed round 3 first-sees only ~0.4% of findings). The adaptive stop is the real
  control and usually ends sooner. Policy + rationale: `$DOCS_DIR/invariants/council-rounds.md`.
- `PHASE=4-ship`, `STEP=18`, `SLUG=$SLUG`

A non-zero exit (1 error, 2 partial-reviewer-failure) is BLOCKING — do NOT fall back to
subagents. Exit 4 re-merges and retries per the loop doc.

**Kill criteria:** cross-tenant leak, injection, missing auth, hardcoded secret, vacuous test.

- [ ] Plan review concerns passed via `--context`
- [ ] Roles selected
- [ ] Council ran (independent agents/LLM calls)
- [ ] Output saved to `$SPEC_DIR/council-code-review.md`
- [ ] If REQUEST_CHANGES/BLOCK: fix, re-run
- [ ] Final verdict: APPROVE (or STEAL-only)

### Step 19: CI Green Check

```bash
gh pr checks <PR_NUMBER> --watch
```

`gh pr checks --watch` exits non-zero if any required check fails — that *is* the
hard gate. If checks fail, fix the failure and push; don't merge through red CI.

- [ ] CI green (`gh pr checks --watch` exited 0)

Evidence (council): roles= | verdicts= | iterations= | final=
Evidence (CI): status= | fixes=

---

## Step 18b: Plan-conformance + architecture-fidelity gate

An INDEPENDENT check that the built code actually matches `plan.md` + `decisions.md` +
the regenerated `MAP.md`, WITHOUT trusting the builder's Phase-3 self-report (the completion
oracle, the fidelity score, `progress.md`). Two signals, different failure modes:
1. a **semantic reviewer** (thinking model + reasoning budget) → ADR-style report
   (matches / 3-way deviations / unrecorded architectural decisions / untested plan items),
2. an **independent test re-run** of the repo's own suite (real pass/fail, not reported).

**Advisory — NEVER blocks the merge.** It surfaces a human-review artifact; observability,
not enforcement. `progress.md` is fed as a CLAIM to challenge, never as ground truth.

```bash
# 1. Regenerate MAP.md from the ACTUALLY-shipped code (moved here from Step 21 so the
#    conformance reviewer sees the real architecture). SHA-keyed → cheap; never blocks.
SOURCE_COMMIT=$(python3 -c "import json; print(json.load(open('$SPEC_DIR/FEATURE_STATE.json')).get('source_commit_at_start',''))" 2>/dev/null || echo "")
MAP_OUT=$(timeout 180 python3 -m lib.ship.codebase_map --refresh \
  --repo-root "$REPO_ROOT" --worktree "$WORKTREE" --base "$BASE_BRANCH" \
  --spec-dir "$SPEC_DIR" --source-commit "$SOURCE_COMMIT" \
  2>&1 || echo "codebase-map ship refresh skipped (non-blocking)")
# Surface the one-line backfill notice ONLY when a MAP.md was seeded or drift refreshed
# (silent on no-op). The seed/drift decision is made in Python; this just echoes it.
printf '%s\n' "$MAP_OUT" | grep -a '^📍 codebase-map:' || true

# 2. Capture the MAP diff (architecture delta = EVIDENCE for the reviewer, not the oracle).
#    `git add -N` first so a brand-new MAP.md (first feature to touch a package, e.g. when
#    the Phase-2 plan-time refresh was skipped) still shows up in the diff.
git -C "$WORKTREE" add -N $(git -C "$WORKTREE" ls-files --others --exclude-standard '**/MAP.md') 2>/dev/null || true
git -C "$WORKTREE" --no-pager diff "$BASE_BRANCH" -- '**/MAP.md' > $SPEC_DIR/.council-ship/map-diff.txt 2>/dev/null || true

# 3. Build a CONFORMANCE-scoped diff. NOTE: this is BROADER than the Step-18 code-council
#    diff — conformance cares about ALL plan deliverables, not just code, so it includes
#    docs/ (and commands/ scripts/) AND root-level *.md (README.md, CHANGELOG.md, …) via
#    the glob-magic pathspec ':(glob)*.md' (the :(glob) prefix anchors `*` so it matches
#    ONLY top-level markdown, never specs/**/*.md or a nested MAP.md). Without it, a plan
#    item that ships a root doc reads as "missing from the diff" — a false corner-cut /
#    "fabricated self-report" deviation (hit on parallelize-phase2 T4b). Still excludes
#    MAP.md (own artifact) and specs/ (the plan/decisions are already in --context).
#    `git add -N` any UNTRACKED files in scope first — a brand-new module/test/doc not yet
#    committed would otherwise be invisible to `git diff` and read as a false "missing" /
#    "untested" deviation.
git -C "$WORKTREE" add -N -- $(git -C "$WORKTREE" ls-files --others --exclude-standard \
  lib/ services/ templates/ tests/ docs/ commands/ scripts/ ':(glob)*.md') 2>/dev/null || true
git diff "$BASE_BRANCH" -- lib/ services/ templates/ tests/ docs/ commands/ scripts/ \
  ':(glob)*.md' ':(exclude)**/MAP.md' > $SPEC_DIR/.council-ship/conformance-diff.patch

# 4. Run the gate (semantic reviewer + independent test re-run). Exits 0 always.
timeout 900 python3 -m lib.conformance.gate \
  --spec-dir "$SPEC_DIR" --diff "$SPEC_DIR/.council-ship/conformance-diff.patch" \
  --map-diff "$SPEC_DIR/.council-ship/map-diff.txt" \
  --output "$SPEC_DIR/.council-ship" --repo-root "$WORKTREE" \
  --feature-slug "$SLUG" 2>&1 || echo "conformance gate skipped (non-blocking)"
```

Print `$SPEC_DIR/.council-ship/plan-conformance-review.md` **inline** for the human. Read it
as a reviewer: confirm any 🔴 corner-cut deviations against their cited `file:line`, sanity-
check unrecorded architectural decisions (file an ADR / amend `decisions.md` if real), and
note whether the independent test re-run agrees with what Phase 3 reported. The reviewer
over-flags by design — low-confidence items are leads, not verdicts.

- [ ] MAP regenerated from shipped code; map diff captured
- [ ] Conformance gate ran (or "skipped — non-blocking"); report printed inline
- [ ] `conformance-report.json` exists and is valid JSON
- [ ] Independent test re-run result noted (agrees / disagrees with Phase-3 report)
- [ ] 🔴 corner-cut deviations triaged (confirm/dismiss); unrecorded decisions actioned (amend `decisions.md` / file ADR / accept)

Evidence (conformance): status= | corner_cut= | equivalent= | improvement= | unrecorded= | untested= | test_rerun= | arch_fidelity=

---

## Step 20: Finalize

Bug log (FIX) + verify and commit all phase artifacts to the branch. (Condense, Merge,
and Wrap-up are now their own steps — 21, 22, 23 — so none of them can be silently
skipped after a terminal action.)

### Bug log (FIX mode only)

> **MODE GATE:** FIX only.

If the repo has a bug-log convention, record a bug entry there. The "N/A — no
convention" fallback is **only** valid after probing all three canonical
locations and recording the result. Vague "N/A" without probes is a corner-cut.

**Probe (all three required before fallback):**

```bash
# 1. process/bugs/ with BUG-NNN.json
test -d process/bugs && ls process/bugs/BUG-*.json 2>/dev/null | head -1

# 2. docs/bugs/
test -d docs/bugs && ls docs/bugs 2>/dev/null | head -1

# 3. GitHub Issues enabled on the repo
gh issue list --limit 1 2>/dev/null
```

If ANY probe returns a result, that location IS the convention — log the bug
there. Only when ALL three probes come back empty/absent may the "Decisions"
section of the PR description carry the FIX rationale instead.

- [ ] _(FIX only)_ Probe results recorded: `process/bugs/`=___, `docs/bugs/`=___,
  `gh issue list`=___
- [ ] _(FIX only)_ Bug logged at ___, OR all three probes empty → rationale in
  PR Decisions section

### Finalize

- [ ] All phase artifacts verified: `ls $SPEC_DIR/`
- [ ] Every box in all phase checklists ticked
- [ ] Commit artifacts: `git add $SPEC_DIR/ && git commit -m "spec($SLUG): complete all phases"`

---

## Step 21: Condense

Finalize durable artifacts **before** the merge so they ride the squash into the base.
The manifest + event run first (they read `timings.jsonl`, which the prune removes);
then `summary.md`, prune, commit, push.

### Generate telemetry manifest

Before pruning, generate the run manifest from accumulated telemetry data.

```bash
# Collect timing data (one JSON object per line; safe to pass raw).
TIMINGS=$(cat $SPEC_DIR/timings.jsonl 2>/dev/null || echo "")

# Collect cost data. cost-log.jsonl is one record per OpenRouter call with
# {"usd": <float>, "model": "...", "phase": N, "step": M}. Use jq if present;
# otherwise awk a total. Falls back to {} when the log is missing.
if [ -f "$SPEC_DIR/cost-log.jsonl" ]; then
  if command -v jq >/dev/null 2>&1; then
    COSTS=$(jq -s '{total_usd: (map(.usd // 0) | add), n_calls: length,
                    by_phase: (group_by(.phase // "?") | map({(.[0].phase|tostring): (map(.usd // 0) | add)}) | add)}' \
              "$SPEC_DIR/cost-log.jsonl")
  else
    TOTAL=$(awk -F'"usd":' 'NF>1 {split($2, a, ","); sum += a[1]} END {printf "%.4f", sum+0}' "$SPEC_DIR/cost-log.jsonl")
    N=$(wc -l < "$SPEC_DIR/cost-log.jsonl" | tr -d ' ')
    COSTS="{\"total_usd\": $TOTAL, \"n_calls\": $N}"
  fi
else
  COSTS="{}"
fi

# Collect git stats. $BASE_BRANCH was exported by /feature in commands/feature.md.
FILES_CHANGED=$(git diff "$BASE_BRANCH" --stat | tail -1 | grep -oE '[0-9]+ file' | grep -oE '[0-9]+' || echo 0)
LINES_ADDED=$(git diff "$BASE_BRANCH" --numstat | awk '{s+=$1} END {print s+0}')
LINES_REMOVED=$(git diff "$BASE_BRANCH" --numstat | awk '{s+=$2} END {print s+0}')
```

Write `$SPEC_DIR/manifest.json` with the structure defined in the plan. Include:
- **slug, date, mode, size_classified** — from FEATURE_STATE.json
- **size_actual** — files_changed, lines_added, lines_removed from git diff
- **timing** — derive phase durations from timings.jsonl (first start to last end per phase)
- **llm_calls** — from cost-log.jsonl (count and cost per script)
- **human_waits** — count MANDATORY STOPs that were actually triggered (from checklist ticks)
- **questions** — count asked, rounds used vs skipped
- **council** — plan + code review verdicts, iterations, findings count
- **quality** — gate pass/fail, build iterations, traceability, lint, fidelity
- **tests** — agent test result, affected specs, unit tests added, mutation score
- **conformance** — from `$SPEC_DIR/.council-ship/conformance-report.json` (Step 18b):
  `status`, deviation counts by bucket (corner_cut / justified_equivalent / improvement),
  unrecorded-decision count, untested-item count, `architecture_fidelity.verdict`, and the
  independent `test_rerun.passed` (note when it disagrees with the Phase-3 self-report)

- [ ] `manifest.json` generated

### Emit feature_run_summary event (event-log)

Emit the comprehensive run-summary event (+ `step_timing` / `user_question`
children) into the event-log. This is the runtime trigger for the cost+timing
rollup — it reads `FEATURE_STATE.json`, `timings.jsonl`, `cost-log.jsonl`,
`questions.jsonl` and computes git stats itself (no manifest.json dependency).
Best-effort and **timeout-wrapped so it can never block a release**:

```bash
# Guard: an unset $WORKTREE would expand to --worktree "" → fall back to --repo-root
# (the main repo on base) → empty range → the exact zero-stats bug this fix removes.
[ -n "$WORKTREE" ] || echo "WARN: \$WORKTREE unset — run-summary git stats will be empty (see _git_stats)." >&2
timeout 30 python3 -m lib.observability.summary --emit-run-summary \
  --spec-dir "$SPEC_DIR" --repo-root "$REPO_ROOT" --base-branch "$BASE_BRANCH" \
  --worktree "$WORKTREE" --session-id "$SLUG" 2>&1 || echo "run-summary emission skipped (non-blocking)"
```

- [ ] `feature_run_summary` event emitted (or skipped — non-blocking)

### Emit code_map event (event-log) — powers code-reuse recall

Index the reusable code this feature shipped so a future `/feature` run can recall it
(`--kind code`). Reads `plan.md`'s `## WHAT` for intent + the source files changed vs
`$BASE_BRANCH`. Best-effort and timeout-wrapped — **never blocks a release**:

```bash
# --worktree "$WORKTREE": the changed-files diff MUST be computed in the FEATURE
# worktree, not --repo-root (the main worktree on $BASE_BRANCH, where base...HEAD
# is empty so files would never be indexed).
timeout 20 python3 -m lib.ship.code_map --emit \
  --repo-root "$REPO_ROOT" --slug "$SLUG" --spec-dir "$SPEC_DIR" \
  --base "$BASE_BRANCH" --worktree "$WORKTREE" 2>&1 || echo "code_map emission skipped (non-blocking)"
```

- [ ] `code_map` event emitted (or skipped — non-blocking)

### Verify the codebase map — present the map DIFF + commit

`MAP.md` was already regenerated from the ACTUAL shipped code at **Step 18b** (the
plan-conformance gate consumes the map diff), so this step does NOT re-refresh — it presents
the map diff for human review and ensures the regenerated `MAP.md` files ride the condense
commit. (If Step 18b was skipped, run the refresh here as a fallback — see 18b's command.)

```bash
# Present the map diff to the human (separate from the code diff). map-refresh-log.md
# annotates which lines were stale-on-entry-refreshed vs changed-by-this-feature.
git -C "$WORKTREE" --no-pager diff "$BASE_BRANCH" -- '**/MAP.md' || true
```

Show the map diff inline and call out anything where the description changed in a way the
code diff alone wouldn't reveal (a file's responsibility drifted, a new cross-concern) — the
Step-18b conformance report already flags these as unrecorded architectural decisions. If a
cohesion smell was emitted, it is already in `.feature-workflow/cleanup-opportunities.md`
(flag, don't gate). Commit the regenerated `MAP.md` files with the condense commit.

- [ ] Map diff presented (regenerated at Step 18b; fallback-refreshed here only if 18b skipped)
- [ ] `map-refresh-log.md` consulted (stale-on-entry vs changed-by-feature)

### Telemetry delivery gate (hard-block — event-log-fs-leak D3/C2)

Refuse to finish the run while this run's telemetry is still undelivered to the GCP
event-log. This is **boundary** enforcement (run cannot COMPLETE), not a mid-operation
kill: transient blips during the run queue to the gitignored `0o600` fallback, and this
gate reconciles them before the squash-merge so a `/feature` run can never silently lose
telemetry. Bounded (timeout) + audited bypass so it can neither hang nor trap a release.

```bash
# Runs in a subshell so a BLOCKING `exit 1` terminates only this gate, not the
# orchestrator (per the template's subshell expectation).
(
  set -e
  _count() { python3 -c "from lib.observability.summary_stats import count_fallback_events as c; print(c('$REPO_ROOT'))" 2>/dev/null || echo 0; }
  QCOUNT=$(_count)
  if [ "${QCOUNT:-0}" -gt 0 ]; then
    echo "telemetry-gate: $QCOUNT event(s) queued in the fallback — attempting bounded reconcile…" >&2
    # timeout is a backstop; reconcile writes the queue via atomic tmp+rename, so a
    # SIGTERM cannot corrupt it. Derive --expected-org-id FROM THE QUEUE so reconcile
    # can never push cross-tenant (ship-council): if all queued events share one
    # org_id, pass it; mixed/unknown → omit + warn (operator override:
    # FEATURE_EVENT_LOG_ORG_ID).
    ORG=$(python3 -c "
import json
orgs=set()
try:
    for l in open('$REPO_ROOT/.feature-workflow/event-log-fallback.jsonl'):
        l=l.strip()
        if l: orgs.add(json.loads(l).get('event',{}).get('org_id'))
except OSError: pass
print(next(iter(orgs)) if len(orgs)==1 else '')" 2>/dev/null || echo "")
    [ -z "$ORG" ] && ORG="${FEATURE_EVENT_LOG_ORG_ID:-}"
    ORG_FLAG=""
    if [ -n "$ORG" ]; then ORG_FLAG="--expected-org-id $ORG"; else
      echo "telemetry-gate: org_id not uniform/unknown — reconcile runs without --expected-org-id" >&2
    fi
    timeout 60 python3 -m lib.observability.fallback reconcile \
      --config "$REPO_ROOT/.feature-workflow/event-log.yaml" \
      --fallback-path "$REPO_ROOT/.feature-workflow/event-log-fallback.jsonl" \
      $ORG_FLAG 2>&1 || echo "telemetry-gate: reconcile exited non-zero (transient remaining)" >&2
    QCOUNT=$(_count)
  fi
  if [ "${QCOUNT:-0}" -gt 0 ]; then
    # Reason-aware: an auth lapse needs re-auth, not a blind retry.
    python3 -c "from lib.observability.summary_stats import count_fallback_by_reason as c; import json; b=c('$REPO_ROOT'); print('telemetry-gate: queued by reason:', json.dumps(b), file=__import__('sys').stderr); (print('telemetry-gate: re-auth required — run: gcloud auth application-default login', file=__import__('sys').stderr) if b.get('backend_unavailable_auth') else None)" || true
    if [ -n "${FEATURE_SHIP_TELEMETRY_SKIP:-}" ]; then
      # Audited bypass — durable record in decisions.md (survives the condense keep-list).
      printf -- '- **TELEMETRY-BYPASS** %s by %s — shipped with %s undelivered event(s); reason=%s\n' \
        "$(date -u +%FT%TZ)" "${USER:-unknown}" "$QCOUNT" "${FEATURE_SHIP_TELEMETRY_SKIP_REASON:-unspecified}" \
        >> "$SPEC_DIR/decisions.md"
      echo "telemetry-gate: BYPASSED via FEATURE_SHIP_TELEMETRY_SKIP (audited in decisions.md)." >&2
    else
      echo "BLOCKING: $QCOUNT event(s) still undelivered to the GCP event-log after reconcile." >&2
      echo "  Telemetry must reach GCP before merge (D3). Fix delivery (re-auth / backend reachable)," >&2
      echo "  then re-run Step 21. Emergency bypass (audited):" >&2
      echo "    FEATURE_SHIP_TELEMETRY_SKIP=1 FEATURE_SHIP_TELEMETRY_SKIP_REASON='<why>' <re-run>" >&2
      exit 1
    fi
  fi
)
[ $? -ne 0 ] && { echo "STOP: telemetry delivery gate blocked the ship." >&2; exit 1; }
```

- [ ] Fallback queue empty after bounded reconcile (or audited bypass recorded in `decisions.md`)
- [ ] If blocked: delivery fixed (re-auth / backend up) and Step 21 re-run

Evidence (telemetry-gate): queued_before= | reconciled= | queued_after= | bypass=

### Condense

Write `$SPEC_DIR/summary.md`: feature one-liner, verdict (write **SHIPPED** — this file
is committed to the branch and only becomes public if the Step 22 squash-merge happens,
so no post-merge reversal is needed), PR URL, council outcome, list of files touched
(paths + brief role), open issues (if any), and any decisions worth surfacing. ≤80 lines.

Then prune `$SPEC_DIR/` to a stable keep-list. Everything outside the list gets
removed — the goal is "a future maintainer can read 5 small files and understand
this feature" rather than wading through per-step artifacts.

```bash
# Keep these — the rest of $SPEC_DIR/ is intermediate scaffolding.
# plan.html is the approved review page (audit trail of what was approved); the
# .approval-nonce sidecar and .plan-body*.html/.plan-diagram.json scratch files are
# NOT kept (single-use, regenerated per render) and fall through to prune.
# .plan-meta.json IS kept (plan-review-signal D9/D21): it is the machine-readable export
# the plan.html "Audit & sources" block links + the structured substance the redesigned
# page renders; dropping it would break the audit cross-check and any tooling reading it.
# event-log-fs-leak D1: manifest.json + cost-log.jsonl are per-run TELEMETRY whose
# durable home is the GCP `feature_run_summary` event — committing them duplicates
# GCP and is the metadata the repo should not carry. They fall through to prune.
KEEP="summary.md plan.md plan.html decisions.md .plan-meta.json"
for f in "$SPEC_DIR"/*; do
  name=$(basename "$f")
  case " $KEEP " in
    *" $name "*) ;;
    *) git rm -r --ignore-unmatch -- "$f" 2>/dev/null || rm -rf -- "$f" ;;
  esac
done
git add "$SPEC_DIR/"
git commit -m "spec($SLUG): condense to summary"
# Push condensed artifacts to the PR branch BEFORE the Step 22 merge. BLOCKING:
# if the push fails, the condense commit is local-only and would NOT ride the
# squash — STOP rather than merge without it.
git push || { echo "BLOCKING: condense push failed — fix and re-push before merging." >&2; exit 1; }
```

- [ ] `summary.md` written (≤80 lines, verdict SHIPPED)
- [ ] `$SPEC_DIR/` pruned to keep-list (summary, plan, plan.html, decisions — telemetry
  manifest.json/cost-log.jsonl intentionally dropped per D1; lives in GCP)
- [ ] Condense commit pushed to the PR branch (so it lands in the squash)

Evidence (condense): manifest= | event= | summary_lines= | pruned= | pushed=

---

## Step 22: Merge

### Merge decision

- [ ] Ask via `AskUserQuestion`: Merge PR now?
- [ ] **MANDATORY STOP**: Wait for decision
- [ ] If yes: `gh pr merge <PR_NUMBER> --squash --delete-branch` (carries the Step 21
  condensed artifacts into the squash)
- [ ] If no: ask about branch deletion (branch + `summary.md` persist un-published; a
  SHIPPED `summary.md` only becomes public via the squash-merge, so no reversal)

> ⚠️ Merging does NOT end the run — **Step 23: Wrap-up** (worktree removal + session
> summary) still follows below. Do not stop here.

Evidence (merge): decision= | pr_state=

---

## Step 23: Wrap-up

### Remove worktree

```bash
cd $REPO_ROOT
git worktree remove $WORKTREE
```

If fails: warn user. After removal:
```bash
git push origin --delete $SLUG 2>/dev/null || true
git branch -D $SLUG 2>/dev/null || true
git remote prune origin
```

- [ ] Worktree removed (or user notified)
- [ ] Remote + local branch cleaned up

### Session Summary

Print three sections:

**1. Session Status**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SESSION STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PR:       <merged / open #URL / not created>
CI:       <green / pending / failed>
Tests:    <passing / failing / not run>
Open issues: <list or "none">
Cost:     <total OpenRouter spend>

VERDICT:  <SAFE TO CLOSE / WORK REMAINING — reason>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**2. What Was Built** — 3-5 bullet points

**3. Manual Verification Steps** — 3-8 concrete steps with expected results

- [ ] Session status printed
- [ ] What Was Built printed
- [ ] Verification steps printed

Evidence: verdict= | PR_final= | CI_final= | open_issues= | cost=
