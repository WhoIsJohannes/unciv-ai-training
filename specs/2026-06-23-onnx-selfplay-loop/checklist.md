# ONNX policy bridge + self-play training loop — Phase 1: Discovery

Progress tracker for Steps 0–3. Tick boxes and fill evidence after EVERY step.
Save this file to disk after each step.

> **Slug**: `onnx-selfplay-loop` | **Branch**: `onnx-selfplay-loop` | **Started**: 2026-06-23
> **Spec folder**: `specs/2026-06-23-onnx-selfplay-loop/` | **Mode**: _(Step 0)_ | **Size**: _(Step 0)_

**RULES:** (1) MANDATORY STOP = present output and WAIT for user response — **unless**
the step produced nothing decision-worthy or surprising (council=APPROVE clean, scan=no
surprises, no findings, no exceptions). In that case, print a one-line status and
continue. Error refusals (e.g. "below the Opus minimum, STOP") and failure escalations (3-strike,
iter-3-fail, plan retreat) NEVER skip. When in doubt, stop. (2) Questions via
`AskUserQuestion` with 2-4 options. (3) One step at a time. (4) BLOCKING = KILL, gate
fail ×3, policy violation. (5) Artifacts in `specs/2026-06-23-onnx-selfplay-loop/`. (6) Update
`FEATURE_STATE.json` after each step. (7) Log OpenRouter calls to `cost-log.jsonl`.
(8) **Telemetry**: log step start/end to `specs/2026-06-23-onnx-selfplay-loop/timings.jsonl`
(`{"step": N, "phase": 1, "event": "start|end", "ts": "..."}`).
(8b) **Q&A capture**: after every `AskUserQuestion` gate, append one line to
`specs/2026-06-23-onnx-selfplay-loop/questions.jsonl`: `{"ts", "phase", "step", "header", "question",
"options": [...], "chosen", "source"}`. Consumed at ship by
`lib.observability.summary` → `user_question` events.
(9) **Default-rigor.** When a decision is binary AND one option matches established
rigor (write tests / fix root cause / hit real DB / reuse over rebuild / retreat
over patch / address vs forget) and the other is a corner-cut, choose rigor
without asking. Only surface `AskUserQuestion` when the user must provide info
Claude cannot determine: state (resume, dev URL), intent (BUILD/FIX, merge
yes/no), repo fact (FE_DIR/BE_DIR), or genuine tradeoff (UI mockup time budget,
council roster scope). Does NOT cover aesthetic/scope tradeoffs where "more"
isn't strictly more rigorous — those remain user decisions.

### Size Guide

| Size | Description | Steps affected |
|------|-------------|----------------|
| **S** | Well-defined, single-file or few-file change. No ambiguity. | Skip Step 2 (light scan), skip domain presets, 0-1 Qs in Step 3 |
| **M** | Multi-file feature, some design decisions needed. | Full workflow, combined stops where noted |
| **L** | Cross-cutting, multi-system, or high-risk feature. | Full workflow, all stops enforced individually |

---

## Step 0: Pre-flight

**Model check:** **Opus is the MINIMUM.** Verify the model via system context. The
passing set (and the effort/sub-agent rule) is defined once in
`docs/invariants/model-policy.md` — read it if your tier isn't obviously passing. If the
running model is not clearly in the passing set, STOP and ask the user — never self-assert
that an unlisted model qualifies.

**Effort:** Sub-agents needing deep reasoning inherit the session model (omit the
`model` param — the session already passed the Opus-minimum gate); never explicitly
downgrade them to a mid/light tier. (See `docs/invariants/model-policy.md`.)

Verify you are in the worktree on the correct branch.

```bash
git config core.hooksPath .githooks
```

### Resume check

Look for `specs/2026-06-23-onnx-selfplay-loop/FEATURE_STATE.json`. If found: display progress, ask user via
`AskUserQuestion` (A: Resume / B: Start fresh / C: Abort). If resuming: mark prior steps
`[x] RESUMED`, read checkpoint/phase output files, fast-forward to `current_step`.

### Size assessment

Assess size using signals below. State chosen size and move on — no user confirmation needed.

| Signal | S | M | L |
|--------|---|---|---|
| Files touched | 1-3 | 4-10 | 10+ |
| New DB tables/columns | 0 | 0-2 | 3+ |
| New API endpoints | 0-1 | 2-4 | 5+ |
| Design ambiguity | None | Some | Significant |
| Security surface | None | Minor | Auth/PII/billing |
| Cross-system | No | Maybe | Yes |

### Mode detection

Auto-detect: **BUILD** ("add", "create", "implement", "new", "redesign") or
**FIX** ("fix", "bug", "broken", "regression", "error", "crash"). Ambiguous → ask user.

### Save state

Capture **source_commit_at_start** — the build's starting HEAD before any commits land
in the worktree. Phase 3 Step 14 reads this for per-item `source_commit:` metadata in
`.feature-workflow/cleanup-opportunities.md`. Recording it once at Step 0 (not at scan
time) prevents drift across the three writers (Step 14, `/cleanup --address-oldest`,
`/cleanup-backfill`). See decisions.md D16 in `pre-feature-cleanup-nudge`.

```bash
SOURCE_COMMIT_AT_START=$(git rev-parse HEAD)
cat > specs/2026-06-23-onnx-selfplay-loop/FEATURE_STATE.json << STATEEOF
{ "slug": "onnx-selfplay-loop", "started": "2026-06-23", "mode": "$MODE", "size": "$SIZE",
  "current_phase": 1, "current_step": 1, "last_completed_step": 0,
  "spec_dir": "specs/2026-06-23-onnx-selfplay-loop", "repo_root": "/Users/j/Unciv-onnx-selfplay-loop",
  "source_commit_at_start": "$SOURCE_COMMIT_AT_START" }
STATEEOF
```

- [ ] Model meets the Opus minimum (see `docs/invariants/model-policy.md`; unknown → STOP and ask)
- [ ] Hooks configured
- [ ] In worktree on branch `onnx-selfplay-loop`
- [ ] Resume check: no prior state / resumed from step N / fresh start
- [ ] Mode: **BUILD** / **FIX**
- [ ] Size: **S** / **M** / **L**
- [ ] `source_commit_at_start` captured (= build-start HEAD before any new commits)
- [ ] `FEATURE_STATE.json` written

Evidence: model= | dir= | branch= | resume= | mode= | size=

---

## Step 1: Load Context

- [ ] Read `/Users/j/.claude/feature-workflow-internal/docs/invariants/features.md`. Identify the feature to build.
- [ ] If the repo has a per-feature PRD or spec doc (e.g. under `specs/`,
  `docs/specs/`), read the relevant sections.

### Context routing

Pick the **single best category** for which invariant docs to load (all paths are under
`/Users/j/.claude/feature-workflow-internal/docs/invariants/`):

| Category | Load |
|----------|------|
| **frontend** | `code-quality.md`, `security.md`, `user-experience.md` + **design guide** (see below) |
| **backend** | `code-quality.md`, `security.md`, `architecture.md` |
| **fullstack** | `code-quality.md`, `security.md`, `user-experience.md`, `architecture.md` + **design guide** (see below) |
| **infra** | `security.md`, `architecture.md` |
| **tooling** | `code-quality.md` |

Add `testing.md` if the feature involves tests.

**Design guide (frontend/fullstack only):** Read `/Users/j/.claude/feature-workflow-internal/docs/design/README.md` for the
index, then read the specific `/Users/j/.claude/feature-workflow-internal/docs/design/*.md` files relevant to the current
task (buttons, badges, colors, navigation, content-table, settings-page, help-guidance,
icons). These rules MUST be respected for any UI/UX work — do not guess design patterns.

⚡ **PARALLEL:** Read all routed docs in a single batch.

### Domain preset (M/L only)

> **SIZE GATE:** Skip for **S**.

Select domain preset — **hint** for which conditional council roles are likely
relevant in Phase 2. The Phase 2 Step 11 "Pick roster" sub-step reads
`lib/llm/council/roles/lenses.md` directly and may add or omit roles based on
the feature's actual prompt; the preset below is a starting bias, not a
force-include list.

| Preset | Auto-include roles | Weight boost |
|--------|-------------------|--------------|
| **cx-platform** | `domain_fidelity`, `end_user`, `accessibility` | UX, domain fidelity |
| **integrations** | `practitioner` (always-on), `cost_efficiency`, `data_privacy_legal` | reliability, latency |
| **auth-security** | `security_red_team` (always-on), `data_privacy_legal`, `compliance` | security, compliance |
| **data-pipeline** | `practitioner` (always-on), `data_privacy_legal`, `cost_efficiency` | data integrity, latency |
| **ui-ux** | `end_user`, `accessibility`, `power_user`, `support_agent` | UX, accessibility |
| **api** | `practitioner` (always-on), `security_red_team` (always-on), `b2b_buyer` | reliability, security |
| **tooling** | _(Core 6 only — no conditional bias)_ | maintainability, DX |
| **custom** | _(specify in `decisions.md`)_ | _(specify)_ |

- [ ] Category selected
- [ ] Routed docs loaded
- [ ] Domain preset selected (M/L)

### Prior-work recall (advisory — best-effort, never blocks)

Before asking the user anything, query the event-log for related prior decisions and
for how the user answered similar questions before. **Advisory only**: surface what
comes back as provenance-tagged context (source repo + date + score) — NEVER pre-fill
or auto-select an `AskUserQuestion` option from it. If recall returns nothing (or the
backend is down / opted-out), it prints nothing and you continue silently.

```bash
# Decisions/designs related to this feature, then how the user answered similar Qs.
timeout 20 python3 -m lib.observability.recall \
  --query "ONNX policy bridge + self-play training loop — onnx-selfplay-loop" --kind decision --repo-root "/Users/j/Unciv-onnx-selfplay-loop" --scope repo || true
timeout 20 python3 -m lib.observability.recall \
  --query "ONNX policy bridge + self-play training loop — onnx-selfplay-loop" --kind answer --repo-root "/Users/j/Unciv-onnx-selfplay-loop" --scope repo || true
```

- [ ] Prior-work recall ran (or "no hits / backend unavailable — continued")
- [ ] Ask via `AskUserQuestion`: Present context + initial understanding (include any
  🔎 recalled prior work as context the user may draw on) and ask if anything is
  ambiguous. **Skip when**: task scope is unambiguous after the light scan and there's
  no clarifying value-add (applies to S/M/L equally — the old "ask at least one" was
  ceremony).

Evidence: feature= | category= | docs= | preset= | recall= | Q&A=

---

## Step 2: Light Codebase Scan

> **SIZE GATE:** Skip for **S** features (S gets deep scan in Phase 2).

**ISOLATION RULE:** Run as `Agent` tool (`subagent_type: "Explore"`, thoroughness "medium").

⚡ **PARALLEL:** Can run in background while presenting Step 1's AskUserQuestion.

Launch Explore agent with mode-appropriate mission:

**BUILD**: Find existing patterns, reusable components, similar precedents (5-10 files).
**FIX**: Find affected code paths, related tests, recent changes.

- [ ] Explore agent launched (NOT in main context)
- [ ] Report saved to `specs/2026-06-23-onnx-selfplay-loop/codebase-scan-light.md`
- [ ] Key findings summarized

Evidence: scan_mode= | findings_count= | top_3=

---

## Step 3: Clarifying Questions (Round 1)

> **SIZE GATE:** S: 0-1 iter, skip if clear. M: 1-2 iter **if scope still has
> unknowns** (skip when confident). L: 2+ iter, **MANDATORY STOP** — skip only when
> scope is exhaustively defined by the user prompt + Step 2 scan.

Focus: clarify scope, confirm understanding, identify biggest unknown.

### Loop: clarify until scope is sharp

Repeat: **Think** (what's the biggest remaining unknown about scope?) → **Act**
(ask ONE `AskUserQuestion` with 2-4 concrete options) → **Observe** (record
answer, re-assess remaining unknowns). Exit when scope is clear enough that the
codebase scan and Phase 2 design can proceed without major redirection. Zero
iterations is fine for well-defined tasks (S only).

- [ ] Iterations: ___ (or "0 — task fully defined")
- [ ] Key answers recorded

Evidence: iterations= | answers= | remaining_unknowns=

---

## Phase 1 Complete — Handoff

### Write discovery output

Write `specs/2026-06-23-onnx-selfplay-loop/discovery-output.md`: Mode, Size, Feature (1-line), Context category,
Domain preset + roles, Light scan summary, Invariant docs loaded, Open questions, Round 1 Q&A.

### Update state

```bash
cat > specs/2026-06-23-onnx-selfplay-loop/FEATURE_STATE.json << 'STATEEOF'
{ "slug": "onnx-selfplay-loop", "started": "2026-06-23", "mode": "$MODE", "size": "$SIZE",
  "current_phase": 2, "current_step": 4, "last_completed_step": 3,
  "spec_dir": "specs/2026-06-23-onnx-selfplay-loop", "repo_root": "/Users/j/Unciv-onnx-selfplay-loop" }
STATEEOF
```

- [ ] `discovery-output.md` written
- [ ] `FEATURE_STATE.json` updated to phase 2
- [ ] Cost log: _(total spend this phase, or "$0")_

**Phase 1 complete.** The orchestrator will now load `phase2-design.md`.
