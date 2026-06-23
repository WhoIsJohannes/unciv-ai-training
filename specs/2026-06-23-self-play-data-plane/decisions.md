# Decisions — Self-Play Data Plane for Unciv

## Phase 1 (locked)
- **D1. Determinism scope = GATE TO SIM PATH.** Add `deterministicShuffle: Boolean = false` (default off) threaded through GameSetupInfo/GameParameters; only the sim/data-plane path enables it. `GameStarter.kt:322` shuffle uses `gameContext.stateBasedRandom("GameStarter")` only when set. Interactive play byte-identical. Policy RNG reuses `stateBasedRandom(label)` keyed on `(gameId, civ, turn)`. Generator enforces `seed != 0` (Simulation currently sets `seed=0`). [user clarification, Phase 1 Step 3]
- **D2. Scenario envelope = FULL STANDARD RANGE.** Generator randomizes across all map sizes (incl. Large/Huge) and up to Unciv's max players, with guardrails (land/civ ≥ 80, MapRegions retry). Accepted tradeoff: larger tensors, slower gen. [user clarification, Phase 1 Step 3]
- **D3. Spatial sizing (leading, to confirm at plan).** Per-entity tokens fixed-width to the max envelope; spatial plane sized to each game's actual `tileList` length (fixed within a game), self-described in provenance — avoids catastrophic padding when small maps coexist with Huge in one dataset. [Claude interpretation]

## Phase 2 — Step 5 intake council roster (8 roles, L cap ≤14)
Rationale: this is a data-plane / info-flow feature with NO UI and NO real PII, so user/business/compliance/accessibility/i18n lenses don't apply. Selected:
- **Core 6** (always-on): skeptic, architect, practitioner, product_manager, qa_testing, security_red_team.
  - `security_red_team` is load-bearing here = the fair-info **leakage** lens (hidden-state exfil into features).
  - `qa_testing` = the 10 acceptance criteria must be testable (leakage/determinism/tile-gate).
- **domain_fidelity** (conditional): Civ V information-semantics is a specialist domain where the wrong abstraction silently leaks/mis-gates — does the fair-info model match what a human actually sees on each screen?
- **cost_efficiency** (conditional): full-standard-range → large spatial tensors + shard volume + generation throughput.
Excluded (mismatch, would be padding a skeptic-watched council): data_privacy_legal (no personal data — synthetic game state), ethics_responsible_ai (no model in the loop — data plane only), end_user/power_user/onboarding/support/b2b/marketing/finance/investor/accessibility/i18n/compliance (no UI, no users, no billing, no regulated domain). `plan_conformance` is a dedicated Phase-4 pass, never a roster role.
Models: vendor-diverse (openai/anthropic/x-ai/deepseek/google); deep models (opus-4, gemini-2.5-pro, deepseek-r1) reserved for architect / security_red_team / domain_fidelity / product_manager where depth is load-bearing.

## Phase 2 — Step 5 intake council triage (38 findings, round 1)
Legend: 🔴 critical · 🟡 worth-considering · ⚪ low/noise. "→" = resolution.

### 🔴 Critical
- **Determinism vs byte-identical interactive play** (architect) → RESOLVED by D1 gate; plan states precisely: interactive play keeps the UNSEEDED `otherPlayers.shuffle()` (unchanged); only `deterministicShuffle=true` (sim path) routes through `stateBasedRandom("GameStarter")`. INCORPORATE wording.
- **No observability for emitter/shards** (practitioner) → SCOPE-DOWN: offline batch data-gen, not a 24/7 service. Adopt lightweight = structured per-shard log lines (shardId, turn, bytes, checksum, error) + a run-summary line + per-game try/catch isolation (skip+log+continue). REJECT Prometheus/`/healthz`/separate-JVM gold-plating (skeptic concurs). INCORPORATE (light).
- **Missing success metrics for P0 capabilities** (PM) → INCORPORATE measurable acceptance: 0% hidden-state leakage (leakage test), 100% mask-parity on sampled turns, byte-identical replay checksum, fingerprint-drift detection. The 10 acceptance tests ARE the metrics.
- **Undefined scope for "full standard range"** (PM) → RESOLVED by code: predefined sizes = Tiny–Huge (r40 ≈5k tiles); Civ5Huge(r128) is a commented-out reference, NOT selectable. Generator uses predefined Tiny–Huge; per-entity caps configurable in SampleConfig; spatial plane per-game-sized. No raw-float demographics emitted (down-gate).
- **Fair-info model lacks enumerated allow/deny list** (QA) → INCORPORATE: plan includes an explicit per-field encoding inventory (EXACT / TRADE / RANK-BUCKET / TILE-SEEN / ESPIONAGE / BROADCAST / HIDDEN), which the leakage test diffs against.
- **Determinism has no measurable definition** (QA) → INCORPORATE: `calculateChecksum` = stable hash over shard bytes; contract = identical (gameId, seed≠0, deterministicShuffle=true) ⇒ byte-identical shards + header.
- **Espionage `getTechsToSteal` leaks the tech NAME SET** (security) AND **"spies DO reveal the list — flag-only is a domain error"** (domain_fidelity) → GENUINE FORK, see USER QUESTION below. Prompt says flag-only (`.isNotEmpty()`), names HIDDEN.

### 🟡 Worth considering (INCORPORATE)
- **PolicyProvider injection method** (architect) → optional `policyProvider: PolicyProvider = RandomPolicy()` param on `automateCivMoves`, threaded from `Simulation`/the gen runner.
- **Serialization impedance** (architect) → emitter serializes a SUBSET (the fair-info feature fields) via explicit field→tensor mapping; NOT full `GameInfo` libGDX-Json.
- **Fixed-width overflow** (architect) → clamp to cap + set an `overflow` flag in the step record + log to an overflow tally; reader validates caps vs schema. Never silent corruption.
- **@Synchronized unsafe across coroutine suspension** (architect) → emitter is ONE file per shard (per CoroutineName), no shared writer → structural thread-safety.
- **`shouldHideCivCount` bypassable** (security) → featurizer does NOT use UI logic; unmet civs contribute zeros+mask everywhere incl. ranks; ranks computed among MET majors (+ self); global best/avg/worst is identity-free.
- **RNG-label covert channel** (security) → SCOPED: policy RNG keyed only on visible `(gameId, civ, turn)`; observation is read fog-correctly by the featurizer independent of engine RNG labels, so engine-label audit is out of scope (doesn't flow into features). Noted.
- **`religions` ArrayList order** (security) → sort canonically (stable) before building vocab → fingerprint stability.
- **Partial-visibility fog encoding** (domain_fidelity) → spatial plane carries a per-tile visibility-state channel (never-explored / explored-not-visible / currently-visible); PERSISTENT channels (terrain/feature/resource/road) for explored; TRANSIENT channels (units/city/improvement-state) gated on currently-visible. (Refines the prompt.)
- **City-state indirect info channel** (domain_fidelity) → a met CS reveals its major ally even for unmet majors → encode with the engine's third-party identity masking (`getCivName`).
- **Trade/score visibility timing** (domain_fidelity) → already in prompt (trade-mask = met∧major∧¬CS; score EXACT for met). Confirm.
- **Single shared Simulation blast radius / SLO** (practitioner, QA) → per-game try/catch isolation + soft target (note expected turns/sec from SimBenchmark), NOT a hard SLO. INCORPORATE (light).

### ⚪ Low / noise (REJECT, with reason)
- cost_efficiency ×3 ("LLM token costs", "caching LLM calls", "per-call LLM cost") → HALLUCINATION: there are NO LLM calls in this data plane. REJECT. (Its valid point — bound map-size + tensor cap — kept above.)
- PM "conduct 10 human gameplay sessions to validate fairness" → impractical; the leakage/tile-gate/down-gate/unmet unit tests are the validation. REJECT.
- skeptic scope-cuts (drop fair model / provenance / determinism flag / full range) → contradict explicit user decisions + the prompt's core goals; acknowledged as YAGNI tension, each subsystem kept minimal, but NOT cut.

### Spec-vs-reality gaps (Claude decides at plan)
- **`originalMajorCapitalsOwned` does not exist** → derive domination numerator from original-capital city ownership if a clean signal exists, else emit with denom/value-mask=0. Default: derive-if-clean-else-mask.
- **Defensive strength** no direct City field → compute via `CityCombatant(city)` (verify in build).
- **Spy "set up in city"** state → verify `Spy` state (`isSetUpForEspionage`/`action`) in build for the spy-gate.

### D4. Fair-info depth (espionage / spaceship) = SPLIT (domain-faithful spy) [user, Step 5 join]
- Spy gate (X has a SET-UP spy in O's city): encode the **stealable-tech multi-hot** over the tech vocab (`getTechsToSteal(O)`), alongside the spy-gated city interior. Rationale: genuinely human-visible, earned via spy investment (domain_fidelity).
- Spaceship victory progress: encode **count only** (`currentsSpaceshipParts.sumOf{}`); part identities are NEVER encoded in the fair model.
- The passive/met path still NEVER encodes O's tech LIST; the steal-list (a derived subset) is encoded ONLY under the spy gate. omniscientOpponents=true exposes raw for ablation.
- Leakage tests must verify: no spy ⇒ no stealable-tech bits; spaceship part identities never appear in fair mode.

## Phase 2 — Step 11 plan-review council (40 findings, round 1) → APPROVE w/ refinements
Verdict APPROVE after folding R1–R15 into plan.md (see `council-plan-review.md`). No architectural change; all criticals mitigated as under-specified-detail tightenings. Key binding refinements: canonical entity ordering (R1); meta (overflow flag/fingerprint/gameId/seed) NOT in observation tensor (R2); checksum=records-only (R6); deterministic overflow drop order (R8); demographics rank/bucket+tie-break algorithm (R9); shard workerId collision guard (R10); atomic .tmp + truncation-tolerant reader (R11); scenario maxAttempts=100 (R13); complete fair-info encoding-status table as test oracle (R14); Tiny single-thread determinism first (R15). Rejected: drop-flag/drop-interface/hardcode-config/async-queue/VP-ceremony/3rd-party-politics-strip (each with reason in plan.md).

## D5. Shard format v2 — variable-count + u8 (size fix) [user concern: Tiny game was 9MB]
v1 stored every entity list DENSELY PADDED to caps (40 civ slots, 64 cities, 192 units, 64×130
construction mask, 192×40 promotion mask, 64×80 spy-tech) as f32, EVERY turn — ~95% zero-padding
for a small game → 9 MB for a Tiny 10-turn game (defeats the purpose vs a JSON snapshot).
**Fix (AlphaStar/RLDS-standard):** entity lists are VARIABLE-count — store ONLY present entities
(fixed width per token); padding-to-cap moves to the data loader. Masks + the spatial plane are u8
(not f32). Caps remain as overflow bounds. Record format: per-step, each VARIABLE block is u16-count
prefixed. Result: **9,032,512 → 240,897 bytes = 37.5× smaller**; 6 KB/step now dominated by the
spatial map plane (4303 u8 = the actual 331-tile map — legitimate, scales with map size). All 12
fairness/determinism/provenance tests still pass; Python reader reads it CRC-clean. Reader decodes
FIXED (len) vs VARIABLE (u16 count × perItem) per the header layout's `kind`/`perItem`/`dtype`.
