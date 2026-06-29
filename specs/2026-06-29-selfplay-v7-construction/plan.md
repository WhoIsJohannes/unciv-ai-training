# Plan — selfplay-v7-construction (v7: per-city PRODUCTION control head)

**Mode**: BUILD · **Size**: L · **Base**: master (fork) · **Branch**: selfplay-v7-construction

## WHAT & WHY (how it solves the ask)
The learned policy controls only **tech + policy** (two civ-global, one-per-turn decisions),
so in the symmetric 1v1 Random-opponent eval it sits below 50% (v5 small-rung 40.7%). v7 gives
the net its first **PER-ENTITY** lever — **per-city construction** (what each city builds) — a
variable-count decision conditioned on *which* city. We answer: does controlling production move
the learner toward/past 50% on Medium? **Ship criterion (D-C5):** ship the infra + honest RESULTS
as long as we PROVE directional improvement (**construction-ON beats construction-OFF, p<0.05**, in
≥1 rung), even below 50%; crossing 50% is a reported milestone, not a gate.

## HOW — design (decisive, from the prompt; refined by scan + council)

### Ordering contract (council C2 + scan surprise #1) — load-bearing
`x.cities.sortedBy { it.id }`, **capped at `caps.maxOwnCities`**, is the canonical own_cities order.
[AI_CODE] Extract it into ONE shared function `Featurizer.orderedOwnCities(civ): List<City>` and
reuse it in: the featurizer mask/tokens, the chooseAndApply decision loop, and the recorder. Row `i`
of `mask_construction` / `construction_action` / `construction_logp` / the ONNX per-city output ALL
correspond to `orderedOwnCities(civ)[i]`. A determinism+parity test asserts the alignment.

### (A) CONTROL — pre-fill at the civ-turn hook  [AI_CODE]
In `DataPlaneHooks.chooseAndApply`, AFTER tech/policy, when `config.controlConstruction`:
- Iterate `orderedOwnCities(civ)`. For each city, decide a construction **iff the city would choose
  this turn** = its current construction is a `PerpetualConstruction` (idle) OR queue empty AND it has
  ≥1 legal construction. Otherwise record −1 (no decision).
- For a deciding city: read its `mask_construction` row from `obs`, call
  `policy.chooseConstructionWithLogp(civ, city, cityRow=i, legalMask, turn)` → `(idx, logp)`; if
  `idx≥0` and `vocab.constructionId(idx)` is non-null and `city.cityConstructions.canConstruct(name)`,
  pre-fill via `city.cityConstructions.setCurrentConstruction(name)` (sets queue[0]). Record `idx`,`logp`.
- Edge cases (council C7/M30): 0 cities → empty blocks; empty legal mask → −1; capture/destruction
  between obs and apply is impossible (same synchronous `handleCivTurn`); invalid reverse-map → guarded
  by `canConstruct` → fall back to −1 (heuristic builds).
- `ConstructionAutomation.chooseNextConstruction` early-returns when
  `DataPlaneHooks.controls(city.civ) && <policy pre-filled this city this turn>` (mirror adoptPolicy:302).
  Cities the policy did NOT decide → heuristic untouched. Decision cadence == the heuristic's (decide
  only when idle) — intentional (council M29); a mid-build city keeps its item.

### (B) SEAM — `PolicyProvider.chooseConstructionWithLogp`  [AI_CODE]
Add `fun chooseConstructionWithLogp(civ, city, cityRow: Int, legalMask: BooleanArray, turn: Int):
Pair<Int,Float>` with a **default uniform-legal impl** (mirrors `chooseIndexWithLogp`; RandomPolicy
inherits it). Kept minimal — one method, no new class (council M11): the per-entity signature genuinely
needs the city/cityRow that the civ-level seam lacks, so the new method is justified, not abstraction tax.
`OnnxPolicy` implements it: ONE memoized forward per `(game|civ|turn)` ALSO produces per-city
construction logits `[ncities][constrW]`; index row `cityRow`; sample via `MaskedChoice.chooseWithLogp`
(single RNG draw — byte-identical replay). chooseIndex/chooseIndexWithLogp (tech/policy) unchanged.
**Graceful fallback** (council C4): if `OUTPUT_CONSTRUCTION` is absent / NaN / wrong city-dim / no
legal action after masking → return `(-1, 0f)` → heuristic builds (this is also the OFF path).

### (C) NET — per-city construction head  [AI_CODE]
`RichPolicyValueNet`: capture the own_cities **pre-pool** token embeddings `[B,Ncities,token_dim]`
(currently discarded by `masked_pool` in `_TokenEncoder`). Add
`construction_head = MLP(per_city_emb ⊕ broadcast trunk h) → [B,Ncities,constrW]`, `constrW` read from
the schema (buildings+units, never hardcoded). `forward` returns `(tech, policy, construction, value)`.
Tech/policy/value heads + dims unchanged (no encoder rewrite). `export_onnx._RichPolicyOnly` returns
`(tech, policy, construction)` (value still dropped); `output_names += OUTPUT_CONSTRUCTION`;
`dynamic_axes["construction_logits"]={0:"batch",1:"n_cities"}`.

### (D) RECORD — two VARIABLE blocks  [AI_CODE]
`recordStep` appends `Block("construction_action", DT_F32, VARIABLE, perItem=1, cAct)` and
`Block("construction_logp", DT_F32, VARIABLE, perItem=1, cLogp)`, one row per `orderedOwnCities`
(−1 / 0 where no decision). Observation does NOT re-emit per city (one obs/step). Bump
`SampleSchema.VERSION 4→5` (Kotlin :27) + `schema.py SCHEMA_VERSION 4→5` (:20) in lockstep — old v4
shards refuse (`expect_compatible`), perishable by design (council C3 → v7 replay starts FRESH, no v6
carryover). Reader is descriptor-generic (`reader.py:_decode_blocks`) → both new f32 VARIABLE blocks
round-trip with NO reader change. Construction is a per-entity block, NOT a `MASK_HEADS` slot (council
M13); `actions`/`behavior_logp` stay width-4 FIXED, unchanged.

### (E) TRAIN — construction logp summand  [AI_CODE]
`dataset.py`: load `construction_action`/`construction_logp` (ragged [T,Ncities_i]) + `mask_construction`;
**pad to max Ncities** (action=−1, logp=0, mask=0 — inert rows contribute 0) BEFORE tensorizing so the
micro-batch `forward_chunk_fn(lo,hi)` step-index slicing stays aligned (scan surprise #7).
`train.py`: live per-step logp `= _masked_logp(tech)+_masked_logp(policy)+Σ_cities _masked_logp(construction_city)`
(−1 → 0, exactly the existing gating). The **stored `behavior_logp`/old_logp** = the SAME sum captured at
generation → importance ratio `exp(logp−old_logp)` stays a per-step scalar covering all heads jointly
(council C1/C10; the recorded city set+order is frozen in the sample so old/new logp align city-for-city
across the K=4 replay window). The per-step GAE advantage (terminal ±1, current-value critic) multiplies
the WHOLE summed logp — construction inherits the step's advantage (shared, research-confirmed standard;
no per-decision reward, no shaping). Entropy MAY add the per-city construction term. Clip/value/GAE math
otherwise verbatim. Risk (council C9 + research insight #2): the Σ_cities log-ratio grows with city count;
the existing `logratio.clamp(-20,20)` + PPO clip bound it — watch ratio/clip-fraction telemetry on ON arms.

### Plan-council refinements (folded — PR1–PR6)
- **PR1**: AC#5 gate = the DETERMINISTIC zero-summand bit-identical-weights test (max|Δw|<1e-6); the OFF-vs-v6 training-curve match is confirmatory only, not a gate.
- **PR2**: COUNT construction-fallback events (idx<0 / illegal / NaN / missing output); the ON arms must report ≈0 fallbacks (asserted in analyze_v7), else the ON-vs-OFF comparison is contaminated → flagged invalid. A guard-failure records −1 (policy abstains; heuristic builds; train contributes 0 — no gradient mixing).
- **PR3**: fail-loud assert at ONNX load / first inference that `construction_logits` width == `vocab.buildingCount+vocab.unitCount` (mirror `test_contract_failloud`) — no silent vocab mismatch.
- **PR4**: explicit null-guard — `val name = vocab.constructionId(idx) ?: <record −1, skip>` BEFORE any `canConstruct` call.
- **PR5**: bench-onnx is a PRE-run GATE — measure ON-arm throughput BEFORE launching the multi-hour batch; a <70% head is fixed before launch, not discovered mid-run.
- **PR6**: v7 runs in a FRESH OUT_ROOT, replay starts EMPTY, NO cross-schema ckpt resume from v6 dirs; rollback = revert branch (v5 shards/replay perishable & separate).

### (F) NO-OP SAFETY — `--control-construction {on,off}`, default ON  [AI_CODE]
Positional Python→JVM bridge (scan surprise #3): `run_loop.py --control-construction` →
`gradle_selfplay(["gen",…, str(control).lower()])` → `SelfPlayRunner.gen/eval` parses positional
`args[8]` → `SampleConfig.controlConstruction`. OFF ⇒ no construction control (heuristic builds), the
decision loop records **zero** construction decisions → `Σ_cities _masked_logp(construction) ≡ 0` →
loss/gradient **identical to v6** (the head gets zero gradient, inert). **No-op proof (council C4/M14/M28)**:
OFF construction summand == 0 (asserted) AND the OFF curve reproduces v6 within fp tolerance (and v5 at K=1).

### Vocab inverse (scan surprise #2 / trap)  [AI_CODE]
Add `Vocab.constructionId(idx: Int): String?` inverting the **0-indexed mask space** (matches
`constructionMask`/net logits): `idx<buildingCount → buildingName(idx)`; `else → unitName(idx−buildingCount)`;
out-of-range → null. NOT the 1-indexed `constructionCode` inverse (off-by-one trap). Round-trip asserted
in the parity test.

## Experiment  [AI_CODE]
`run_v7.sh` (mirrors run_v6.sh, resumable `--resume`, heartbeat logs, per-round ckpt): **4 arms**
sequential, all `--replay-window 4 --continual --micro-batch-steps 256`, structured/Medium, 16 rounds,
gen 16 / eval 80, turn-cap 250:
`small-OFF`, `small-ON`, `medium-OFF`, `medium-ON` (construction is the ONLY axis within each rung pair).
Then per-arm 200-game ceiling eval @ eval-seed 4242424 + z-tests vs the fixed blind baseline.
`analyze_v7.py` (mirrors analyze_v6): per-rung ON-vs-OFF two-proportion **one-sided z-test** (H1: ON>OFF),
draws/timeouts/crashes = non-win, 200 fixed denominator; AND each arm's win-rate vs the **50% break-even**
(z/p), stated plainly. Run order: parity + no-op + legality green → THEN the Medium run (per AC).

## FROZEN / NON-GOALS (council M19 — explicit)
FROZEN: terminal-only ±1 reward (NO shaping); tech/policy heads + dims; PPO clip/value/entropy/GAE math
(only the logp now sums the construction term); v5 continual + v6 replay machinery. NON-GOALS: NO
promotion / great-person / diplomatic-vote heads (next features, reuse this per-entity infra); NO
unit-movement / target head (spatial, deliberately heuristic); NO encoder rewrite (head reads existing
own_cities embeddings); NO autoregressive city-conditioning (independent per-city is the correct first
cut). Asserted-heuristic: unit movement, promotion, great-person, diplomatic-vote.
OUT OF SCOPE (council C24/C25/m31 — single-trust local training, no adversary): ONNX file signing,
replay data-poisoning defenses, reader hardening beyond the existing buffer-safe `<H` decode.

## Reuse (all internal; no new libraries)
`MaskedChoice.chooseWithLogp` (sampler), Observation VARIABLE-block format, `reader.py` descriptor-generic
decode, `_masked_logp` + v6 micro-batch/replay/old_logp machinery, `TestGame` fixtures, `run_v6.sh` /
`analyze_v6.py` as templates. No duplication introduced.

## Tests (definitions — council C6; RED-first in Step 11)
- **LEGALITY (AC#1)**: every applied construction ∈ that city's mask; recorded `construction_action[i]` ==
  the construction actually queued for `orderedOwnCities(civ)[i]`; zero illegal across the eval.
- **PARITY (AC#2)**: JVM per-city construction logits == Python reference for a fixed obs, atol 1e-4
  (extend the existing logit-parity harness to the per-city output) + `constructionId` round-trip.
- **SCHEMA (AC#4)**: VERSION 4→5 lockstep; old v4 shard refuses; the two new VARIABLE blocks round-trip
  with no reader change (Python round-trip test).
- **NO-OP (AC#5)**: OFF → construction summand ≡ 0; OFF reproduces v6 within fp tolerance (and v5 at K=1).
- **DETERMINISM**: single RNG draw/decision; identical order; byte-identical shards on replay incl. blocks.
- **THROUGHPUT (AC#6)**: bench-onnx — per-city head inference cost; ≥70% of heuristic baseline.

## Component shape (clean-structure anticipation — advisory)
~2 new files (`run_v7.sh`, `analyze_v7.py`) + surgical edits to ~18 existing files; the construction head is
a ~30-line block in `model.py`; the shared `orderedOwnCities` helper consolidates a currently-duplicated
literal. No module is expected to mix responsibilities or pass ~300 lines from this change.

## Walkthrough
See `## Walkthrough` below — one civ-turn traced end to end.

## Walkthrough (one representative civ-turn, illustrative mock values — not measured)
Civ "Rome" (controlled, construction ON), turn 80, owns 3 cities; vocab: buildingCount=40, unitCount=18 → constrW=58.

| Hop | Component | Contract | Mock data |
|---|---|---|---|
| 1 | `handleCivTurn(Rome)` → `featurizer.observe` | Observation w/ VARIABLE `mask_construction` | `orderedOwnCities=[c12,c31,c54]` (sorted by id, ≤cap); mask rows: c12 legal {Granary(idx5), Worker(idx40)}, c31 mid-building Wonder (will be skipped), c54 idle on Perpetual-Gold legal {Library(idx8),Warrior(idx41)} |
| 2 | `chooseAndApply` tech/policy | actions[0..1] | tech→idx14 logp −2.3; policy→idx3 logp −1.6 |
| 3 | construction loop over `orderedOwnCities` | per-city `(idx,logp)` | c12 deciding → `chooseConstructionWithLogp(row0)`; c31 NOT idle → −1; c54 idle → `chooseConstructionWithLogp(row2)` |
| 4 | `OnnxPolicy` (ONE memoized forward) | `construction_logits[3][58]` | row0 masked-softmax (single RNG draw) → idx5 (Granary) logp −1.1; row2 → idx8 (Library) logp −0.9 |
| 5 | pre-fill + heuristic guard | `setCurrentConstruction` | c12.queue[0]="Granary"; c54.queue[0]="Library"; `ConstructionAutomation` skips c12,c54 (controls && pre-filled); c31 untouched |
| 6 | `recordStep` | two VARIABLE blocks, 3 rows | `construction_action=[5,−1,8]`, `construction_logp=[−1.1,0,−0.9]`; stored old_logp = −2.3−1.6−1.1−0.9 = −5.9 (sum heads+cities) |
| 7 | (train, later) `dataset.py`+`train.py` | per-step logp + GAE adv | pad to Nmax; live logp recomputes over recorded [5,−1,8]; ratio=exp(logp−(−5.9)); ×step advantage (terminal ±1 → GAE) |

If `OUTPUT_CONSTRUCTION` were missing (OFF arm / old model): hop 3 returns −1 for all cities → heuristic builds → `construction_action=[−1,−1,−1]` → train summand 0 → identical to v6.
