# Deep Codebase Scan (Phase 2, Step 7)

Focus: NEW build-derisking angles (module graph, tests, entrypoints, widths, collisions). Seam map is in `codebase-scan-light.md`.

## 1. Gradle module graph
- `settings.gradle.kts`: modules = `desktop`, `core`, `tests`, `server` (+ `android` if SDK present).
- `desktop/build.gradle.kts:103` → `implementation(project(":core"))`. `android/build.gradle.kts:160` → `implementation(project(":core"))`. `server` does NOT depend on core.
- Kotlin 2.1.21 (`gradle/libs.versions.toml`), JVM target 1.8 (core+desktop), Java sourceCompat 21, Android compileSdk 36.
- **`gradle/libs.versions.toml` is the version catalog** — add `onnxruntime` there, reference via `libs.onnxruntime`.
- Scan view: `compileOnly` in core is APK-safe (Android packages only `implementation`/`api`). **But** (council FND-0002/0006) core referencing ORT types at all is a dependency-inversion smell. → **RESOLUTION: OnnxPolicy in `desktop` (ORT dep there only); RoutingPolicy + SimStats in `core` (no ORT).** Sidesteps both; no compileOnly hack.

## 2. Test infrastructure (templates to mirror)
- JUnit4 + `tests/src/com/unciv/testing/GdxTestRunner.kt` (`@RunWith(GdxTestRunner::class)`, HeadlessApplication render loop). Mockito 5.20.
- `tests/src/com/unciv/testing/TestGame.kt` — fresh cloned GnK ruleset per test; `makeHexagonalMap(radius)`, `addCiv()`, `addCity()`, `addUnit()`.
- `tests/src/com/unciv/logic/simulation/dataplane/FairnessAndDeterminismTests.kt`:
  - **Legality template**: `maskParity_techMatchesEngine()` (:183), `maskParity_policyMatchesEngine()` (:195) — `ruleset.technologies.keys.forEachIndexed { idx, name -> assertEquals(x.tech.canBeResearched(name), mask[idx]) }`. Mirror for the OnnxPolicy legality test (assert chosen index is always legal).
  - **Determinism template**: `assertArrayEquals(build(), build())` (:144) over featurizer/recorder bytes.
  - Construction: `Featurizer(g.gameInfo, Vocab(g.ruleset), SampleConfig(...))`.
- Python: pytest, `python/tests/test_reader.py`, synthetic `_build_shard(version=SCHEMA_VERSION,...)`. **No `pyproject.toml`/`requirements.txt`** — python is dev-only today (numpy-only reader). `unciv_train` needs its own `pyproject.toml` (torch, onnxruntime, numpy, matplotlib, deps on `unciv_dataplane`).

## 3. Desktop entrypoint wiring
- `JavaExec` task pattern confirmed (`dataGen`, `simBench`). Add `selfPlayGen` + `selfPlayEval` the same way (mainClass `com.unciv.app.desktop.SelfPlayRunner`, args list).
- Headless bootstrap (ConsoleLauncher/DataPlaneGen): `Log.backend=DesktopLogBackend()`, `UncivGame(true)`, `UncivGame.Current=game`, `GameSettings{showTutorials=false; turnsBetweenAutosaves=10000}`, `RulesetCache.loadRulesets(consoleMode=true)`, (`TileSetCache`/`SkinCache` load).
- **Pin nations**: `val nation=Nation().apply{name=...}; ruleset.nations[name]=nation; Player(nation)` (Player ctor takes a Nation → sets `chosenCiv`). MapSize.Tiny, GnK, difficulty King/Prince (simulation), noBarbarians/noRuins/noNaturalWonders.

## 4. GnK vocab widths (resolve empirically)
- techCount ≈ 80. **policyCount ≈ 80** (NOT 70 — `ruleset.policies` flattens branch openers like "Tradition" in alongside leaf policies). Task said 70; data-plane plan said ~70. **Authoritative value = `vocab.policyCount` at runtime, read from generated `schema.json` `mask_policy.len`.** The contract NEVER hardcodes — confirmed-empirically at build (generate one shard → read schema.json). This discrepancy is exactly why the contract is runtime-derived.

## 5. Collisions / blast radius
- No name collisions for OnnxPolicy/RoutingPolicy/SimStats/SelfPlayRunner.
- `Simulation.binomialTest()` private, called only internally → safe to extract to public `SimStats`.
- `NextTurnAutomation.onCivTurn` (single nullable var) already used by the data plane — only one policy at a time per run; RoutingPolicy reuses the existing `install(policy)` seam unchanged.
- `.gitignore`: `dataplane-shards*/` + `*.bin` already ignored. ADD: `*.onnx`, `selfplay-output/`, `curve.csv`, `python/**/__pycache__/`, `.pytest_cache/`, `*.egg-info/`, `training-runs/`.

## Top surprises
1. **policyCount is likely ~80, not 70** — branch openers are flattened into `ruleset.policies`. Vindicates the runtime-derived-width design; the contract reads the real number from `schema.json`.
2. **OnnxPolicy belongs in `desktop`, not `core`** — the cleanest reconciliation of the task's stated path vs the council's dependency-inversion concern; the data plane already places heavy concerns (DataPlaneGen) in desktop.
3. **`NextTurnAutomation.onCivTurn` is already the instrumented seam** — RoutingPolicy plugs into the existing single-policy `install()` with zero data-plane change.
4. **Python is dev-only (no packaging yet)** — `unciv_train` adds the first real `pyproject.toml`; clean slot, no existing packaging to fight.
5. **`TestGame` cloned-ruleset pattern** is why determinism tests are clean (no global state pollution) — reuse it for the legality + parity Kotlin tests.
6. **Widths are NOT in gradle/build** — fully runtime-discovered → robust to ruleset changes; no rebuild needed when content changes, only shard regen.
