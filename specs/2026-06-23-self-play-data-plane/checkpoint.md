# Checkpoint — Phase 3 complete

**What was built:** the full self-play data plane (see `build-output.md`). 13 dataplane Kotlin files
(1259 LOC) + 4 gated engine edits + desktop entrypoint + 5 Python reader files (448 LOC) + 12
acceptance tests.

**Key files:**
- Featurizer/fairness: `core/.../dataplane/{Featurizer,FairOpponentModel,Observation,LegalActionMasks}.kt`
- Format/provenance: `{SampleSchema,SampleConfig,ShardFormat,Vocab,RulesetFingerprint,TrajectoryEmitter}.kt`
- Policy/generator/glue: `{PolicyProvider,ScenarioGenerator,DataPlaneHooks}.kt`
- Engine seams: `NextTurnAutomation.kt`, `GameStarter.kt`, `GameParameters.kt`, `Simulation.kt`
- Entrypoint: `desktop/.../DataPlaneGen.kt` + `:desktop:dataGen`
- Reader: `python/unciv_dataplane/`

**Gate status:** core/desktop/tests compile; **full suite 730 tests 0 fail** (no regressions);
12/12 dataplane acceptance tests green; 6 python reader tests green; end-to-end gen→shard→python
read CRC-clean.

**Test spec:** `agent-test-spec.md` · **Council plan review:** `council-plan-review.md` (APPROVE).

**Build/test commands (JDK 21 needed):**
```
JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home ./gradlew :tests:test --tests 'com.unciv.logic.simulation.dataplane.*'
JAVA_HOME=... ./gradlew :desktop:dataGen --args="<outDir> <maxTurns> <episodes> <seedBase> [maxMapRadius]"
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest tests/test_reader.py
```

**Open issues:** see `build-output.md` "Open issues" + the cleanup backlog. None blocking.

**Traceability:** every plan module → implemented + covered (acceptance table in `build-output.md`
maps AC1–AC10 to tests). Unit-intent action labels + religion-conversion numerator are documented v1
deviations (cleanup backlog), not gaps in the fairness/determinism/provenance guarantees.

**Next:** Phase 4 (ship) — create branch PR. Outward-facing; awaiting user go-ahead (not auto-shipped).
