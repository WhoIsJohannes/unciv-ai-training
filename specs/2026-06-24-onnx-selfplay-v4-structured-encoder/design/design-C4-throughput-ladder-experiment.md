# Design — C4-throughput-ladder-experiment

## Summary
D8 wraps OnnxPolicy.forwardRich in Timers.timeThis("onnxForward") and EXTENDS the existing SimBenchmark.kt with a new `onnx <model>` mode: it injects OnnxPolicy as the learner via the exact same RoutingPolicy+DataPlaneContext+Simulation path SelfPlayRunner.eval already uses, measures data-gen turns/s and ms/decision from wall-clock divided by OnnxPolicy.decisionCount() (authoritative, since Timers' aggregates are Log-gated), and REJECTs the rung if turns/s < 70% of the heuristic-baseline turns/s the bench already computes — reporting ms/decision + turns/s for every rung. D7 adds a demand-driven ladder in run_loop.py that starts at the smallest rung and scales up one rung only when eval win-rate is still rising over last K rounds AND the train/eval gap is small, and STOPs on no-improvement, D8 budget violation, or OOM (the whole-round dense batch over ~1261 Medium tiles at train.py:265). The structured encoder is dispatched via `--variant structured` (alias `rich-v2`) reusing the existing rich trainer/exporter plumbing with the new nn.Module. D10 holds budget constant vs the v3 rich-critic baseline, produces eval curves on Tiny AND Medium, and pre-registers the PRIMARY acceptance: structured BEATS v3 rich-pool on Medium at p<0.05 via the existing analyze.py `_two_proportion_z` over the final 200-game eval, Tiny must not regress; the from-scratch-per-round training-regime confound is stated explicitly and a null/negative result is reported plainly (v2 ethos).

## Detailed design
# C4 Design — Throughput Guard (D8), Ladder (D7), Experiment (D10)

All three deliverables are grounded in the actual worktree source at
`/Users/j/Unciv-onnx-selfplay-loop`. Key seams already exist and are reused verbatim:
the eval injection path (`SelfPlayRunner.kt:178-215`), the rich export/train dispatch
(`run_loop.py:74-94`, `:211-222`), the two-proportion z-test (`analyze.py:50-59`), and the
heuristic-baseline benchmark (`SimBenchmark.kt`).

---

## D8 — Throughput guard

### D8.1 Wrap `forwardRich` in Timers
`OnnxPolicy.forwardRich` is `desktop/src/com/unciv/app/desktop/OnnxPolicy.kt:127-138`. Wrap the
session-run body:

```kotlin
// OnnxPolicy.kt:127
private fun forwardRich(obs: Observation): Pair<FloatArray, FloatArray> =
    com.unciv.logic.automation.Timers.timeThis("onnxForward") {
        val inputs = buildRichTensors(env, obs)
        try {
            session.run(inputs).use { res ->
                val tech = row(res.get(SampleSchema.OnnxContract.OUTPUT_TECH).get() as OnnxTensor)
                val policy = row(res.get(SampleSchema.OnnxContract.OUTPUT_POLICY).get() as OnnxTensor)
                tech to policy
            }
        } finally {
            for (t in inputs.values) try { t.close() } catch (_: Exception) {}
        }
    }
```
`Timers.timeThis` (companion, `Timers.kt:106-107`) is a **no-op unless timing is enabled**
(`timingEnabledTimestamp != TIMING_DISABLED`, set only by `startTiming()` which itself returns
early unless `Log.shouldLog()` — `Timers.kt:19-27`). So the wrap is zero-overhead in
generation/eval and only records when the bench (or a debug session) turns timing on. **Because
Timers is Log-gated and cleared per turn, the bench does NOT depend on the Timers aggregate for
its headline number** — it derives ms/decision from wall-clock ÷ `OnnxPolicy.decisionCount()`
(`OnnxPolicy.kt:85-86`), which is always populated. The Timers wrap is the required per-span
diagnostic (and lets a `--debug` run print the onnxForward p50/p95 via `Timers.endTiming()`),
not the gate input.

### D8.2 Extend SimBenchmark with an ONNX-policy mode
`SimBenchmark.main` (`SimBenchmark.kt:50-147`) currently always runs the heuristic A1/A2/B
parts. Add an arg-dispatched ONNX mode that (a) still computes the heuristic baseline turns/s on
the **same** base game (so the 70% gate is apples-to-apples), then (b) runs the same batch with
OnnxPolicy injected as the learner, and (c) emits `BENCH| ONNX ...` lines + a `BENCH| RUNG ...`
verdict line the ladder parses.

The injection is **identical to `SelfPlayRunner.eval` (`SelfPlayRunner.kt:193-203`)** — reuse
`RoutingPolicy(LEARNER, onnx, RandomPolicy)` + `DataPlaneContext(config, vocab, policy, fp)` +
`Simulation(...)`. SimBenchmark already constructs a base game, but with a *different* config
(Medium, 6 majors+6CS+barbs) than SelfPlayRunner (Tiny 2-civ). The D8 spec says "Medium, 6
majors+6CS+barbs" → keep SimBenchmark's `buildBaseGame()` (`:157-181`). The learner civ is one
of the `BenchCiv*` nations; route it through OnnxPolicy and leave the rest heuristic.

Concretely add to `SimBenchmark.kt`:

```kotlin
// imports
import com.unciv.logic.simulation.dataplane.DataPlaneContext
import com.unciv.logic.simulation.dataplane.DataPlaneHooks
import com.unciv.logic.simulation.dataplane.RandomPolicy
import com.unciv.logic.simulation.dataplane.RoutingPolicy
import com.unciv.logic.simulation.dataplane.RulesetFingerprint
import com.unciv.logic.simulation.dataplane.SampleCaps
import com.unciv.logic.simulation.dataplane.SampleConfig
import com.unciv.logic.simulation.dataplane.SampleSchema
import com.unciv.logic.simulation.dataplane.Vocab
import com.unciv.logic.automation.Timers

private const val GATE_FRACTION = 0.70   // reject a rung below 70% of heuristic turns/s

// in main(), right after bootstrap + buildBaseGame, BEFORE the existing A1/A2/B:
val onnxModel = args.firstOrNull { it.endsWith(".onnx") }
// the learner nation is the first bench major civ
val learnerCivId = "BenchCiv1"
```

`main` dispatch — when an `.onnx` arg is present, run heuristic baseline (single + MT) on the
base game, then the ONNX rung, then the gate:

```kotlin
if (onnxModel != null) {
    runOnnxRung(newGame, onnxModel, learnerCivId)
    System.out.flush(); System.exit(0)
}
// else: existing heuristic-only path (unchanged)
```

The rung runner — note the heuristic baseline here uses the SAME MT batch shape as the ONNX run
so turns/s is comparable; A* OFF for both (the deterministic apples-to-apples already used at
`:78,89`):

```kotlin
@OptIn(ExperimentalTime::class)
private fun runOnnxRung(base: GameInfo, modelPath: String, learnerCivId: String) {
    val cores = Runtime.getRuntime().availableProcessors(); val threads = minOf(cores, 8)
    setAStar(false)

    // --- heuristic baseline on the SAME base game / batch shape (the 100% reference) ---
    runFullGames(base, gamesPerThread = 1, threads = threads, maxTurns = MT_CAP_TURNS) // warm
    val baseline = runFullGames(base, gamesPerThread = 1, threads = threads, maxTurns = MT_CAP_TURNS)
    println("BENCH| ONNX baseline (heuristic, A*off, x$threads): ${"%.1f".format(baseline.turnsPerSec)} turns/s")

    // --- ONNX rung: inject OnnxPolicy as the learner via the eval-identical path ---
    val ruleset = base.ruleset
    val fp = RulesetFingerprint.compute(ruleset)
    val vocab = Vocab(ruleset)
    val config = SampleConfig(enabled = false, deterministicShuffle = true, caps = SampleCaps.DEFAULT)
    val onnx = OnnxPolicy(modelPath, vocab, config, DataPlaneHooks.defaultRngFor(),
                          eval = true, SampleSchema.VERSION, fp)
    val policy = RoutingPolicy(learnerCivId, onnx, RandomPolicy(DataPlaneHooks.defaultRngFor()))
    val dp = DataPlaneContext(config, vocab, policy, fp)

    Timers.singleton.startTiming()  // enable onnxForward span recording (needs Log debug; see note)
    val sim = Simulation(base, simulationsPerThread = 1, threadsNumber = threads,
                         maxTurns = MT_CAP_TURNS, statTurns = listOf(),
                         dataPlane = dp, scoreLeaderOnTimeout = true, seedBase = 0)
    val t0 = System.nanoTime()
    sim.start()
    val secs = (System.nanoTime() - t0) / 1e9
    val turns = sim.steps.sumOf { it.turns }
    val decisions = onnx.decisionCount()
    onnx.close()
    Timers.singleton.endTiming()    // prints onnxForward p50/p95 IF Log.shouldLog()

    val turnsPerSec = turns / secs
    val msPerDecision = if (decisions > 0) secs * 1000.0 / decisions else Double.NaN
    val frac = turnsPerSec / baseline.turnsPerSec
    val pass = frac >= GATE_FRACTION
    println("BENCH| ONNX rung model=$modelPath: ${"%.1f".format(turnsPerSec)} turns/s, " +
            "${"%.2f".format(msPerDecision)} ms/decision, decisions=$decisions " +
            "(${"%.0f".format(frac*100)}% of heuristic)")
    println("BENCH| RUNG verdict=${if (pass) "PASS" else "REJECT"} " +
            "turns_per_sec=${"%.3f".format(turnsPerSec)} ms_per_decision=${"%.4f".format(msPerDecision)} " +
            "frac_of_baseline=${"%.4f".format(frac)} gate=$GATE_FRACTION")
    if (!pass) System.exit(3)  // non-zero rc → run_loop treats the rung as budget-violating
}
```

**Why decisionCount() not Timers for ms/decision:** `decisionCount()` increments once per legal
net decision (`OnnxPolicy.kt:98`); two head-calls per turn share ONE forward via the memo
(`OnnxPolicy.kt:104-110`), so wall-clock ÷ decisions is the true per-decision cost the gate
cares about. The `onnxForward` Timers span (only when `Log.shouldLog()`) gives the isolated ORT
latency distribution for diagnostics. **Note (limitation):** `Timers.singleton.startTiming()`
short-circuits unless `Log.shouldLog()` — if the bench wants the span table it must raise the log
level (e.g. `Log.backend = DesktopLogBackend()` already set in `main`, plus a debug threshold).
This is a diagnostic nicety; the gate never depends on it.

### D8.3 Gradle: pass the model path to simBench
`desktop/build.gradle.kts:60-67` registers `simBench` with no args. JavaExec ignores `--args` for
non-`application` tasks unless `args(...)` is set; add a property passthrough:

```kotlin
tasks.register<JavaExec>("simBench") {
    dependsOn(tasks.getByName("classes"))
    mainClass.set("com.unciv.app.desktop.SimBenchmark")
    classpath = sourceSets.main.get().runtimeClasspath
    workingDir = assetsDir
    isIgnoreExitValue = false   // CHANGED from true: let rc=3 (rung REJECT) propagate to run_loop
    jvmArgs = listOf("-Xmx4G")
    (project.findProperty("benchArgs") as String?)?.let { args(it.split(" ")) }
}
```
Invoke: `./gradlew :desktop:simBench -PbenchArgs="onnx <abs/policy_round_r.onnx>"`. The ladder
(D7) shells this and parses the `BENCH| RUNG ...` line + rc.

### D8 reporting
Every rung prints exactly two machine-readable lines (`BENCH| ONNX rung ...` human + `BENCH|
RUNG verdict=...` parseable). The ladder records `turns_per_sec`, `ms_per_decision`,
`frac_of_baseline`, `verdict` per rung.

---

## D7 — Ladder orchestration in run_loop.py

### D7.1 `--variant structured` (alias `rich-v2`) dispatch
`train_round` (`run_loop.py:74-94`) already dispatches `rich-critic` through
`token_specs_from_schema` + `train_actor_critic_rich` + the `("rich", token_specs)` export mode.
The structured encoder is the SAME contract (multi-tensor, masks) plus the new neighbor
index/mask inputs (sibling cluster C2/C3) — so it reuses the rich train/export plumbing with a
different `nn.Module`. Add a branch:

```python
# run_loop.py:87 — extend train_round
if variant in ("rich-critic", "structured", "rich-v2"):
    token_specs = contract.token_specs_from_schema(schema_path)
    structured = variant in ("structured", "rich-v2")
    net, stats = tr.train_actor_critic_rich(
        trajectories_or_steps, dims, token_specs, epochs=args.epochs, lr=args.lr, seed=seed,
        gamma=args.gamma, lam=args.lam, value_coef=args.value_coef,
        entropy_coef=args.entropy_coef, clip_eps=args.clip_eps,
        structured=structured, rung=getattr(args, "_rung", "small"))  # NEW kwargs (C2/C3 add)
    return net, stats, ("rich", token_specs)
```
`train_actor_critic_rich` (`train.py:240-275`) gains `structured`/`rung` kwargs that, when set,
build `StructuredPolicyValueNet(dims, token_specs, rung=...)` instead of `RichPolicyValueNet`
(`train.py:261`) — the encoder-agnostic `_optimize_actor_critic` core (`:115-194`) is UNTOUCHED
(hard constraint). The export path (`run_loop.py:214-222`) is unchanged: `export_rich` already
emits whatever inputs the net's `_RichPolicyOnly` wrapper traces (the new neighbor index/mask
ride through the same positional-names mechanism C2/C3 extend in `export_onnx.py:109-115`).

Also widen the CLI choices (`run_loop.py:142`):
```python
ap.add_argument("--variant",
                choices=["v1-reinforce", "blind-critic", "rich-critic", "structured", "rich-v2"],
                default="blind-critic", ...)
```

### D7.2 Ladder definition + signals
Add a rung ladder (small→medium→large = constructor-arg capacity of the ONE concrete
`StructuredPolicyValueNet`, per decision FND-0001/0003 — NOT a plugin framework). New CLI args:

```python
ap.add_argument("--ladder", action="store_true",
                help="demand-driven rung escalation (structured variant)")
ap.add_argument("--rungs", default="small,medium,large")
ap.add_argument("--rung-window", type=int, default=3, help="K rounds for the rising/gap test")
ap.add_argument("--gap-threshold", type=float, default=0.15,
                help="max train-eval gap to call a rung underfitting (escalate-eligible)")
ap.add_argument("--simbench", action="store_true",
                help="run the D8 throughput gate on each round's model before accepting the rung")
```

Per-round signals the ladder consumes (all already in the curve row, `run_loop.py:226-231`):
- **eval win-rate trend** over last `rung-window` rounds (rising = monotone-ish increase, i.e.
  `wr[-1] > wr[-K] + epsilon`).
- **train/eval gap** = a proxy for over/under-fitting. We do not have a held-out train win-rate,
  so use: `gap = clamp(mean_value_on_wins - eval_winrate)` is NOT meaningful; instead use the
  **value-fit signal**: high `value_loss` with rising eval winrate ⇒ underfitting (escalate);
  low `value_loss` + flat/declining eval winrate ⇒ saturated/overfit-for-this-capacity (don't
  escalate, stop). Concretely `underfitting = (value_loss > gap_threshold) and rising`. This is
  the available, source-grounded proxy (`stats["value_loss"]`, `:228`). **OPEN QUESTION below.**
- **D8 budget** = `frac_of_baseline >= 0.70` from the simbench RUNG line.
- **OOM** = the gradle/train subprocess raising / rc≠0 on the whole-round dense batch
  (`train.py:265` builds one padded batch over all ~1261 Medium tiles).

### D7.3 Orchestration logic
A thin controller wrapping the existing per-round body. Escalate one rung when BOTH (a) rising
AND (b) underfitting (small train/eval gap toward the model = it can absorb more capacity); STOP
when a rung doesn't improve over the previous rung's best OR violates D8 OR OOMs.

```python
def _rising(wr: list[float], k: int, eps: float = 0.005) -> bool:
    if len(wr) < k + 1: return True   # too early to judge → keep going on the current rung
    return wr[-1] > wr[-1 - k] + eps

def run_ladder(args, ...):
    rungs = args.rungs.split(",")
    ri = 0
    rung_best = {}            # rung -> best eval winrate seen
    r = 0
    while r < args.rounds and ri < len(rungs):
        args._rung = rungs[ri]
        try:
            row = run_one_round(r, args)         # the existing body (gen→train→export→eval)
        except OOMError:                          # raised by train_round on CUDA/host OOM
            log_decision(rungs[ri], r, "STOP", "OOM on whole-round dense batch")
            break
        if args.simbench:
            bench = run_simbench_gate(row["model_path"])   # parses BENCH| RUNG line + rc
            row.update(bench)
            if not bench["pass"]:
                log_decision(rungs[ri], r, "STOP", f"D8 budget {bench['frac']:.2f}<0.70")
                break
        wr = [x["winrate"] for x in rows_on_rung(rungs[ri])]
        rising = _rising(wr, args.rung_window)
        underfit = row["value_loss"] > args.gap_threshold
        prev_best = rung_best.get(rungs[ri - 1]) if ri > 0 else None
        if prev_best is not None and row["winrate"] <= prev_best + 0.005 and not rising:
            log_decision(rungs[ri], r, "STOP", "no improvement over previous rung")
            break
        rung_best[rungs[ri]] = max(rung_best.get(rungs[ri], 0.0), row["winrate"])
        if rising and underfit:
            log_decision(rungs[ri], r, "SCALE_UP", f"rising & underfit (vloss={row['value_loss']:.3f})")
            ri += 1
        else:
            log_decision(rungs[ri], r, "HOLD", f"rising={rising} underfit={underfit}")
        r += 1
```

`log_decision` appends to a `ladder.jsonl` next to `curve.csv`:
`{"round": r, "rung": rung, "decision": "...", "reason": "...", "winrate":..., "value_loss":...,
"turns_per_sec":..., "ms_per_decision":..., "frac_of_baseline":...}`. This satisfies "Record
decision + signals per rung."

`run_simbench_gate` shells the D8 task:
```python
def run_simbench_gate(model_path: Path) -> dict:
    cmd = [str(GRADLEW), ":desktop:simBench", "--console=plain",
           f"-PbenchArgs=onnx {model_path}"]
    p = subprocess.run(cmd, cwd=str(REPO), env=_env(), capture_output=True, text=True,
                       timeout=args.gradle_timeout)
    m = re.search(r"BENCH\| RUNG verdict=(\w+) turns_per_sec=([\d.]+) "
                  r"ms_per_decision=([\d.eE+-]+) frac_of_baseline=([\d.]+)", p.stdout + p.stderr)
    if not m:
        raise RuntimeError("simBench produced no RUNG line")
    return {"pass": m.group(1) == "PASS" and p.returncode == 0,
            "turns_per_sec": float(m.group(2)), "ms_per_decision": float(m.group(3)),
            "frac": float(m.group(4))}
```

OOM detection: wrap the `train_round` call in `train.py`/`run_loop.py` to catch
`torch.cuda.OutOfMemoryError` and host `MemoryError` and re-raise a typed `OOMError` the ladder
catches. (Micro-batching the dense whole-round batch is **explicitly out of scope** — future
work, per the prompt.)

---

## D10 — Experiment: structured vs v3 rich-pool

### D10.1 Pre-registered acceptance (PRIMARY)
**structured BEATS v3 rich-critic on Medium at p<0.05**, via a two-proportion z-test over the
final 200-game eval; **Tiny must not regress**. The z-test is the EXISTING
`analyze._two_proportion_z` (`analyze.py:50-59`) — no new statistics code. Reuse
`analyze.main`'s ceiling-eval machinery (`:120-138`) which already runs a fresh high-N Medium
eval on the two final models and applies the z-test; extend it to compare `structured-medium` vs
`rich-critic-medium` instead of rich-vs-blind:

```python
# analyze.py — add a v4 comparison block (mirrors AC3 :120-133)
structured = {v: _read_curve(root / f"{v}-medium" / "curve.csv") for v in ["rich-critic", "structured"]}
sm = root / "structured-medium" / f"policy_round_{len(structured['structured'])-1}.onnx"
rm = root / "rich-critic-medium" / f"policy_round_{len(structured['rich-critic'])-1}.onnx"
re_ = run_loop.evaluate(rm, args.ceiling_games, args.turn_cap, args.threads, args.eval_seed, args.gradle_timeout, "Medium")
se  = run_loop.evaluate(sm, args.ceiling_games, args.turn_cap, args.threads, args.eval_seed, args.gradle_timeout, "Medium")
z, p = _two_proportion_z(int(se["wins"]), int(se["games"]), int(re_["wins"]), int(re_["games"]))
ac1 = {"rich_v3": re_, "structured": se, "z": z, "p_one_sided_structured_gt_rich": p,
       "significant_p05": p < 0.05}
```
`ceiling_games` default is already 200 (`analyze.py:77`) → "final 200-game eval" is the existing
default. The z is one-sided P(structured>rich).

### D10.2 Exact run commands, seeds, budgets (budget held CONSTANT vs v3)
"Hold budget constant" = identical `--rounds`, `--gen-games`, `--eval-games`, `--epochs`,
`--turn-cap`, threads, and the SAME seeds for both arms. Four curves (variant × {Tiny, Medium}):

```bash
ROOT=$REPO/training-runs/v4-experiment
COMMON="--rounds 12 --gen-games 24 --eval-games 100 --turn-cap 250 --epochs 8 \
        --threads 12 --gen-seed 1000 --eval-seed 999000"

# v3 baseline (rich-pool) — Tiny + Medium
python -m unciv_train.run_loop --variant rich-critic --map-size Tiny   --out $ROOT/rich-critic-tiny   $COMMON
python -m unciv_train.run_loop --variant rich-critic --map-size Medium --out $ROOT/rich-critic-medium $COMMON

# v4 structured — Tiny + Medium (SAME budget + SAME seeds)
python -m unciv_train.run_loop --variant structured  --map-size Tiny   --out $ROOT/structured-tiny    $COMMON
python -m unciv_train.run_loop --variant structured  --map-size Medium --out $ROOT/structured-medium  $COMMON

# Acceptance: final 200-game Medium z-test + overlay curves (Tiny + Medium)
python -m unciv_train.analyze --root $ROOT --ceiling-games 200 --turn-cap 250 --threads 12 \
       --eval-seed 4242424
```
Seeds: per-round gen seed = `gen_seed + r*1000` (`run_loop.py:193`), eval seed fixed at
`eval_seed` (`:224`) — both arms identical ⇒ same maps, same opponent rolls, only the encoder
differs. The ceiling eval uses a DIFFERENT seed (`4242424`, `analyze.py:80`) than the per-round
eval so the final acceptance is not the model's training-feedback eval (no leakage).

### D10.3 z-test computation (verbatim, already in tree)
`analyze.py:50-59`: pooled-proportion one-sided z, `p_one = 0.5*erfc(z/√2)` — no scipy. For
`structured` (w1,n1) vs `rich` (w2,n2) at n1=n2=200 it is exactly the AC3 formula re-pointed.
Significant iff `p < 0.05`.

### D10.4 Guardrails (the v3 decline signature)
Two guardrails reported every round, both on Medium:
1. **train/eval gap per rung** — from the ladder's `value_loss` + `winrate` (D7.2) and the
   curve's `mean_value`/`grad_norm` (`run_loop.py:229`). A widening gap (eval winrate falling
   while value_loss stays low) is the over-capacity signature.
2. **round-over-round Medium curve** — the existing `plot()` (`run_loop.py:113-123`) writes
   `curve.png`; add a guardrail check that flags if Medium winrate **declines** over the last K
   rounds (`wr[-1] < wr[-1-K] - 0.02`) — this is the v3 decline signature the user accepted as a
   confound. The analyze overlay (`_overlay`, `analyze.py:62-71`) plots structured vs rich on
   both Tiny and Medium.

### D10.5 Confound statement (MUST be in the report, verbatim intent)
> v4 adds encoder capacity but leaves from-scratch-per-round training (the recorded root cause of
> the v3 Medium regression, per MEMORY: selfplay-roadmap-bottleneck) UNTOUCHED. A positive result
> means the structured encoder helps **despite** that confound; a null/negative result does NOT
> isolate the encoder from the training regime. Per the v2 ethos, a null or negative result is
> reported plainly and is a valid outcome — the pre-registered metric (D10.1) is falsifiable.

OOM / micro-batching of the whole-round dense batch (`train.py:265`) is **out of scope** (future
work), per the prompt and decisions.md.

### D10.6 Parity guard extension (cross-cluster dependency)
`test_parity.py:77-131` (`test_jvm_python_rich_logits_match`) must be extended by C2/C3 to feed
the new neighbor-index/mask tensors (and an adjacency-parity test must guard the Python↔Kotlin
neighbor replication, FND-0036). C4 depends on that test passing before any rung runs — the D8
gate and D10 z-test are only meaningful if JVM inference == Python training numerics. I flag this
as the cross-cluster precondition; the fixture/dims widening lives in C2/C3's edits to
`test_parity.py` and `parityRunRich` (`SelfPlayRunner.kt:351-396`).

## Exact edits
- **desktop/src/com/unciv/app/desktop/OnnxPolicy.kt** [forwardRich, lines 127-138]: Wrap the buildRichTensors+session.run body in Timers.timeThis("onnxForward"){ ... } (companion form com.unciv.logic.automation.Timers.timeThis). Keep the try/finally tensor-close inside the block; return the pair as the block result.
  _why:_ D8 requirement: instrument the per-decision ONNX forward. No-op unless Timers timing is enabled, so zero overhead in gen/eval.
- **desktop/src/com/unciv/app/desktop/SimBenchmark.kt** [imports (top, after line 22) + main() dispatch (after buildBaseGame, ~line 65) + new private fun runOnnxRung]: Add imports for DataPlaneContext, DataPlaneHooks, RandomPolicy, RoutingPolicy, RulesetFingerprint, SampleCaps, SampleConfig, SampleSchema, Vocab, Timers. Add const GATE_FRACTION=0.70. In main(), detect an .onnx arg; if present call runOnnxRung(newGame, model, "BenchCiv1") then exit. Add runOnnxRung: heuristic-baseline runFullGames on the same base, then inject OnnxPolicy via RoutingPolicy+DataPlaneContext+Simulation (eval-identical to SelfPlayRunner.eval:193-203), measure turns/s + ms/decision (wall-clock / onnx.decisionCount()), print BENCH| ONNX + BENCH| RUNG verdict=PASS|REJECT lines, System.exit(3) on reject.
  _why:_ D8: ONNX-policy throughput mode + 70%-of-heuristic gate, reusing the existing eval injection path (reuse-over-rebuild).
- **desktop/build.gradle.kts** [simBench task block, lines 60-67]: Set isIgnoreExitValue=false (was true) so rc=3 propagates; add (project.findProperty("benchArgs") as String?)?.let { args(it.split(" ")) } to pass the model path.
  _why:_ D8: the ladder must pass the rung's .onnx and observe a non-zero rc when the gate fails.
- **python/unciv_train/run_loop.py** [train_round, lines 87-94]: Replace the `if variant == "rich-critic"` branch with `if variant in ("rich-critic","structured","rich-v2")`; set structured = variant in ("structured","rich-v2"); pass structured=structured, rung=getattr(args,"_rung","small") into train_actor_critic_rich. Export mode stays ("rich", token_specs).
  _why:_ D7: --variant structured (alias rich-v2) dispatch reusing the rich train/export plumbing with the new nn.Module.
- **python/unciv_train/run_loop.py** [argparse --variant choices, line 142]: Add "structured" and "rich-v2" to the choices list.
  _why:_ D7: expose the new variant on the CLI.
- **python/unciv_train/run_loop.py** [new CLI args after line 164 + new run_ladder/run_simbench_gate/_rising/log_decision functions + main() ladder branch]: Add --ladder, --rungs (default small,medium,large), --rung-window (3), --gap-threshold (0.15), --simbench. Add _rising(), run_simbench_gate() (shell :desktop:simBench -PbenchArgs and parse BENCH| RUNG), log_decision() (append ladder.jsonl). Refactor the per-round body (188-249) into run_one_round(r,args) and add run_ladder() that escalates one rung when rising&&underfit, stops on no-improvement/D8-violation/OOM, recording decision+signals per rung.
  _why:_ D7: demand-driven ladder orchestration with the exact start/scale-up/stop rules and per-rung recording.
- **python/unciv_train/train.py** [train_actor_critic_rich signature + net construction, lines 240-261]: Add structured: bool=False, rung: str="small" kwargs; when structured, construct StructuredPolicyValueNet(dims, token_specs, rung=rung) instead of RichPolicyValueNet (line 261). _optimize_actor_critic core UNCHANGED.
  _why:_ D7: swap the nn.Module only, honoring the frozen training core (hard constraint).
- **python/unciv_train/analyze.py** [AC3 block + main, lines 120-143]: Add a v4 acceptance block: read structured-medium + rich-critic-medium curves, run a fresh 200-game Medium eval on each final model, apply _two_proportion_z(structured, rich) for one-sided P(structured>rich), record significant_p05; add Tiny non-regression check (structured-tiny final winrate >= rich-critic-tiny final - 0.02); extend overlays to include structured on Tiny and Medium.
  _why:_ D10: PRIMARY acceptance (structured>v3 on Medium p<0.05) reusing the existing z-test; Tiny-no-regress guardrail.

## New inputs/tensors
- No NEW ONNX input tensors are authored by C4. C4 consumes whatever C2/C3 add. The neighbor-index [N,6] (int64) and neighbor-mask [N,6] (float32) that the gather-GNN needs are MODEL INPUTS added by C3 to export_rich (export_onnx.py:109-115) and built JVM-side in OnnxPolicy.buildRichTensors (OnnxPolicy.kt:153-165) from the live TileMap; C4's D8 bench is the runtime that first exercises them (so they must be wired before the ladder's small rung runs).
- C4 adds NO shard tensors. The coords (spatial ch13/ch14) and map-dims (in global) that the Python adjacency builder reads are added by C1 (Featurizer); C4 only reads them transitively via OnnxPolicy at inference and via the rich trainer at train time.

## Lockstep sites
- SampleSchema.VERSION 2->3 (SampleSchema.kt:22) — D8/D10 only consume it via OnnxPolicy(SampleSchema.VERSION) at SelfPlayRunner.kt:163,193,251 and the new SimBenchmark.runOnnxRung; no width edits in C4 but the bench/eval gate FAILS LOUD if a rung's model schema_version != live (OnnxPolicy.kt:58) — this is the desired guard
- OnnxContract.CONTRACT_VERSION_RICH 2->3 (SampleSchema.kt) + contract.py:18 — OnnxPolicy.kt:59 gate must accept the structured model's contract version; C4's bench reuses OnnxPolicy so it inherits whatever C1 sets (gate must accept {1,2,3} or {2,3})
- OnnxPolicy.FALLBACK_WIDTH (OnnxPolicy.kt:150-151) and contract._TOKEN_WIDTH_FALLBACK (contract.py:81-82) — C4 does not edit these but the bench exercises them at inference; must stay in lockstep (C1 makes them fail-loud vs schema)
- SimBenchmark learner nation id "BenchCiv1" must match a major civ produced by buildBaseGame() (SimBenchmark.kt:159-161) — if NUM_MAJOR_CIVS naming changes, the bench routing breaks silently (learner never routed to net, decisionCount=0)
- analyze.py ceiling-eval model path pattern policy_round_{N-1}.onnx (analyze.py:123-124) must match run_loop.py:211 model_path naming for both rich-critic-medium and structured-medium
- run_loop.py CURVE_COLS (run_loop.py:109-110) — if D7 adds rung/turns_per_sec/ms_per_decision to the row, they go to ladder.jsonl NOT curve.csv (keep curve.csv schema stable for analyze._read_curve which reads round/winrate/games/pval only, analyze.py:36-37)

## Export safety
C4 introduces NO new ONNX ops and does not author the encoder — it consumes the exported model. opset-17 safety for C4 reduces to: (1) the D8 bench loads the rung's .onnx through the unchanged OnnxPolicy/ORT session path (OnnxPolicy.kt:50, SessionOptions intra-op=1) — any opset-17-incompatible op authored by C2/C3 would surface here as an ORT session-create failure at the FIRST rung, which is exactly the spec's 'validate export on the SMALL rung FIRST' gate (the ladder starts at small, so the smallest structured model is the first thing the bench loads). (2) The new neighbor-index/mask MODEL INPUTS that C2/C3 add to export_rich (export_onnx.py:109-115, positional names + dynamic axes {1:'n_<name>'}) must be exercised by the bench: because SimBenchmark reuses OnnxPolicy.forwardRich -> buildRichTensors which iterates RICH_TOKEN_NAMES + masks (OnnxPolicy.kt:153-165), C4 depends on C3 having the JVM build neighbor-index/mask from the live TileMap and added them to buildRichTensors AND to the rich-input inventory check (OnnxPolicy.kt:66-74). If they are not built JVM-side, the bench session.run throws (missing input) — a fail-loud, not a silent wrong answer. (3) The gather-GNN (Gather+Mul+ReduceSum/Mean, all core <=opset13) and hand-rolled attention (Linear+matmul+masked softmax) are the C2/C3 export-safe realizations; C4's contribution is that the bench is the runtime smoke test that proves the exported small-rung graph actually executes in ORT before any larger rung or the D10 z-test runs. No scatter_add/index_add/MultiheadAttention anywhere in the path C4 touches.

## Determinism/provenance
D8 bench determinism: runOnnxRung uses Simulation(seedBase=0) and DataPlaneHooks.defaultRngFor() exactly like SelfPlayRunner.eval, eval=true ⇒ OnnxPolicy is argmax (deterministic, OnnxPolicy.kt:28-29,97 via MaskedChoice eval-mode). Same base game (buildBaseGame, fixed) ⇒ turns/s is reproducible modulo wall-clock noise; the gate compares ratio-to-baseline measured in the SAME process/run to cancel machine variance. Provenance: the bench reuses OnnxPolicy's provenance gate (OnnxPolicy.kt:51-74) so a rung model with a mismatched schema_version/contract_version/ruleset_fingerprint FAILS LOUD at bench load (not a silent miscompare) — the bench inherits C1's fingerprint bump (adding spatial channels changes Vocab.canonicalSections fingerprint, so old/v3 models are correctly refused by the structured rung's gate). D10 determinism: both arms share gen-seed/eval-seed (run_loop.py:193,224) so maps+opponent rolls are identical; the ceiling eval uses a held-out seed 4242424 (analyze.py:80) distinct from training-feedback eval seeds ⇒ no eval leakage into the acceptance z-test. The two-proportion z (analyze.py:50-59) is pure-math (erfc), no RNG.

## Open questions
- D7.2 train/eval-gap proxy: there is NO held-out train win-rate recorded (the trainer only emits value_loss/grad_norm/mean_value at run_loop.py:228-229). I propose 'underfitting := value_loss > gap_threshold AND eval-winrate rising' as the available proxy. The prompt says '(b) train/eval gap small (underfitting)' — a true train/eval gap needs a train-set evaluation pass the pipeline does not currently do. Should C4 add a lightweight train-batch argmax win-rate (re-running eval on the gen seeds), or is the value_loss proxy acceptable? This is the one design fork I cannot settle from source.
- D8 learner nation: SimBenchmark.buildBaseGame names majors BenchCiv1..6 (SimBenchmark.kt:159-161) but routes the WHOLE game (no pinned LEARNER constant like SelfPlayRunner). Confirm routing OnnxPolicy to exactly one major (BenchCiv1) and leaving the other 5 heuristic is the intended 'data-gen throughput' measurement, vs routing ALL majors to the net (heavier ONNX load, more realistic for a self-play-all-civs regime). The prompt says 'inject OnnxPolicy as the learner' (singular) ⇒ I chose one civ; flag for confirmation.
- D8 Timers span table requires Log.shouldLog()=true (Timers.kt:19) which the bench does not force. The wall-clock/decisionCount ms/decision is the gate input and is always available, but if the spec wants the onnxForward p50/p95 printed, the bench must raise the desktop log level. Confirm whether the span table is required output or diagnostic-only.
- D7 OOM detection: torch.cuda.OutOfMemoryError vs host MemoryError on the CPU whole-round dense batch (train.py:265). The current pipeline is CPU torch (no CUDA seen). On CPU, OOM may manifest as the OS killing the process (no catchable exception) — the ladder may need to detect a non-zero/killed train subprocess rc rather than a Python exception. Micro-batching is out of scope, but OOM DETECTION robustness on CPU is an open question.
- D8 gate fraction 0.70 is the prompt's spec, applied to MT turns/s with A* OFF. Confirm the gate should use the A*-OFF heuristic baseline (deterministic, the existing apples-to-apples at SimBenchmark.kt:78,89) rather than best-of(A*on,A*off) — A* ON can be faster (SimBenchmark.kt:127), making the gate stricter.

## Risks
- SimBenchmark uses 6 generic BenchCiv nations with no FairOpponentModel-meaningful identity; routing one to OnnxPolicy may produce degenerate decisions (the net was trained on the 2-civ Tiny/Medium SelfPlayRunner setup, not 6-major Medium). The turns/s measurement is still valid (it measures inference cost, not skill), but ms/decision could differ from the real gen config. -> D8 measures THROUGHPUT (turns/s, ms/decision), not win-rate — degenerate decisions do not invalidate the cost gate. Document that the bench config (6 majors) is a throughput stress-test, distinct from the 2-civ gen/eval config. If realism matters, add a second bench config matching SelfPlayRunner's 2-civ setup.
- isIgnoreExitValue=false on simBench means ANY non-rung gradle failure (compile, asset load) now fails the task and could be misread by the ladder as a rung REJECT. -> run_simbench_gate distinguishes: REJECT requires the BENCH| RUNG verdict=REJECT line AND rc=3; a missing RUNG line raises RuntimeError (build failure), which the ladder treats as an abort, not a rung-stop. rc is checked alongside the parsed verdict.
- The D7 ladder's 'rising AND underfit' escalation could oscillate or never escalate if value_loss is consistently low even while underfitting (the value head near-zero init, model.py:_small_init_value_head, keeps early value_loss small). -> Gate escalation on the rung-window trend not a single round; gap_threshold is a CLI knob (default 0.15) tunable per observed value_loss scale. log_decision records the signals every round so the threshold can be calibrated from a dry run.
- D10 budget-constant claim is undermined if structured's larger model changes per-round wall-clock so much that the gradle-timeout (1800s, run_loop.py:164) trips on Medium, aborting the run and producing an incomparable (shorter) curve. -> The D8 gate (>=70% turns/s) caps per-decision cost BEFORE the full run; if a rung fails the gate the ladder stops it rather than letting it run into a timeout. For the fixed-variant D10 runs (no ladder), raise --gradle-timeout for the structured-medium arm and verify both arms complete the same --rounds (compare curve.csv row counts before running analyze).
- analyze.py reuses run_loop.evaluate which routes the structured (contract-v3) model through OnnxPolicy; if C1/C3 have not bumped the OnnxPolicy contract gate to accept v3 (OnnxPolicy.kt:59-62), the ceiling eval and the bench both throw at load, blocking the entire D10 acceptance. -> C4 explicitly depends on C1's CONTRACT_VERSION_RICH bump and the gate accepting the structured contract; flagged as a lockstep precondition. The failure is fail-loud (check() error), not a silent wrong z-test, so it cannot corrupt the acceptance result.

## VERDICT
```json
{
  "cluster": "C4 \u2014 throughput guard (D8), ladder orchestration (D7), experiment (D10)",
  "export_safe": true,
  "lockstep_complete": true,
  "seam_preserved": true,
  "determinism_ok": false,
  "parity_feasible": true,
  "issues": [
    {
      "severity": "critical",
      "issue": "D8 ruleset-fingerprint mismatch makes the whole throughput gate unrunnable. SimBenchmark.buildBaseGame() injects synthetic nations BenchCiv1..6 into ruleset.nations (SimBenchmark.kt:159-161). RulesetFingerprint.compute hashes Vocab.canonicalSections, which INCLUDES `NATIONS to ruleset.nations.keys.toList()` (Vocab.kt:90, RulesetFingerprint.kt:20). The training pipeline (SelfPlayRunner.setupRuleset, SelfPlayRunner.kt:93-97) registers only SimulationCiv1/SimulationCiv2, so every exported structured/rich model is stamped with that fingerprint. The design's runOnnxRung computes the fingerprint from the bench's mutated ruleset (BenchCiv1..6) and passes it as expectedRulesetFingerprint to OnnxPolicy. OnnxPolicy's gate `check(mFingerprint == expectedRulesetFingerprint)` (OnnxPolicy.kt:63) therefore throws for EVERY rung at model-load time \u2014 the bench cannot load even the correct current-round model. The design's provenance section actively misreads this as a feature ('old/v3 models correctly refused') without seeing that the bench's own ruleset yields a different fingerprint than the model was exported under, so the right model is refused too.",
      "fix": "Do NOT reuse SimBenchmark's 6-major Medium buildBaseGame for the ONNX rung. Instead build the bench game from the SAME ruleset/nation setup the training pipeline uses (SelfPlayRunner.setupRuleset \u2192 SimulationCiv1/SimulationCiv2 + buildBaseGameInfo). Cleanest: have runOnnxRung construct the base via the SelfPlayRunner 2-civ config (or a refactored shared helper) so the fingerprint matches the exported model. This simultaneously fixes the ms/decision attribution problem (see below) and the apples-to-apples baseline. Alternatively (worse) read the model's own stamped fingerprint and pass it through, but that defeats the provenance guarantee."
    },
    {
      "severity": "major",
      "issue": "ms/decision is a distorted, unrepresentative metric in the proposed 6-major / 1-routed config. DataPlaneHooks.handleCivTurn fires per-civ for ALL major civs (DataPlaneHooks.kt:70,99-102), running Featurizer.observe(civ) for every major every turn; RoutingPolicy then sends only BenchCiv1 to OnnxPolicy and the other 5 to RandomPolicy (RoutingPolicy.kt). So decisionCount() counts only BenchCiv1's net decisions, but the wall-clock numerator includes the time of 5 RandomPolicy civs plus 5 extra featurizations per turn. wall-clock / decisionCount therefore overstates the true per-ONNX-forward cost by a large factor and is not the 'true per-decision cost' the design claims. (The ratio-based turns/s gate is still topologically sound, but ms/decision reporting is misleading.)",
      "fix": "Run the bench in the same 2-civ topology as SelfPlayRunner.eval (one learner + one opponent), which both makes decisionCount cover the dominant net-driven civ and matches the real gen/eval config. If a 6-major stress config is also wanted, report it separately and label ms/decision there as 'per-routed-decision incl. opponent-turn overhead', not as isolated forward cost."
    },
    {
      "severity": "major",
      "issue": "Manual Timers start/end around sim.start() is dead/contradictory. Simulation.start() ALREADY calls Timers.singleton.startTiming() at Simulation.kt:109 (which clears all spans) and Timers.singleton.endTiming() at Simulation.kt:180 (which prints and sets timingEnabledTimestamp=TIMING_DISABLED). The design's `Timers.singleton.startTiming()` before sim.start() is wiped by line 109's clear, and its `Timers.singleton.endTiming()` after sim.start() runs against an already-disabled state (no-op, no table). So the design's claim that this 'enables onnxForward span recording' is wrong \u2014 recording is governed entirely by Simulation's internal timing window, and that only records anything if Log.shouldLog() is true at the moment Simulation calls startTiming.",
      "fix": "Remove the manual startTiming/endTiming around sim.start(). The forwardRich Timers.timeThis wrap (D8.1) is fine and zero-overhead, but the span table only appears if Log.shouldLog() is true when Simulation's own startTiming runs. If the p50/p95 table is a required D8 output, raise the desktop log level BEFORE sim.start() (and document the gate never depends on it, which the design does correctly state)."
    },
    {
      "severity": "major",
      "issue": "run_one_round / run_ladder reference a row field that does not exist. The design's run_ladder calls run_simbench_gate(row['model_path']) and rows_on_rung, but the actual per-round row dict (run_loop.py:226-231) has keys round/games/winrate/pval/n_steps/loss/value_loss/entropy/mean_value/grad_norm/diverged/ret_pos/onnx_decisions \u2014 there is no 'model_path' and no per-rung tagging. The refactor 'extract body 188-249 into run_one_round returning row' must also surface the model_path (run_loop.py:211) and the rung tag, or the ladder breaks at the first simbench call.",
      "fix": "In the run_one_round refactor, return model_path (out / f'policy_round_{r}.onnx') and the active rung alongside the row, or have run_ladder reconstruct model_path from (out, r). Add a 'rung' field to the row so rows_on_rung() can filter. None of this touches curve.csv's CURVE_COLS (correctly kept stable for analyze._read_curve)."
    },
    {
      "severity": "minor",
      "issue": "D7.2 train/eval-gap signal is acknowledged as not source-grounded. There is no held-out train win-rate in the pipeline; the only available stats are value_loss/grad_norm/mean_value (run_loop.py:228-229). The proposed proxy 'underfitting := value_loss > gap_threshold AND rising' is a heuristic, and value_loss scale is uncalibrated (value head is small-init, model.py _small_init_value_head, so early value_loss is tiny \u2014 the escalation may never trigger). The design honestly flags this as its one unresolved fork.",
      "fix": "Either (a) add a lightweight train-batch argmax win-rate pass over the gen seeds to get a real train/eval gap, or (b) keep the value_loss proxy but make gap_threshold a calibrated knob and gate on the rung-window trend (design already does the latter). Acceptable as a documented open question; recommend a dry-run calibration before relying on auto-escalation."
    },
    {
      "severity": "minor",
      "issue": "CPU OOM is not a catchable exception. The pipeline is CPU torch (no CUDA); the whole-round dense batch (train.py:265 build_rich_batch) OOM typically manifests as the OS killing the train subprocess, not torch.cuda.OutOfMemoryError or a Python MemoryError the ladder can catch. The design flags this but its run_ladder still catches a Python OOMError.",
      "fix": "Detect OOM via the gradle/train subprocess being killed (non-zero/negative rc, e.g. SIGKILL \u2192 rc 137) rather than a Python exception. The ladder already shells gradle for simbench; apply the same rc-inspection discipline to the train step."
    },
    {
      "severity": "minor",
      "issue": "isIgnoreExitValue=false on simBench (build.gradle.kts:60-67 edit) conflates build failures with rung REJECT. Any compile/asset/JVM failure now also yields a non-zero task rc, which the ladder could misread as a budget violation. The design mitigates by requiring BOTH a parsed 'BENCH| RUNG verdict=REJECT' line AND rc=3, treating a missing RUNG line as an abort \u2014 this is correct, but depends on the System.exit(3) code being distinguishable from gradle's own failure codes.",
      "fix": "Keep the dual check (RUNG line + specific rc). Confirm gradle surfaces the JVM System.exit(3) as a distinguishable task failure (gradle wraps non-zero exec rc); if not reliably distinguishable, prefer parsing the RUNG verdict line as the sole source of truth and ignore rc for PASS/REJECT, using rc only to detect a hard crash (no RUNG line)."
    },
    {
      "severity": "minor",
      "issue": "Inconsistent gradle arg-passing mechanism. selfPlay is invoked with `--args=` (run_loop.gradle_selfplay:42), which JavaExec supports natively; the design instead adds a -PbenchArgs project-property passthrough for simBench. Both work, but the extra mechanism is unnecessary divergence and an extra failure surface (split-on-space breaks paths with spaces).",
      "fix": "Prefer the existing `--args=` convention used by selfPlay (JavaExec supports --args for non-application tasks since Gradle 4.9), e.g. ./gradlew :desktop:simBench --args='onnx <path>', avoiding the -PbenchArgs split-on-space. If keeping -PbenchArgs, guard against spaces in the model path."
    },
    {
      "severity": "minor",
      "issue": "Cross-cluster export shape assumption is slightly mis-stated (out of C4 scope but C4 depends on it). The design says neighbor-index/mask ride through export_rich's token_specs.items() loop (export_onnx.py:109-115). That loop emits every entry as a float [1,N,width] token + a [1,N] presence mask, but neighbor-index is int64 [N,6] (gather indices), not a float token with a presence mask. C2/C3 must add neighbor-index/mask as DISTINCT inputs (int64 index, float/bool mask, fixed degree-6 axis), not via the token loop, and OnnxPolicy's rich-input inventory check (OnnxPolicy.kt:66-74) plus buildRichTensors must build them JVM-side or session.run throws missing-input. C4 is the first runtime that exercises this; the design correctly flags it as a precondition but understates the shape/dtype divergence.",
      "fix": "No C4 edit; note in the plan that C2/C3 must add neighbor-index (int64) and neighbor-mask as separate export inputs with their own dynamic axes, and extend OnnxPolicy.RICH_TOKEN_NAMES/inventory + buildRichTensors accordingly, BEFORE the D8 small-rung smoke runs. The fail-loud behavior (missing input \u2192 throw) is the desired guard."
    }
  ],
  "verdict": "REVISE",
  "corrected_notes": "Verified against real source at /Users/j/Unciv-onnx-selfplay-loop. Most anchors are accurate: SimBenchmark.kt (main 50-147, buildBaseGame 157-181, runFullGames 196-203, isIgnoreExitValue task 60-67), OnnxPolicy.kt (forwardRich 127-138, provenance gate 51-74, decisionCount 85-86/98, constructor signature matches the runOnnxRung call exactly), Timers.kt (timeThis companion 106-107, Log-gated startTiming 19-27), Simulation signature (seedBase/dataPlane/scoreLeaderOnTimeout all present), run_loop.py (train_round 74-94 with rich branch 87-93, --variant choices 142, per-round body 188-249, model_path naming 211), analyze.py (_two_proportion_z 50-59, AC3 block 120-138, ceiling_games default 200 @77, eval_seed 4242424 @80; D10.1 z-arg ordering structured-as-p1 is correct), train.py (train_actor_critic_rich 240-275, RichPolicyValueNet at 261, _optimize_actor_critic untouched), export_onnx.py (export_rich 82-138, positional names + dynamic axes, opset 17). The frozen seam is preserved (only nn.Module swapped). The z-test reuse is correct and parity is feasible.\n\nTHE BLOCKER is the D8 ruleset-fingerprint mismatch (critical issue #1): SimBenchmark's BenchCiv1..6 nations vs the pipeline's SimulationCiv1/2 produce different RulesetFingerprints (nations are in Vocab.canonicalSections @ Vocab.kt:90), so OnnxPolicy's provenance gate refuses EVERY rung model at load \u2014 the entire throughput gate is dead-on-arrival as designed. The fix (build the bench game from the SelfPlayRunner 2-civ ruleset/nation setup instead of buildBaseGame's 6-major Medium) ALSO resolves the major ms/decision-attribution distortion (issue #2) and the apples-to-apples baseline, since DataPlaneHooks runs the policy for all majors but only the routed civ hits ONNX. Secondary fixes: drop the redundant manual Timers start/end around sim.start() (Simulation manages timing internally at 109/180); surface model_path+rung from the run_one_round refactor (the row dict has no model_path field); treat CPU OOM as a killed-subprocess rc not a Python exception. The determinism flag is set false specifically because the fingerprint gate breaks the bench's ability to run deterministically at all (it throws), not because the seeding logic is wrong \u2014 the seeding/argmax determinism reasoning is otherwise sound."
}
```
