#!/usr/bin/env bash
# v7 per-city CONSTRUCTION experiment driver. Resumable: each arm uses --resume (continues from the
# last completed round). Construction is the ONLY axis vs the v6 baseline; replay held at K=4.
#
#   THREADS=12 OUT_ROOT=../training-runs/v7 ./run_v7.sh
#
# FOUR arms run SEQUENTIALLY (a 12/14-core box; parallel arms would oversubscribe CPU). Each is the v5
# continual + v6 replay config (structured, Medium, 16 rounds, gen 16 / eval 80, K=4, micro-batch 256).
# The ONLY axis is --control-construction:
#   small-off  : rung small,  construction OFF  (== v6 tech+policy only; the no-op baseline)
#   small-on   : rung small,  construction ON   (v7: net drives per-city production)
#   medium-off : rung medium, construction OFF
#   medium-on  : rung medium, construction ON
# Then a per-arm 200-game Medium ceiling eval @ eval-seed 4242424 + z-tests vs the fixed blind baseline
# (58/200 = 28.9%) via analyze_v5, and the v7 ON-vs-OFF + 50%-break-even comparison (analyze_v7).
#
# PR5: a bench-onnx throughput PRE-GATE runs FIRST (a fast warmup model benched with construction ON);
# a <70% head ABORTS before the multi-hour batch (override with FORCE=1).
set -uo pipefail
cd "$(dirname "$0")"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home}"
PY="${PY:-python3}"; [ -x .venv/bin/python ] && PY=".venv/bin/python"
ROOT="${OUT_ROOT:-../training-runs/v7}"
TH="${THREADS:-12}"
ROUNDS="${ROUNDS:-16}"
GRADLEW="${GRADLEW:-../gradlew}"
mkdir -p "$ROOT"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

COMMON=(--variant structured --map-size Medium --rounds "$ROUNDS"
        --gen-games 16 --eval-games 80 --turn-cap 250 --threads "$TH"
        --epochs 8 --lr 1e-3 --gamma 0.99 --lam 0.95 --value-coef 0.5
        --entropy-coef 0.01 --clip-eps 0.2 --gen-seed 1000 --eval-seed 999000
        --continual --resume --replay-window 4 --micro-batch-steps 256)

# ---- PR5: throughput pre-gate (bench the per-city construction head BEFORE the multi-hour batch) ----
bench_gate() {
  local warm="$ROOT/_bench_warmup"
  if [ ! -f "$warm/policy_round_0.onnx" ]; then
    log "PR5 warmup: producing a structured construction model (Tiny, 1 round) for the throughput bench"
    "$PY" -m unciv_train.run_loop --variant structured --rung small --map-size Tiny --rounds 1 \
      --gen-games 2 --eval-games 2 --turn-cap 80 --threads "$TH" --control-construction on \
      --no-continual --out "$warm" >/dev/null 2>&1 || { log "PR5 warmup FAILED — skipping pre-gate"; return 0; }
  fi
  local model="$warm/policy_round_0.onnx"
  [ -f "$model" ] || { log "PR5: no warmup model — skipping pre-gate"; return 0; }
  log "PR5 bench-onnx (construction ON) on Medium"
  local out
  out=$("$GRADLEW" selfPlay --console=plain --args="bench-onnx $model 150 Medium $TH 777000 true" 2>&1 | grep -a '^BENCH|' || true)
  log "PR5 ${out:-<no BENCH line>}"
  case "$out" in
    *verdict=REJECT*) if [ "${FORCE:-0}" = "1" ]; then log "PR5 REJECT but FORCE=1 — proceeding"; else
        log "PR5 ABORT: per-city head throughput < 70% of heuristic baseline. Re-run with FORCE=1 to override."; exit 2; fi ;;
    *verdict=PASS*) log "PR5 PASS: per-city head ≥70% throughput — proceeding" ;;
    *) log "PR5 inconclusive (no verdict parsed) — proceeding" ;;
  esac
}
bench_gate

FAILED_ARMS=""
run_arm() {  # <subdir> <rung> <on|off>
  local sub="$1" rung="$2" cc="$3"
  log "ARM ${sub} (rung=$rung, control-construction=$cc, $ROUNDS rounds, continual, K=4)"
  if "$PY" -m unciv_train.run_loop "${COMMON[@]}" --rung "$rung" --control-construction "$cc" \
        --out "$ROOT/$sub"; then
    log "ARM ${sub} completed all $ROUNDS rounds"
  else
    log "ARM ${sub} did NOT finish this invocation (rc=$?) — re-run ./run_v7.sh to --resume"
    FAILED_ARMS="$FAILED_ARMS $sub"
  fi
}

# --- The 4 arms (construction is the ONLY axis within each rung pair) ---
run_arm structured-small-off  small  off || true
run_arm structured-small-on   small  on  || true
run_arm structured-medium-off medium off || true
run_arm structured-medium-on  medium on  || true

# --- Per-arm 200-game ceiling eval (seed 4242424) + z-tests vs the fixed blind baseline ---
for ARM in structured-small-off structured-small-on structured-medium-off structured-medium-on; do
  case " $FAILED_ARMS " in *" $ARM "*) log "SKIP ceiling eval for $ARM (arm incomplete)"; continue ;; esac
  log "CEILING EVAL + z-tests: $ARM"
  CC=off; case "$ARM" in *-on) CC=on ;; esac   # ON arms MUST measure the ceiling WITH construction control
  "$PY" -m unciv_train.analyze_v5 --root "$ROOT/$ARM" --label "$ARM" \
      --ceiling-games 200 --turn-cap 250 --threads "$TH" --eval-seed 4242424 \
      --control-construction "$CC" \
    || log "ANALYZE_V5 $ARM FAILED"
  [ -f "$ROOT/$ARM/acceptance_v5.json" ] && cp "$ROOT/$ARM/acceptance_v5.json" "$ROOT/$ARM/acceptance_v7.json"
done

# --- v7 ON-vs-OFF + 50%-break-even comparison ---
if [ -z "$FAILED_ARMS" ]; then
  log "ON-vs-OFF + 50%-BREAK-EVEN COMPARISON (analyze_v7)"
  "$PY" -m unciv_train.analyze_v7 --root "$ROOT" || log "ANALYZE_V7 FAILED"
  log "DONE — 4 arms complete; per-arm acceptance in $ROOT/*/acceptance_v7.json; compare in $ROOT/acceptance_v7_compare.json"
else
  log "INCOMPLETE — arms did not finish this invocation:$FAILED_ARMS"
  log "Re-run: THREADS=$TH OUT_ROOT=$ROOT ./run_v7.sh  (--resume continues from the last completed round; replay refills from disk)"
  exit 1
fi
