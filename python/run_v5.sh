#!/usr/bin/env bash
# v5 CONTINUAL-TRAINING experiment driver. Resumable: each arm uses --resume, so re-launching
# continues from the last completed round (per-round curve.csv + ckpt/opt sidecars).
#
#   THREADS=12 OUT_ROOT=../training-runs/v5 ./run_v5.sh
#
# Two arms run SEQUENTIALLY (a 14-core box; 12 gradle threads saturate it — parallel arms would
# oversubscribe CPU and slow both):
#   ARM A (primary, AC1/AC2): structured, SMALL rung, Medium, 16 rounds, continual, micro-batch no-op
#                             (small rung ran clean on Medium in v4 → only the REGIME changed).
#   ARM B (AC3):              structured, MEDIUM rung, Medium, 16 rounds, continual, --micro-batch-steps
#                             (micro-batching newly enables this; it OOM'd in v4).
# Then per-arm 200-game Medium ceiling eval at seed 4242424 + z-tests vs the FIXED v4 baselines
# (analyze_v5), and a bench-onnx throughput gate (>=0.70) on the primary arm's final model.
set -uo pipefail
cd "$(dirname "$0")"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home}"
PY="${PY:-python3}"; [ -x .venv/bin/python ] && PY=".venv/bin/python"
ROOT="${OUT_ROOT:-../training-runs/v5}"
TH="${THREADS:-12}"
ROUNDS="${ROUNDS:-16}"
MB="${MICRO_BATCH:-256}"
mkdir -p "$ROOT"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

COMMON=(--variant structured --map-size Medium --rounds "$ROUNDS"
        --gen-games 16 --eval-games 80 --turn-cap 250 --threads "$TH"
        --epochs 8 --lr 1e-3 --gamma 0.99 --lam 0.95 --value-coef 0.5
        --entropy-coef 0.01 --clip-eps 0.2 --gen-seed 1000 --eval-seed 999000
        --continual --resume --replay-window 1)   # v6: pin v5 single-round semantics (K=4 is now the default)

# Track per-arm completion so we don't run a ceiling eval on a half-finished arm and don't report a
# false "DONE" (ship-council FND-0010/0003/0015/0022/0031). --resume makes re-running the driver
# continue an incomplete arm, so a non-zero exit here = "not finished THIS invocation", not "broken
# forever" — but we still gate the eval + final status on it.
FAILED_ARMS=""

run_arm() {  # <subdir> <rung> <micro_batch_steps>
  local sub="$1" rung="$2" mb="$3"
  log "ARM ${sub} (rung=$rung, micro_batch=$mb, $ROUNDS rounds, continual)"
  if "$PY" -m unciv_train.run_loop "${COMMON[@]}" --rung "$rung" --micro-batch-steps "$mb" \
        --out "$ROOT/$sub"; then
    log "ARM ${sub} completed all $ROUNDS rounds"
    return 0
  else
    log "ARM ${sub} did NOT finish this invocation (rc=$?) — re-run ./run_v5.sh to --resume"
    FAILED_ARMS="$FAILED_ARMS $sub"
    return 1
  fi
}

# --- ARM A: primary (small rung; micro-batch no-op → apples-to-apples vs v4, only regime differs) ---
run_arm structured-small-Medium small 0 || true
# --- ARM B: medium rung (micro-batched — newly enabled by v5; would OOM in v4) ---
run_arm structured-medium-Medium medium "$MB" || true

# --- Ceiling evals (200 games, seed 4242424) + z-tests vs fixed v4 baselines (47/204, 58/200) ---
# Skip an arm that did not finish (its final ONNX would reflect an incomplete run).
for ARM in structured-small-Medium structured-medium-Medium; do
  case " $FAILED_ARMS " in
    *" $ARM "*) log "SKIP ceiling eval for $ARM (arm incomplete)"; continue ;;
  esac
  log "CEILING EVAL + z-tests: $ARM"
  "$PY" -m unciv_train.analyze_v5 --root "$ROOT/$ARM" --label "$ARM" \
      --ceiling-games 200 --turn-cap 250 --threads "$TH" --eval-seed 4242424 \
      || log "ANALYZE $ARM FAILED"
done

# --- bench-onnx throughput gate (>=0.70) on the primary arm's final model (reuses gradle_selfplay) ---
case " $FAILED_ARMS " in
  *" structured-small-Medium "*) log "SKIP bench-onnx (primary arm incomplete)" ;;
  *)
    FINAL=$(ls -1 "$ROOT"/structured-small-Medium/policy_round_*.onnx 2>/dev/null | sort -V | tail -1)
    if [ -n "$FINAL" ]; then
      FINAL_ABS="$(cd "$(dirname "$FINAL")" && pwd)/$(basename "$FINAL")"
      log "BENCH-ONNX: $FINAL_ABS"
      "$PY" -c "from unciv_train import run_loop as R; print(R.gradle_selfplay(['bench-onnx','$FINAL_ABS','200','Medium','$TH','777000'],3600.0))" \
          2>/dev/null | grep -E 'BENCH\||verdict=' || log "BENCH produced no verdict line"
    fi ;;
esac

if [ -n "$FAILED_ARMS" ]; then
  log "INCOMPLETE — these arms did not finish this invocation:$FAILED_ARMS"
  log "Re-run: THREADS=$TH OUT_ROOT=$ROOT ./run_v5.sh  (--resume continues from the last completed round)"
  exit 1
fi
log "DONE — both arms complete; per-arm results in $ROOT/*/acceptance_v5.json"
