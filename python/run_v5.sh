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
        --continual --resume)

# --- ARM A: primary (small rung; micro-batch no-op → apples-to-apples vs v4, only regime differs) ---
log "ARM A: structured small rung, Medium, $ROUNDS rounds, continual"
"$PY" -m unciv_train.run_loop "${COMMON[@]}" --rung small --micro-batch-steps 0 \
    --out "$ROOT/structured-small-Medium" || log "ARM A FAILED (continuing)"

# --- ARM B: medium rung (micro-batched — newly enabled by v5; would OOM in v4) ---
log "ARM B: structured MEDIUM rung, Medium, $ROUNDS rounds, continual, micro-batch=$MB"
"$PY" -m unciv_train.run_loop "${COMMON[@]}" --rung medium --micro-batch-steps "$MB" \
    --out "$ROOT/structured-medium-Medium" || log "ARM B FAILED (continuing)"

# --- Ceiling evals (200 games, seed 4242424) + z-tests vs fixed v4 baselines (47/204, 58/200) ---
for ARM in structured-small-Medium structured-medium-Medium; do
  log "CEILING EVAL + z-tests: $ARM"
  "$PY" -m unciv_train.analyze_v5 --root "$ROOT/$ARM" --label "$ARM" \
      --ceiling-games 200 --turn-cap 250 --threads "$TH" --eval-seed 4242424 \
      || log "ANALYZE $ARM FAILED"
done

# --- bench-onnx throughput gate (>=0.70) on the primary arm's final model (reuses gradle_selfplay) ---
FINAL=$(ls -1 "$ROOT"/structured-small-Medium/policy_round_*.onnx 2>/dev/null | sort -V | tail -1)
if [ -n "$FINAL" ]; then
  FINAL_ABS="$(cd "$(dirname "$FINAL")" && pwd)/$(basename "$FINAL")"
  log "BENCH-ONNX: $FINAL_ABS"
  "$PY" -c "from unciv_train import run_loop as R; print(R.gradle_selfplay(['bench-onnx','$FINAL_ABS','200','Medium','$TH','777000'],3600.0))" \
      2>/dev/null | grep -E 'BENCH\||verdict=' || log "BENCH produced no verdict line"
fi
log "DONE — per-arm results in $ROOT/*/acceptance_v5.json"
