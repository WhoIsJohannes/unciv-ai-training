#!/usr/bin/env bash
# v6 REPLAY-BUFFER experiment driver. Resumable: each arm uses --resume, so re-launching continues
# from the last completed round (per-round curve.csv + ckpt/opt sidecars + the on-disk replay refill).
#
#   THREADS=12 OUT_ROOT=../training-runs/v6 ./run_v6.sh
#
# FOUR arms run SEQUENTIALLY (a 12/14-core box; parallel arms would oversubscribe CPU). Each is the v5
# continual config (structured, Medium, 16 rounds, gen 16 / eval 80, continual) — the ONLY axis is the
# replay window K (1 = no replay = v5; 4 = recent-window replay):
#   small-K1  : rung small,  --micro-batch-steps 0   --replay-window 1   (EXACT v5 ARM A repro → AC1 no-op, must hit 40.7%)
#   small-K4  : rung small,  --micro-batch-steps 256  --replay-window 4   (D8: chunk the 4× batch; math-identical)
#   medium-K1 : rung medium, --micro-batch-steps 256  --replay-window 1   (reference v5 ARM B 46.6%)
#   medium-K4 : rung medium, --micro-batch-steps 256  --replay-window 4
# Then per-arm 200-game Medium ceiling eval @ eval-seed 4242424 + z-tests vs the fixed blind baseline
# (58/200 = 28.9%) via analyze_v5, and the v6 AC1/AC2 comparison (analyze_v6: K=1 vs K=4 per rung).
set -uo pipefail
cd "$(dirname "$0")"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home}"
PY="${PY:-python3}"; [ -x .venv/bin/python ] && PY=".venv/bin/python"
ROOT="${OUT_ROOT:-../training-runs/v6}"
TH="${THREADS:-12}"
ROUNDS="${ROUNDS:-16}"
mkdir -p "$ROOT"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

COMMON=(--variant structured --map-size Medium --rounds "$ROUNDS"
        --gen-games 16 --eval-games 80 --turn-cap 250 --threads "$TH"
        --epochs 8 --lr 1e-3 --gamma 0.99 --lam 0.95 --value-coef 0.5
        --entropy-coef 0.01 --clip-eps 0.2 --gen-seed 1000 --eval-seed 999000
        --continual --resume)

FAILED_ARMS=""
run_arm() {  # <subdir> <rung> <micro_batch_steps> <replay_window>
  local sub="$1" rung="$2" mb="$3" k="$4"
  log "ARM ${sub} (rung=$rung, micro_batch=$mb, replay_window=$k, $ROUNDS rounds, continual)"
  if "$PY" -m unciv_train.run_loop "${COMMON[@]}" --rung "$rung" --micro-batch-steps "$mb" \
        --replay-window "$k" --out "$ROOT/$sub"; then
    log "ARM ${sub} completed all $ROUNDS rounds"
  else
    log "ARM ${sub} did NOT finish this invocation (rc=$?) — re-run ./run_v6.sh to --resume"
    FAILED_ARMS="$FAILED_ARMS $sub"
  fi
}

# --- The 4 arms (small-rung is the headline 40.7% comparison; medium-rung is the higher-capacity check) ---
run_arm structured-small-K1  small  0   1 || true
run_arm structured-small-K4  small  256 4 || true
run_arm structured-medium-K1 medium 256 1 || true
run_arm structured-medium-K4 medium 256 4 || true

# --- Per-arm 200-game ceiling eval (seed 4242424) + z-tests vs the fixed blind baseline ---
for ARM in structured-small-K1 structured-small-K4 structured-medium-K1 structured-medium-K4; do
  case " $FAILED_ARMS " in
    *" $ARM "*) log "SKIP ceiling eval for $ARM (arm incomplete)"; continue ;;
  esac
  log "CEILING EVAL + z-tests: $ARM"
  "$PY" -m unciv_train.analyze_v5 --root "$ROOT/$ARM" --label "$ARM" \
      --ceiling-games 200 --turn-cap 250 --threads "$TH" --eval-seed 4242424 \
      || log "ANALYZE_V5 $ARM FAILED"
  # mirror the per-arm acceptance json under the v6 name analyze_v6 also accepts (it falls back to v5).
  [ -f "$ROOT/$ARM/acceptance_v5.json" ] && cp "$ROOT/$ARM/acceptance_v5.json" "$ROOT/$ARM/acceptance_v6.json"
done

# --- v6 AC1 (no-op) + AC2 (both framings) comparison: K=1 vs K=4 per rung ---
if [ -z "$FAILED_ARMS" ]; then
  log "AC1/AC2 COMPARISON (analyze_v6)"
  "$PY" -m unciv_train.analyze_v6 --root "$ROOT" || log "ANALYZE_V6 FAILED"
  log "DONE — 4 arms complete; per-arm acceptance in $ROOT/*/acceptance_v6.json; compare in $ROOT/acceptance_v6_compare.json"
else
  log "INCOMPLETE — arms did not finish this invocation:$FAILED_ARMS"
  log "Re-run: THREADS=$TH OUT_ROOT=$ROOT ./run_v6.sh  (--resume continues from the last completed round; the replay window refills from disk)"
  exit 1
fi
