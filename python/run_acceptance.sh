#!/usr/bin/env bash
# Full v2 acceptance run (reproducible). Resumable: each run uses --resume, so re-launching
# continues from the last completed round (per-round curve.csv + checkpoints). Bounded by D13.
#
#   THREADS=12 OUT_ROOT=../training-runs/v2 ./run_acceptance.sh
#
# Produces, under $OUT_ROOT:
#   {v1-reinforce,blind-critic,rich-critic}-tiny/curve.csv   (AC1 + AC2)
#   {blind-critic,rich-critic}-medium/curve.csv              (AC3)
#   acceptance-report.md + overlay plots                      (analyze.py)
set -uo pipefail
cd "$(dirname "$0")"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home}"
PY=.venv/bin/python
ROOT="${OUT_ROOT:-../training-runs/v2}"
TH="${THREADS:-12}"
TINY_ROUNDS="${TINY_ROUNDS:-12}"
MED_ROUNDS="${MED_ROUNDS:-8}"
mkdir -p "$ROOT"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --- Tiny: the three attributable variants (AC1 curves; AC2 = v1-reinforce vs blind-critic) ---
for V in v1-reinforce blind-critic rich-critic; do
  log "TINY $V ($TINY_ROUNDS rounds)"
  $PY -m unciv_train.run_loop --variant "$V" --map-size Tiny --rounds "$TINY_ROUNDS" \
      --gen-games 24 --eval-games 100 --turn-cap 250 --threads "$TH" \
      --replay-window 1 --out "$ROOT/$V-tiny" --resume || log "TINY $V FAILED (continuing)"
done

# --- Medium: blind vs rich (AC3 ceiling — where positional/board info should matter) ---
for V in blind-critic rich-critic; do
  log "MEDIUM $V ($MED_ROUNDS rounds)"
  $PY -m unciv_train.run_loop --variant "$V" --map-size Medium --rounds "$MED_ROUNDS" \
      --gen-games 16 --eval-games 80 --turn-cap 250 --threads "$TH" \
      --replay-window 1 --out "$ROOT/$V-medium" --resume || log "MEDIUM $V FAILED (continuing)"
done

# --- Acceptance analysis (AC1/AC2/AC3 + plots; runs a final high-N Medium ceiling eval) ---
log "ANALYZE"
$PY -m unciv_train.analyze --root "$ROOT" --ceiling-games 200 --turn-cap 250 --threads "$TH" \
    || log "ANALYZE FAILED"
log "DONE"
