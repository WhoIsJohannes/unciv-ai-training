#!/usr/bin/env bash
# v7.3 REPLICATED efficacy experiment. The self-play baseline is HIGH-VARIANCE (a single seed's ceiling
# swings ~9%↔42% for identical code — the "weak baseline" scare was variance, not a regression). So we
# replicate every arm across SEEDS and compare MEAN ceilings + PAIRED per-seed differences.
#
#   THREADS-per-run TH=6 CONC=2 MB=0 ./run_v73rep.sh          # resumable; re-run to continue
#
# 3 arms × N seeds (paired on seed), small rung, Medium, 16 rounds, rw1, + 200-game ceiling @ 4242424:
#   off        : construction OFF            (baseline)
#   on-shared  : construction ON, coef 0     (legacy shared-adv — the v7 negative, matched conditions)
#   on-pcc     : construction ON, coef 0.5   (per-city credit — the fix)
# Each (arm,seed) run is independent + --resume-able; a finished run (acceptance_v5.json present) is SKIPPED
# (so the two OFF runs already done can be dropped in as off_s2000 / off_s3000 and are reused for free).
set -uo pipefail
cd "$(dirname "$0")"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home}"
ROOT="${OUT_ROOT:-../training-runs/v73rep}"; mkdir -p "$ROOT"; ROOT="$(cd "$ROOT" && pwd)"
MB="${MB:-0}"; SEEDS="${SEEDS:-1000 2000 3000 4000}"; TH="${TH:-6}"; CONC="${CONC:-2}"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

run_job() {  # <arm> <seed> <control on|off> <coef>
  local arm=$1 seed=$2 ctrl=$3 coef=$4 out="$ROOT/${arm}_s${seed}"
  if [ -f "$out/acceptance_v5.json" ]; then log "SKIP ${arm}_s${seed} (already complete)"; return 0; fi
  log "RUN ${arm}_s${seed} (control=$ctrl coef=$coef mb=$MB)"
  python3 -m unciv_train.run_loop --variant structured --map-size Medium --rounds 16 \
    --gen-games 16 --eval-games 80 --turn-cap 250 --threads "$TH" --epochs 8 --lr 1e-3 --gamma 0.99 \
    --lam 0.95 --value-coef 0.5 --entropy-coef 0.01 --clip-eps 0.2 --gen-seed "$seed" --eval-seed 999000 \
    --continual --resume --replay-window 1 --rung small --micro-batch-steps "$MB" \
    --control-construction "$ctrl" --construction-credit-coef "$coef" --out "$out" \
    > "$out.log" 2>&1 \
  && python3 -m unciv_train.analyze_v5 --root "$out" --label "${arm}_s${seed}" \
    --ceiling-games 200 --turn-cap 250 --threads "$TH" --eval-seed 4242424 --control-construction "$ctrl" \
    >> "$out.log" 2>&1 \
  && log "DONE ${arm}_s${seed}" || log "FAIL ${arm}_s${seed} (see $out.log; re-run to --resume)"
}

JOBS=()
for s in $SEEDS; do
  JOBS+=("off|$s|off|0.0" "on-shared|$s|on|0.0" "on-pcc|$s|on|0.5")
done
for job in "${JOBS[@]}"; do
  IFS='|' read -r arm seed ctrl coef <<< "$job"
  run_job "$arm" "$seed" "$ctrl" "$coef" &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 15; done
done
wait
log "ALL RUNS DONE — analysis:"
python3 analyze_v73rep.py "$ROOT" "$SEEDS"
