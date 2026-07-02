#!/usr/bin/env bash
# v7.4 BEHAVIOR-CLONING experiment — does BC warm-start let construction beat the no-construction baseline
# (and escape the collapse)? The heuristic is only ~random-level (mediocre), so there's plausible room above
# it — IF the net can start non-collapsed. BC clones the heuristic's picks to provide that non-collapsed
# ~heuristic start + a positive-advantage foothold for RL. Replicated (baseline is high-variance).
#
#   TH=6 CONC=2 ./run_v74bc.sh          # resumable; re-run to continue
#
# 3 arms × N seeds (paired), small rung, Medium, 16 rounds, rw1, mb0, entropy 0.02 leash, + 200-game ceiling:
#   off  : construction OFF                         (baseline — net tech/policy only)
#   on   : construction ON, no BC                   (the collapse — expect ~0%)
#   bc   : construction ON + BC warm-start          (the fix — starts ~heuristic, RL climbs)
# Shared heuristic BC dataset (control OFF, schema 8 construction_current) gen'd once. Finished runs skipped.
set -uo pipefail
cd "$(dirname "$0")"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home}"
ROOT="${OUT_ROOT:-../training-runs/v74bc}"; mkdir -p "$ROOT"; ROOT="$(cd "$ROOT" && pwd)"
SEEDS="${SEEDS:-1000 2000 3000 4000}"; TH="${TH:-6}"; CONC="${CONC:-2}"; ENT="${ENT:-0.02}"; BCEP="${BCEP:-20}"
BCDATA="$ROOT/bcdata"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --- gen the shared heuristic BC dataset once (control OFF ⇒ construction_current = heuristic picks) ---
if [ ! -f "$BCDATA/schema.json" ]; then
  log "GEN heuristic BC dataset (40 games Medium, control off) → $BCDATA"
  ./gradlew selfPlay --console=plain -q \
    --args="gen random $BCDATA 40 250 $TH 5555 Medium false" 2>&1 | grep -aE "SELFPLAY_GEN_DONE|Exception" | head -1
fi

run_job() {  # <arm> <seed> <control on|off> <bc 0|1>
  local arm=$1 seed=$2 ctrl=$3 usebc=$4 out="$ROOT/${arm}_s${seed}"
  if [ -f "$out/acceptance_v5.json" ]; then log "SKIP ${arm}_s${seed} (complete)"; return 0; fi
  local bcflag=(); [ "$usebc" = "1" ] && bcflag=(--bc-pretrain-dir "$BCDATA" --bc-epochs "$BCEP")
  log "RUN ${arm}_s${seed} (control=$ctrl bc=$usebc ent=$ENT)"
  python3 -m unciv_train.run_loop --variant structured --map-size Medium --rounds 16 \
    --gen-games 16 --eval-games 80 --turn-cap 250 --threads "$TH" --epochs 8 --lr 1e-3 --gamma 0.99 \
    --lam 0.95 --value-coef 0.5 --entropy-coef "$ENT" --clip-eps 0.2 --gen-seed "$seed" --eval-seed 999000 \
    --continual --resume --replay-window 1 --rung small --micro-batch-steps 0 \
    --control-construction "$ctrl" --construction-credit-coef 0.0 "${bcflag[@]+"${bcflag[@]}"}" --out "$out" > "$out.log" 2>&1 \
  && python3 -m unciv_train.analyze_v5 --root "$out" --label "${arm}_s${seed}" \
    --ceiling-games 200 --turn-cap 250 --threads "$TH" --eval-seed 4242424 --control-construction "$ctrl" >> "$out.log" 2>&1 \
  && log "DONE ${arm}_s${seed}" || log "FAIL ${arm}_s${seed} (see $out.log; re-run to --resume)"
}

JOBS=()
for s in $SEEDS; do
  JOBS+=("off|$s|off|0" "on|$s|on|0" "bc|$s|on|1")
done
for job in "${JOBS[@]}"; do
  IFS='|' read -r arm seed ctrl usebc <<< "$job"
  run_job "$arm" "$seed" "$ctrl" "$usebc" &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 15; done
done
wait
log "ALL RUNS DONE — analysis:"
python3 analyze_v73rep.py "$ROOT" "$SEEDS" "off,on,bc" "bc-off;bc-on;on-off"
