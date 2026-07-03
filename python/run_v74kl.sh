#!/usr/bin/env bash
# v7.4 KL-LEASH experiment — does a TIGHT clone (120 BC epochs) + KL-to-clone leash make construction
# control SIGNIFICANTLY beat the no-construction baseline? The tight clone alone gave bc−off=+6.6pp (n=4,
# within noise, one drift-to-collapse outlier at s1000). The KL leash anchors construction near the clone to
# kill that drift; run at 8 SEEDS for the statistical power to resolve a ~+7pp effect.
#
#   TH=6 CONC=2 ./run_v74kl.sh          # resumable; re-run to continue
#
# 2 arms × 8 seeds (paired), small/Medium/16-round/rw1/mb0/ent 0.02, + 200-game ceiling:
#   off   : construction OFF                                  (baseline; off_s1000-4000 reused from v74bc)
#   bckl  : construction ON + BC(120ep) clone + KL leash 0.5  (the candidate win)
set -uo pipefail
cd "$(dirname "$0")"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home}"
ROOT="${OUT_ROOT:-../training-runs/v74kl}"; mkdir -p "$ROOT"; ROOT="$(cd "$ROOT" && pwd)"
SEEDS="${SEEDS:-1000 2000 3000 4000 5000 6000 7000 8000}"; TH="${TH:-6}"; CONC="${CONC:-2}"
ENT="${ENT:-0.02}"; BCEP="${BCEP:-120}"; KL="${KL:-0.5}"; BCDATA="$ROOT/bcdata"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

if [ ! -f "$BCDATA/schema.json" ]; then
  log "GEN heuristic BC dataset (40 games Medium, control off) → $BCDATA"
  ./gradlew selfPlay --console=plain -q --args="gen random $BCDATA 40 250 $TH 5555 Medium false" 2>&1 \
    | grep -aE "SELFPLAY_GEN_DONE|Exception" | head -1
fi

run_job() {  # <arm> <seed> <control on|off> <bc 0|1>
  local arm=$1 seed=$2 ctrl=$3 usebc=$4 out="$ROOT/${arm}_s${seed}"
  if [ -f "$out/acceptance_v5.json" ]; then log "SKIP ${arm}_s${seed} (complete)"; return 0; fi
  local bcflag=(); [ "$usebc" = "1" ] && bcflag=(--bc-pretrain-dir "$BCDATA" --bc-epochs "$BCEP" --construction-kl-coef "$KL")
  log "RUN ${arm}_s${seed} (control=$ctrl bc=$usebc kl=$KL)"
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
for s in $SEEDS; do JOBS+=("off|$s|off|0" "bckl|$s|on|1"); done
for job in "${JOBS[@]}"; do
  IFS='|' read -r arm seed ctrl usebc <<< "$job"
  run_job "$arm" "$seed" "$ctrl" "$usebc" &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 15; done
done
wait
log "ALL RUNS DONE — analysis:"
python3 analyze_v73rep.py "$ROOT" "$SEEDS" "off,bckl" "bckl-off"
