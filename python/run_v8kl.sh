#!/usr/bin/env bash
# v8 UNIT-INTENT experiment — does giving the net a per-unit INTENT head (on top of the v7 construction win)
# SIGNIFICANTLY beat NOT controlling units? Mirrors the v7.4 recipe that produced the construction win:
# tight BC clone (120 epochs) + KL-to-clone leash (0.5), 8 seeds, paired off-vs-on, 200-game ceiling.
#
#   TH=6 CONC=2 ./run_v8kl.sh          # resumable; re-run to continue
#
# BASE (both arms): tech+policy + CONSTRUCTION control ON + BC + construction-KL 0.5 (the v7.4 winning arm).
# PAIRED VARIABLE:
#   off : --control-unit-intent off  --unit-kl-coef 0     (v7 base; unit head BC-warmed but inert)
#   on  : --control-unit-intent on   --unit-kl-coef 0.5   (the candidate win)
# small / Medium / 16-round / rw1 / mb0 / ent 0.02, + 200-game ceiling @ eval-seed 4242424.
set -uo pipefail
cd "$(dirname "$0")"
export JAVA_HOME="${JAVA_HOME:-/home/johannes/.jdks/jdk-21.0.11+10}"
export PATH="$JAVA_HOME/bin:$PATH"
# perf: idle OpenMP threads sleep instead of spin-waiting (bit-identical training numerics;
# frees ~2-4 cores for the sim JVMs whenever trainer phases overlap sim phases).
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
# numpy/torch live in the project venv, not system python3 — activate it so `python3` resolves there.
[ -f .venv/bin/activate ] && source .venv/bin/activate
# gradlew lives at the repo root AND gradle must be invoked from there (it locates the build by CWD); this
# script cd's into python/, so the BC gen runs in a `( cd .. && ./gradlew )` subshell. run_loop.py invokes
# gradle itself (cwd=REPO), so only the BC-gen line needs this.
ROOT="${OUT_ROOT:-../training-runs/v8kl}"; mkdir -p "$ROOT"; ROOT="$(cd "$ROOT" && pwd)"
SEEDS="${SEEDS:-1000 2000 3000 4000 5000 6000 7000 8000}"; TH="${TH:-6}"; CONC="${CONC:-2}"
ENT="${ENT:-0.02}"; BCEP="${BCEP:-120}"; CKL="${CKL:-0.5}"; UKL="${UKL:-0.5}"; BCDATA="$ROOT/bcdata"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# Heuristic BC dataset: BOTH controls OFF ⇒ construction_current AND unit_intent_current carry the heuristic
# picks (the supervised BC targets for both heads).
if [ ! -f "$BCDATA/schema.json" ]; then
  log "GEN heuristic BC dataset (40 games Medium, both controls off) → $BCDATA"
  ( cd .. && ./gradlew selfPlay --console=plain -q --args="gen random $BCDATA 40 250 $TH 5555 Medium false false" ) 2>&1 \
    | grep -aE "SELFPLAY_GEN_DONE|Exception" | head -1
  [ -f "$BCDATA/schema.json" ] || { log "FATAL: BC gen produced no schema.json — aborting"; exit 1; }
fi

run_job() {  # <arm> <seed> <unit_ctrl on|off> <unit_kl>
  local arm=$1 seed=$2 uctrl=$3 ukl=$4 out="$ROOT/${arm}_s${seed}"
  if [ -f "$out/acceptance_v5.json" ]; then log "SKIP ${arm}_s${seed} (complete)"; return 0; fi
  log "RUN ${arm}_s${seed} (unit_control=$uctrl unit_kl=$ukl | base: construction on + BC${BCEP} + ckl${CKL})"
  python3 -m unciv_train.run_loop --variant structured --map-size Medium --rounds 16 \
    --gen-games 16 --eval-games 80 --turn-cap 250 --threads "$TH" --epochs 8 --lr 1e-3 --gamma 0.99 \
    --lam 0.95 --value-coef 0.5 --entropy-coef "$ENT" --clip-eps 0.2 --gen-seed "$seed" --eval-seed 999000 \
    --continual --resume --replay-window 1 --rung small --micro-batch-steps 0 \
    --control-construction on --construction-credit-coef 0.0 \
    --bc-pretrain-dir "$BCDATA" --bc-epochs "$BCEP" --construction-kl-coef "$CKL" \
    --control-unit-intent "$uctrl" --unit-kl-coef "$ukl" --out "$out" > "$out.log" 2>&1 \
  && python3 -m unciv_train.analyze_v5 --root "$out" --label "${arm}_s${seed}" \
    --ceiling-games 200 --turn-cap 250 --threads "$TH" --eval-seed 4242424 \
    --control-construction on --control-unit-intent "$uctrl" >> "$out.log" 2>&1 \
  && log "DONE ${arm}_s${seed}" || log "FAIL ${arm}_s${seed} (see $out.log; re-run to --resume)"
}

JOBS=()
for s in $SEEDS; do JOBS+=("off|$s|off|0" "on|$s|on|$UKL"); done
for job in "${JOBS[@]}"; do
  IFS='|' read -r arm seed uctrl ukl <<< "$job"
  run_job "$arm" "$seed" "$uctrl" "$ukl" &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 15; done
done
wait
log "ALL RUNS DONE — analysis:"
python3 analyze_v8.py "$ROOT" "$SEEDS" "off,on" "on-off"
