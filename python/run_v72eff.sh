#!/usr/bin/env bash
# v7.2 PBRS EFFICACY experiment — RESUMABLE. Re-run this script to continue from where it stopped:
#
#   THREADS=12 ./run_v72eff.sh
#
# Two arms (small rung, Medium, 16 rounds, on-policy --replay-window 1, PBRS coef 0.1):
#   off-pbrs : construction OFF + PBRS  (baseline — PBRS on tech/policy only)
#   on-pbrs  : construction ON  + PBRS  (full buildings+units, commit-until-done cadence)
# Then a per-arm 200-game Medium ceiling @ eval-seed 4242424. The question: does shortening the credit
# horizon (PBRS) let the net learn good FULL construction and BEAT the OFF baseline?
# --resume continues each arm from its last completed round (curve.csv + ckpt/opt sidecars); a completed
# arm re-runs 0 training rounds (only its ceiling eval). Stopping mid-round loses only that in-flight round.
set -uo pipefail
cd "$(dirname "$0")"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home}"
PY="${PY:-python3}"; [ -x .venv/bin/python ] && PY=".venv/bin/python"
EFF="${OUT_ROOT:-../training-runs/v72eff}"; mkdir -p "$EFF"; EFF="$(cd "$EFF" && pwd)"
TH="${THREADS:-12}"; ROUNDS="${ROUNDS:-16}"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

arm() {  # <name> <on|off>
  local name=$1 cc=$2
  log "ARM $name (control-construction=$cc, PBRS coef 0.1, $ROUNDS rounds, rw1, continual --resume)"
  "$PY" -m unciv_train.run_loop --variant structured --rung small --map-size Medium --rounds "$ROUNDS" \
    --gen-games 16 --eval-games 80 --turn-cap 250 --threads "$TH" --epochs 8 --lr 1e-3 --gamma 0.99 \
    --lam 0.95 --value-coef 0.5 --entropy-coef 0.01 --clip-eps 0.2 --gen-seed 1000 --eval-seed 999000 \
    --continual --resume --replay-window 1 --micro-batch-steps 256 --control-construction "$cc" \
    --reward-shaping-coef 0.1 --out "$EFF/$name" \
  && "$PY" -m unciv_train.analyze_v5 --root "$EFF/$name" --label "$name" \
    --ceiling-games 200 --turn-cap 250 --threads "$TH" --eval-seed 4242424 --control-construction "$cc"
}

arm off-pbrs off && arm on-pbrs on || { log "INCOMPLETE — re-run ./run_v72eff.sh to continue"; exit 1; }

# Compare: does construction-ON beat construction-OFF, both with PBRS?
"$PY" - "$EFF" <<'PYEOF'
import json, os, sys
from unciv_train.analyze import _two_proportion_z
eff = sys.argv[1]
def ceil(a):
    f = f"{eff}/{a}/acceptance_v5.json"
    c = (json.load(open(f)).get("ceiling") or {}) if os.path.exists(f) else {}
    return c.get("wins"), c.get("games"), c.get("winrate")
off, on = ceil("off-pbrs"), ceil("on-pbrs")
if off[0] is not None and on[0] is not None:
    z, p = _two_proportion_z(on[0], on[1], off[0], off[1])  # one-sided ON > OFF
    print(f"\nPBRS EFFICACY: OFF+PBRS {off[0]}/{off[1]}={off[2]:.3f}  ON+PBRS {on[0]}/{on[1]}={on[2]:.3f}  "
          f"Δ={on[2]-off[2]:+.3f}  z={z:.2f}  p_one_sided={p:.4g}")
    print("VERDICT:", "CONSTRUCTION HELPS (ON>OFF, p<0.05) — PBRS salvages v7" if (on[2] > off[2] and p < 0.05)
          else "NULL/NEGATIVE — construction does not beat OFF even with PBRS")
PYEOF
log "DONE — compare in $EFF/*/acceptance_v5.json"
