#!/usr/bin/env bash
# v7.3 PER-CITY CREDIT efficacy experiment — RESUMABLE. Re-run to continue where it stopped:
#
#   THREADS=12 ./run_v73eff.sh
#
# The attribution fix for the v7 construction negative. THREE arms (small rung, Medium, 16 rounds,
# on-policy --replay-window 1, NO PBRS):
#   off        : construction OFF                      (baseline — tech+policy only, ~v5/v6)
#   on-shared  : construction ON, credit-coef 0        (the v7 negative under MATCHED rw1 conditions:
#                                                        construction in the joint PPO ratio, shared adv)
#   on-pcc     : construction ON, credit-coef 0.5      (THE FIX: separate per-city PG term, each city
#                                                        credited by its OWN economy return via A_city)
# Then a per-arm 200-game Medium ceiling @ eval-seed 4242424. The question: does per-city credit
# assignment turn the construction negative around — does on-pcc BEAT off (move in the right direction),
# and does it beat on-shared (proving the per-city credit MECHANISM, not just re-running construction)?
# --resume continues each arm from its last completed round; a completed arm re-runs only its ceiling eval.
set -uo pipefail
cd "$(dirname "$0")"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home}"
PY="${PY:-python3}"; [ -x .venv/bin/python ] && PY=".venv/bin/python"
EFF="${OUT_ROOT:-../training-runs/v73eff}"; mkdir -p "$EFF"; EFF="$(cd "$EFF" && pwd)"
TH="${THREADS:-12}"; ROUNDS="${ROUNDS:-16}"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

arm() {  # <name> <on|off> <credit-coef>
  local name=$1 cc=$2 coef=$3
  log "ARM $name (control-construction=$cc, credit-coef=$coef, $ROUNDS rounds, rw1, continual --resume)"
  "$PY" -m unciv_train.run_loop --variant structured --rung small --map-size Medium --rounds "$ROUNDS" \
    --gen-games 16 --eval-games 80 --turn-cap 250 --threads "$TH" --epochs 8 --lr 1e-3 --gamma 0.99 \
    --lam 0.95 --value-coef 0.5 --entropy-coef 0.01 --clip-eps 0.2 --gen-seed 1000 --eval-seed 999000 \
    --continual --resume --replay-window 1 --micro-batch-steps 256 --control-construction "$cc" \
    --construction-credit-coef "$coef" --out "$EFF/$name" \
  && "$PY" -m unciv_train.analyze_v5 --root "$EFF/$name" --label "$name" \
    --ceiling-games 200 --turn-cap 250 --threads "$TH" --eval-seed 4242424 --control-construction "$cc"
}

arm off       off 0.0 \
  && arm on-shared on 0.0 \
  && arm on-pcc    on 0.5 \
  || { log "INCOMPLETE — re-run ./run_v73eff.sh to continue"; exit 1; }

# Compare: does per-city credit (on-pcc) beat OFF, and beat on-shared?
"$PY" - "$EFF" <<'PYEOF'
import json, os, sys
from unciv_train.analyze import _two_proportion_z
eff = sys.argv[1]
def ceil(a):
    f = f"{eff}/{a}/acceptance_v5.json"
    c = (json.load(open(f)).get("ceiling") or {}) if os.path.exists(f) else {}
    return c.get("wins"), c.get("games"), c.get("winrate")
off, sh, pcc = ceil("off"), ceil("on-shared"), ceil("on-pcc")
def line(name, c): print(f"  {name:10s} {c[0]}/{c[1]}={c[2]:.3f}" if c[0] is not None else f"  {name:10s} (missing)")
print("\nv7.3 PER-CITY CREDIT EFFICACY (Medium, 200-game ceiling @ 4242424):")
for n, c in (("off", off), ("on-shared", sh), ("on-pcc", pcc)): line(n, c)
if off[0] is not None and pcc[0] is not None:
    z, p = _two_proportion_z(pcc[0], pcc[1], off[0], off[1])           # one-sided on-pcc > off
    print(f"\n  on-pcc vs off : Δ={pcc[2]-off[2]:+.3f}  z={z:.2f}  p_one_sided={p:.4g}")
    # 50% break-even for on-pcc
    zb, pb = _two_proportion_z(pcc[0], pcc[1], pcc[1]//2, pcc[1])
    print(f"  on-pcc vs 50% : Δ={pcc[2]-0.5:+.3f}  crosses_break_even={pcc[2] > 0.5}")
if sh[0] is not None and pcc[0] is not None:
    z2, p2 = _two_proportion_z(pcc[0], pcc[1], sh[0], sh[1])           # one-sided on-pcc > on-shared
    print(f"  on-pcc vs on-shared : Δ={pcc[2]-sh[2]:+.3f}  z={z2:.2f}  p_one_sided={p2:.4g}")
if off[0] is not None and pcc[0] is not None:
    beats_off = pcc[2] >= off[2]
    print("\n  VERDICT:", "PER-CITY CREDIT MOVES THE RIGHT WAY (on-pcc >= off) — attribution fix works"
          if beats_off else "STILL NEGATIVE — per-city credit does not lift construction to the OFF baseline")
PYEOF
log "DONE — compare in $EFF/*/acceptance_v5.json"
