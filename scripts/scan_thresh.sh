#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/scan_thresh.sh <ROOT> <CH> <MODE> [PLOW] [PHI] [STEPS] [SPLIT]
# Examples:
#   ./scripts/scan_thresh.sh cnn_aspp/data/ndws_out 3 lt 10 50 12 val
#   ./scripts/scan_thresh.sh cnn_aspp/data/ndws_out/val/EVTUNK 3 lt 10 50 12 here

ROOT=${1:-data/micro}   # dataset root or a split folder with tiles inside (possibly nested)
CH=${2:-3}              # channel index
MODE=${3:-lt}           # lt or gt
PLOW=${4:-10}           # low percentile
PHI=${5:-90}            # high percentile
STEPS=${6:-15}          # number of thresholds
SPLIT=${7:-auto}        # val|test|auto|here

# Resolve where to look for tiles
case "$SPLIT" in
  here)
    SEARCH="$ROOT"           # use ROOT as-is
    ;;
  val|test)
    SEARCH="$ROOT/$SPLIT"    # append split
    ;;
  auto)
    if compgen -G "$ROOT/val/*.npz" >/dev/null || compgen -G "$ROOT/val/*.pt" >/dev/null \
       || compgen -G "$ROOT/val/**/*.npz" >/dev/null || compgen -G "$ROOT/val/**/*.pt" >/dev/null; then
      SEARCH="$ROOT/val"
    elif compgen -G "$ROOT/test/*.npz" >/dev/null || compgen -G "$ROOT/test/*.pt" >/dev/null \
       || compgen -G "$ROOT/test/**/*.npz" >/dev/null || compgen -G "$ROOT/test/**/*.pt" >/dev/null; then
      SEARCH="$ROOT/test"
    else
      echo "No tiles found under $ROOT/(val|test)" >&2; exit 1
    fi
    ;;
  *)
    echo "SPLIT must be one of: val|test|auto|here" >&2; exit 1
    ;;
esac

# Compute percentiles on masked pixels for this channel (recurse)
TH=$(python - <<PY
from pathlib import Path
import numpy as np, torch
root=Path("$SEARCH")
files=sorted([*root.rglob("*.npz")]+[*root.rglob("*.pt")])
assert files, f"No tiles in {root}"
vals=[]
for p in files:
    if p.suffix==".npz":
        z=np.load(p, allow_pickle=True)
        x=z["inputs"]; m=z["mask"] if "mask" in z else np.ones_like(z["targets"])
    else:
        d=torch.load(p); x=d["inputs"]; m=d.get("mask", np.ones_like(d["targets"]))
    ch=x[int("$CH")].astype("float32")
    mv=(m>0.5).squeeze().astype(bool)
    if mv.any(): vals.append(ch[mv].ravel())
assert vals, f"No valid masked pixels in {root}"
v=np.concatenate(vals) if len(vals)>1 else vals[0]
pL, pH = np.percentile(v, [float("$PLOW"), float("$PHI")]).tolist()
print(f"{pL},{pH}")
PY
)

PLOWV=$(echo "$TH" | cut -d, -f1)
PHIV=$(echo "$TH" | cut -d, -f2)

echo "# Sweep $SEARCH ch=$CH mode=$MODE from ${PLOW}th=${PLOWV} to ${PHI}th=${PHIV} in $STEPS steps"
python - <<PY
import numpy as np, subprocess
search="$SEARCH"; ch=int("$CH"); mode="$MODE"
lo=float("$PLOWV"); hi=float("$PHIV"); steps=int("$STEPS")
for t in np.linspace(lo, hi, steps):
    subprocess.run([
        "python","-m","cnn_aspp.cli.heuristic_threshold",
        "--root",search, "--split", ".",   # '.' means: use folder as-is
        "--channel",str(ch), "--thresh",str(float(t)), "--mode",mode
    ], check=True)
PY
