#!/bin/bash
# Is a regime's full-size ensemble complete?  (login node, no GPU)
#
#   bash ood_ready.sh A20 [mirror six vitc2]     # default: all three layouts; exit 0 iff every member of every layout is ready
#
# A member is ready when its checkpoint exists AND either its training job's sentinel exists
# (dec0928/logs/{rp,pz}_dp_<regime>_<member>.done, written by the job's finish()) or it is listed in the regime's
# REUSED.txt (the copied screen checkpoints, which have no sentinel). A .pth without a sentinel is a run still writing.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REG=${1:?REGIME}; shift
LAYOUTS=${*:-mirror six vitc2}
D=${OOD_DEC:-/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928}
R=${REG,,}; RU=${REG^^}
PY=${PY:-/users/PAS1187/qwu/.conda/envs/realpde/bin/python}
OK=0
for lay in $LAYOUTS; do
  MEM=$(cd "$HERE" && OOD_WT=$D/ood/wt_ood $PY -c "import ood_lib,sys; print(' '.join(ood_lib.layout('$RU','$lay')))") || exit 2
  missing=""
  for p in $MEM; do
    b=$(basename "$p" .pth); m=${b#${R}_}
    if [ ! -s "$p" ]; then missing="$missing $b(no .pth)"; continue; fi
    if [ -f "$D/logs/rp_dp_${R}_$m.done" ] || [ -f "$D/logs/pz_dp_${R}_$m.done" ]; then continue; fi
    grep -q "^ *$b.pth " "$D/ckpt/$RU/REUSED.txt" 2>/dev/null && continue
    missing="$missing $b(no sentinel)"
  done
  if [ -z "$missing" ]; then echo "READY     $RU $lay"; else echo "NOT READY $RU $lay:$missing"; OK=1; fi
done
exit $OK
