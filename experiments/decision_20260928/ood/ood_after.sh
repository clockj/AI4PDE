#!/bin/bash
# Follow-up reads once the mirror runs exist (login node; submits through ood_submit.sh, so OOD_DEPEND chains them).
#
#   bash ood_after.sh i6 A20|A10|relo|rehi|ID   # arm I6: band fitted on the OTHER regimes' held-out dumps, X's ensemble, real path
#   bash ood_after.sh i5                         # arm I5: pick leave-one-regime-out, stream the combined variants
#   bash ood_after.sh timing                     # arm T1: interleaved local step time, a20_six vs a20_mirror (control packages)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${OOD_ROOT:-/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood}
R=$ROOT/runs
PY=${PY:-/users/PAS1187/qwu/.conda/envs/realpde/bin/python}
need() { for f in "$@"; do [ -e "$f" ] || { echo "MISSING $f"; exit 1; }; done; }
D20=$R/a20_mirror/i6/heldout.npz; D10=$R/a10_mirror/i6/heldout.npz; DRE=$R/re_mirror/i6/heldout.npz
case "${1:?what}" in
  i6)
    X=${2:?regime}
    case "$X" in
      A20)  need $D10 $DRE; SRC="$D10 $DRE:relo $DRE:rehi"; RUNX=a20_mirror; REG=A20; NAME=a20_mirror_i6 ;;
      A10)  need $D20 $DRE; SRC="$D20 $DRE:relo $DRE:rehi"; RUNX=a10_mirror; REG=A10; NAME=a10_mirror_i6 ;;
      relo) need $D20 $D10 $DRE; SRC="$D20 $D10 $DRE:rehi"; RUNX=re_mirror; REG=RE; NAME=re_mirror_i6lo ;;
      rehi) need $D20 $D10 $DRE; SRC="$D20 $D10 $DRE:relo"; RUNX=re_mirror; REG=RE; NAME=re_mirror_i6hi ;;
      ID)   need $D20 $D10 $DRE; SRC="$D20 $D10 $DRE:relo $DRE:rehi"; RUNX=id_twins; REG=ID; NAME=id_twins_i6 ;;
      *) echo "i6: A20|A10|relo|rehi|ID"; exit 2 ;;
    esac
    if [ "$X" = ID ]; then TB=$R/id_twins/src/Track_1/bands/k27_sigblend_lam0.5.npz; LAY=twins; MEM=""
    else TB=$R/$RUNX/band/k0/band_k0_sigblend_lam0.5.npz; LAY=list; MEM=$(cat "$R/$RUNX/members.list"); fi
    need "$TB"
    env -u OOD_BAND -u OOD_SIGBLEND OOD_MEMBERS="$MEM" OOD_STAGES="i6fit build variants stream report" \
        OOD_VARIANTS="frozen base control" OOD_I6_SOURCES="$SRC" OOD_I6_TARGET_BAND="$TB" \
        bash "$HERE/ood_submit.sh" "$NAME" "$REG" "$LAY" 01:15:00 ;;
  i5)
    need $R/a20_mirror/summary.json $R/a10_mirror/summary.json $R/re_mirror/summary.json
    OOD_WT=$R/a20_mirror/src $PY "$HERE/ood_i5.py" pick A20=$R/a20_mirror:A20 A10=$R/a10_mirror:A10 relo=$R/re_mirror:relo \
        rehi=$R/re_mirror:rehi --out $R/i5_picks.json | tee $R/i5_picks.txt
    SPECS=$(sed -n 's/^OOD_VARIANT_SPECS="\(.*\)"$/\1/p' $R/i5_picks.txt)
    [ -n "$SPECS" ] || { echo "no change picked in any rotation"; exit 0; }
    names=$(for s in $SPECS; do echo -n "${s%%:*} "; done)
    for pair in "a20_mirror A20 i5_lo_a20" "a10_mirror A10 i5_lo_a10" "re_mirror RE i5_lo_relo i5_lo_rehi"; do
      set -- $pair; run=$1; reg=$2; shift 2
      sp=""; for n in "$@"; do for s in $SPECS; do [ "${s%%:*}" = "$n" ] && sp="$sp $s"; done; done
      [ -z "$sp" ] && continue
      out=$(OOD_DEPEND="${PREVJ:-${OOD_DEPEND:-}}" OOD_MEMBERS="$(cat $R/$run/members.list)" OOD_STAGES="variants stream report" \
        OOD_VARIANTS="frozen base control" OOD_SKIP_EXISTING=1 OOD_VARIANT_SPECS="${sp# }" \
        bash "$HERE/ood_submit.sh" "$run" "$reg" list 00:45:00); echo "$out"
      PREVJ=$(awk '/^SUBMITTED/ {print $2}' <<< "$out")      # one serial chain (the instrument's job cap)
    done
    # in distribution: every rotation's picks on the twins of record
    OOD_DEPEND="${PREVJ:-${OOD_DEPEND:-}}" OOD_BAND=record OOD_STAGES="variants stream report" OOD_VARIANTS="frozen base control" \
      OOD_SKIP_EXISTING=1 OOD_VARIANT_SPECS="$SPECS" bash "$HERE/ood_submit.sh" id_twins ID twins 00:45:00 ;;
  timing)
    need $R/a20_mirror/pkg/control/submission.py $R/a20_six/pkg/control/submission.py
    # private copies: a later variants stage of either run rebuilds its pkg/ directory
    TD=$ROOT/timing/a20; rm -rf "$TD"; mkdir -p "$TD"
    cp -rL $R/a20_mirror/pkg/control "$TD/mirror"; cp -rL $R/a20_six/pkg/control "$TD/six"
    J=$ROOT/jobs/ood_timing_a20.sh
    cat > "$J" <<EOF
#!/bin/bash
#SBATCH -J rp_dp_ood_timing
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:45:00
OOD_WT=$R/a20_mirror/src bash $HERE/ood_timing.sh $TD/mirror $TD/six $R/timing_a20_six_vs_mirror.json 4
EOF
    DESK=$(squeue -M ascend -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,)
    env -u SLURM_JOB_ID sbatch --parsable ${DESK:+--exclude="$DESK"} ${OOD_DEPEND:+--dependency=afterany:$OOD_DEPEND} \
        --chdir="$ROOT" --output="$ROOT/logs/%x_%j.log" "$J" ;;
  *) echo "usage: ood_after.sh i6 X | i5 | timing"; exit 2 ;;
esac
