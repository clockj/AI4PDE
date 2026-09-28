#!/bin/bash
# Chunk 4 addition (coordinator 11:00): ensemble-composition layouts, FAMW with its learned family weight, in-distribution
# twin sets. Submitted over NCHAIN afterany chains (default 4; with the 2 jobs already running that is at most 6 at once).
#
#   bash ood_batch5.sh [NCHAIN] [AFTER_JOBID]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
R=/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/runs
N=${1:-4}; AFTER=${2:-}
declare -a TAIL
for i in $(seq 0 $((N - 1))); do TAIL[$i]=""; done
k=0
sub() {   # sub VAR=value ... -- NAME REGIME LAYOUT TIME
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local c=$((k % N)); k=$((k + 1))
  local dep=${TAIL[$c]:-$AFTER}
  out=$(env -u OOD_BAND -u OOD_MEMBERS -u OOD_BANDSTYLE -u OOD_SKIP_EXISTING -u OOD_VARIANT_SPECS "${envs[@]}" OOD_DEPEND="$dep" \
        bash "$HERE/ood_submit.sh" "$@") || { echo "submit failed: $*"; exit 1; }
  echo "$out  (chain $c)"
  TAIL[$c]=$(awk '/^SUBMITTED/ {print $2}' <<< "$out")
}
V4="frozen base control famw"
# 1. new layouts, full runs (band refit, build, streams, in-distribution read of the regime ensemble)
for reg in RE A20 A10; do
  r=${reg,,}
  lays="ffno4 ffno6 six_vitc2"
  [ "$reg" = A20 ] && lays="$lays t3"
  [ "$reg" = RE ] && lays="$lays t3 t4"
  for lay in $lays; do
    T=00:50:00; [ "$reg" = RE ] && T=01:10:00
    sub OOD_STAGES="band build variants stream idread report" OOD_VARIANTS="$V4" OOD_IDREAD_VARIANTS="$V4" OOD_CSET=0 \
      OOD_REBLANK= -- "${r}_$lay" "$reg" "$lay" "$T"
  done
done
# 2. FAMW re-streamed with its weight recorded, on the layouts already read (a20_vitc2 is last: it may still be running)
for run in re_mirror a20_mirror a10_mirror re_six a20_six a10_six re_vitc2 a10_vitc2 a20_vitc2; do
  reg=$(case $run in re_*) echo RE;; a20_*) echo A20;; a10_*) echo A10;; esac)
  sub OOD_MEMBERS="$(cat $R/$run/members.list)" OOD_STAGES="variants stream idread report" OOD_VARIANTS="famw" \
    OOD_IDREAD_VARIANTS="famw" OOD_CSET=0 OOD_REBLANK= -- "$run" "$reg" list 00:40:00
done
# 3. in distribution: the twins of record with the band refitted the shipping way (static on val_seen AoA 15-20, SIGBLEND on
#    val_unseen), and the same for the twins minus their ViTs; and FAMW with its weight on the band-of-record read
sub OOD_BANDSTYLE=ship OOD_STAGES="band build variants stream report" OOD_VARIANTS="$V4" OOD_CSET=0 OOD_REBLANK= \
  -- id_twins_ship ID twins 00:40:00
sub OOD_BANDSTYLE=ship OOD_STAGES="band build variants stream report" OOD_VARIANTS="$V4" OOD_CSET=0 OOD_REBLANK= \
  -- id_ffno3_ship ID twins_ffno 00:40:00
sub OOD_BAND=record OOD_STAGES="variants stream report" OOD_VARIANTS="famw" OOD_CSET=0 OOD_REBLANK= -- id_twins ID twins 00:30:00
echo "TAILS ${TAIL[*]}"
