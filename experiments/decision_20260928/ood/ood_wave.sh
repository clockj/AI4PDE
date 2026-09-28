#!/bin/bash
# Submit the full-size reads of one or more regimes, in at most TWO concurrent chains (the instrument's job cap).
#
#   bash ood_wave.sh A20                 # one regime
#   bash ood_wave.sh A20 A10 RE          # several; jobs alternate between two afterany chains
#   OOD_DEPEND=<id> bash ood_wave.sh ... # both chains start after that job
#   OOD_WAVE_DRY=1 bash ood_wave.sh ...  # print what would be submitted
#   OOD_WAVE_LAYOUTS="six" OOD_WAVE_TAILS="<id0> <id1>" OOD_WAVE_NEXT=<n> bash ood_wave.sh A20
#                                        # one layout, appended to existing chains (ood_poll.sh); prints "TAILS id0 id1 NEXT n"
#
# Per regime (refused unless ood_ready.sh says the layout is READY; a layout that is not ready is skipped and named):
#   <r>_mirror  layout mirror: frozen base control prev_input statsmask famw + the 16 c_* constants (OOD_CSET=1)
#               + control / statsmask re-blanked with m1 m2 m3                              band refit, build, report
#               + arm I6 held-out dump (ood_i6dump.py, + identity check of the patched fit_band path)
#               + arm I7 post hoc (ood_i7.py: rules Z1, Z2_k, Z3_m on frozen / control dumps, plain and m1-m3; dumps deleted)
#   <r>_six     layout six:    frozen base control                                         band refit, build, report
#   <r>_vitc2   layout vitc2:  frozen base control                                         band refit, build, report
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 1 ] || { echo "usage: ood_wave.sh REGIME [REGIME ...]"; exit 2; }
declare -a JOBS
for REG in "$@"; do
  RU=${REG^^}; r=${REG,,}
  for lay in ${OOD_WAVE_LAYOUTS:-mirror six vitc2}; do
    if ! bash "$HERE/ood_ready.sh" "$RU" "$lay"; then echo "  -> skipping ${r}_$lay"; continue; fi
    T=00:45:00; [ "$lay" = mirror ] && T=01:45:00
    JOBS+=("${r}_$lay $RU $lay $T")
  done
done
[ ${#JOBS[@]} -gt 0 ] || { echo "nothing ready"; exit 1; }
PREV=("${OOD_DEPEND:-}" "${OOD_DEPEND:-}")
if [ -n "${OOD_WAVE_TAILS:-}" ]; then read -r t0 t1 <<< "$OOD_WAVE_TAILS"; PREV=("${t0/-/}" "${t1/-/}"); fi
i=${OOD_WAVE_NEXT:-0}
for spec in "${JOBS[@]}"; do
  read -r NAME RU lay T <<< "$spec"
  if [ "$lay" = mirror ]; then
    ENV=(OOD_CSET=1 "OOD_REBLANK=m1 m2 m3" "OOD_REBLANK_VARIANTS=control statsmask"
         "OOD_STAGES=band build variants stream i6dump i7 idread report" "OOD_I7_MASKS=m1 m2 m3" OOD_I7_KEEP=0 OOD_I6_IDCHECK=1)
  else
    ENV=("OOD_VARIANTS=frozen base control" OOD_CSET=0 OOD_REBLANK= "OOD_STAGES=band build variants stream idread report")
  fi
  c=$((i % 2)); i=$((i + 1))
  if [ "${OOD_WAVE_DRY:-0}" = 1 ]; then
    echo "DRY chain $c after '${PREV[$c]}': env ${ENV[*]} bash ood_submit.sh $NAME $RU $lay $T"; PREV[$c]="<$NAME>"; continue
  fi
  out=$(env -u OOD_STAGES -u OOD_BAND -u OOD_I7_MASKS -u OOD_I7_KEEP -u OOD_MEMBERS -u OOD_SKIP_EXISTING "${ENV[@]}" OOD_DEPEND="${PREV[$c]}" \
        bash "$HERE/ood_submit.sh" "$NAME" "$RU" "$lay" "$T") || { echo "submit failed: $NAME"; exit 1; }
  echo "$out  (chain $c)"
  PREV[$c]=$(awk '/^SUBMITTED/ {print $2}' <<< "$out")
done
echo "TAILS ${PREV[0]:--} ${PREV[1]:--} NEXT $i"
