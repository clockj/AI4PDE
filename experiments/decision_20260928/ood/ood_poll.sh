#!/bin/bash
# Poll the regime ensembles and submit each layout's run the moment it is READY (login node, in the background).
#
#   nohup bash ood_poll.sh "<tail0> <tail1>" > $OOD_ROOT/wave/poll.log 2>&1 &
#
# Every 5 minutes, for A20, A10, RE and the layouts mirror, six, vitc2 (a regime's mirror first): if ood_ready.sh says READY
# and the layout was not submitted yet, submit it through ood_wave.sh onto one of TWO afterany chains (the instrument's cap
# of two running jobs). The chains start after the given tails (ids of jobs of this instrument still queued; "-" = none).
# State: $OOD_ROOT/wave/<regime>_<layout>.submitted (job id), wave/tails. Stops when all nine are submitted, or after 8 h.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${OOD_ROOT:-/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood}
ST=$ROOT/wave; mkdir -p "$ST"
[ -f "$ST/tails" ] || echo "${1:-- -} 0" > "$ST/tails"
END=$(( $(date +%s) + 8 * 3600 ))
while [ "$(date +%s)" -lt "$END" ]; do
  left=0
  for reg in A20 A10 RE; do
    for lay in mirror six vitc2; do
      f=$ST/${reg}_$lay.submitted
      [ -f "$f" ] && continue
      left=$((left + 1))
      [ "$lay" != mirror ] && [ ! -f "$ST/${reg}_mirror.submitted" ] && continue
      bash "$HERE/ood_ready.sh" "$reg" "$lay" > /dev/null 2>&1 || continue
      read -r t0 t1 nx < "$ST/tails"
      out=$(OOD_WAVE_LAYOUTS=$lay OOD_WAVE_TAILS="$t0 $t1" OOD_WAVE_NEXT=${nx:-0} bash "$HERE/ood_wave.sh" "$reg" 2>&1)
      echo "[$(date +%H:%M)] $reg $lay:"; echo "$out" | sed 's/^/    /'
      id=$(awk '/^SUBMITTED/ {print $2}' <<< "$out")
      if [ -n "$id" ]; then
        echo "$id" > "$f"
        awk '/^TAILS/ {print $2, $3, $5}' <<< "$out" > "$ST/tails"
        left=$((left - 1))
      fi
    done
  done
  [ "$left" -eq 0 ] && { echo "[$(date +%H:%M)] all layouts submitted"; exit 0; }
  sleep 300
done
echo "[$(date +%H:%M)] poller timed out with layouts left"
