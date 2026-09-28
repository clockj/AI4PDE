#!/bin/bash
# Wait for phase 8 to finish, then run phase 9. Two guards, because a waiter that degenerates
# into an infinite loop is the failure mode that has actually bitten this project: the loop is
# bounded at 400 iterations (~3.3 h, comfortably past the ~80 min phase 8 needs), and it starts
# phase 9 only if the completion marker is really there rather than falling through on timeout.
set -u
D=/users/PAS1187/qwu/workspace/realpde/main/Track_1/fno_deeponet
for _ in $(seq 1 400); do
  grep -q "phase 8 done" "$D/phase8_train.log" 2>/dev/null && break
  sleep 30
done
if grep -q "phase 8 done" "$D/phase8_train.log" 2>/dev/null; then
  bash "$D/run_phase9.sh"
else
  echo "phase 8 never finished; phase 9 not started" >> "$D/phase9_train.log"
fi
