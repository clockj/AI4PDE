#!/bin/bash
# Phase 10: five more all-data seeds, because the board says ensembling is now FREE.
#
# The 2026-08-05 evening board settled the timing question that rounds 1-6 kept getting wrong.
# Two submissions, one 2.07x faster than the other locally:
#
#   U-Net       local 2.648 ms/window   board 5.976 ms   time_score 91.697
#   FNO x3      local 1.282 ms/window   board 5.931 ms   time_score 91.726
#
# A 1.366 ms/window local difference produced 0.045 ms on the board. Board time is essentially
# all harness overhead and is model-independent at our scale, so `time_score` is a constant
# ~91.7 whatever we ship. Every previous round priced ensemble members against a time penalty
# that does not exist: at ~0.11 ms/member locally, five more members are ~0.006 ms of board
# time, which is 0.005 of time_score.
#
# The ceiling on this is set by staying under the U-Net's measured point rather than by any
# model of the machine: 8 members is ~1.83 ms/window locally, still well under the 2.648 ms
# that measured 5.976 ms on the board. So 8 members cannot cost more board time than a
# submission we have already run. That is the whole argument -- it is an interpolation between
# two measurements, not an extrapolation.
#
# Gains are small and real: 1->2 members was +0.97 SPS, 2->3 +0.23, so 3->8 is worth maybe +0.3
# SPS plus a little on rel-L2/TKE/MVPE. Free is free.
#
# submission.py already loads fno.pth, fno_2.pth ... fno_8.pth, so nothing else changes.
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /users/PAS1187/qwu/workspace/realpde/main/Track_1/unet || exit 1
L=/users/PAS1187/qwu/workspace/realpde/main/Track_1/fno_deeponet/phase10_train.log
F="--arch fno --width 24 --modes-y 8 --modes-x 12 --n-layers 4"
B="--epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15"
G="--lam-grad 1.0"

for s in 4 5 6 7 8; do
  echo "=== fno_all_s$s (all-data seed $s) ===" >> "$L"
  python train2.py --out "fno_all_s$s.pth" $B $F $G --all-data >> "$L" 2>&1
  # Write the marker per seed, not once at the end: phase 8's single trailing echo was lost
  # when the script was reaped, and three waiters span forever on a marker that never arrived.
  echo "=== seed $s saved ===" >> "$L"
done

echo "=== phase 10 done ===" >> "$L"
