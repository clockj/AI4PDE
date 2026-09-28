#!/bin/bash
# Phase 9 (v9 scoring): the field loss is weighted for the wrong channel.
#
# --loss mse runs in NORMALIZED units, so a 1-sigma error in v costs the same as a 1-sigma
# error in u. Since sigma_u / sigma_v = 6.1, that weights v about 37x more than u per unit of
# PHYSICAL error. Nothing the leaderboard measures agrees with that split:
#
#   rel-L2   physical norm over both channels; u carries 85.9% of the error energy
#   MVPE     physical norm at the probes
#   SPS      every interval measured against ONE global sigma in physical units
#
# Measured on val_unseen with the shipped 3-seed ensemble: u sits at 12% relative error and
# contributes 21.9 SPS, v at 39% relative error and contributes 26.9. v is already the more
# heavily weighted channel by a wide margin and is still the worse-predicted one, which is the
# signature of a channel near its own noise floor rather than one starved of capacity. If that
# reading is right, taking weight off v costs little and gives u room.
#
# chan_pow interpolates: 0 is the old loss exactly, 1 is physical units. Baseline to beat is
# the fno_grad three-seed mean; run-to-run sigma for this recipe is 0.015 final, so only a
# swing well past that counts.
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /users/PAS1187/qwu/workspace/realpde/main/Track_1/unet || exit 1
L=/users/PAS1187/qwu/workspace/realpde/main/Track_1/fno_deeponet/phase9_train.log
F="--arch fno --width 24 --modes-y 8 --modes-x 12 --n-layers 4"
B="--epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15"
G="--lam-grad 1.0"

for p in 0.25 0.5 1.0; do
  echo "=== fno_cp$p (chan-pow $p, held out) ===" >> "$L"
  python train2.py --out "fno_cp$p.pth" $B $F $G --chan-pow "$p" >> "$L" 2>&1
done

echo "=== phase 9 done ===" >> "$L"
