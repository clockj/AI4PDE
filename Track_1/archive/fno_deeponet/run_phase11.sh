#!/bin/bash
# Phase 11: spend capacity, because the board says inference is free.
#
# Every round from 1 to 6 priced model size against time_score and chose the 0.90M FNO. The
# 2026-08-05 evening board removed that constraint: two submissions 2.07x apart in local
# latency (2.648 vs 1.282 ms/window) returned time_score 91.697 and 91.726. Board time is
# harness overhead and does not respond to the model. So the size question has never actually
# been asked under the conditions that hold now.
#
# The one piece of evidence is the pre-gradient-loss size ladder: fno_s 82.02, fno_m 81.86,
# fno_l 82.17, fno_xl 82.32 in v9 mean5 -- +0.30 across a 37x parameter range, small but
# monotone at the top. Nobody has ever trained a large FNO WITH the velocity-gradient loss,
# which is the term that moved the small model from 85.95 to 86.68 on the v7 scale and is the
# only change in six rounds to improve rel-L2, TKE and MVPE together. If that term supplies a
# richer training signal, it is exactly the situation where extra capacity should pay more than
# the bare ladder suggests.
#
# Three runs, all HELD OUT so the result is measurable, ~30-45 min each:
#
#   w48    width 48, same modes -- ~4x parameters, the straight capacity axis
#   w64    width 64            -- ~7x, matching the ladder's top end
#   m1220  width 24 but modes (12, 20) -- capacity in the OTHER direction. For a spectral
#          operator, mode truncation and channel width are different bottlenecks: at (8, 12) on
#          a 32x64 grid we keep 8 of 16 ky and 12 of 33 kx, so more than half the resolvable
#          spectrum is discarded before the first layer. If the wake's fine structure is what
#          the model is missing, this helps and width does not.
#
# Baseline to beat: fno_grad three-seed mean 83.217, run-to-run sigma ~0.05, so ~0.15 is the
# bar. Compare on `compare_v9.py --aoa 15 20` as well as the full split: the board resembles
# the high-incidence regime and the full split's easy majority dilutes the signal.
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /users/PAS1187/qwu/workspace/realpde/main/Track_1/unet || exit 1
L=/users/PAS1187/qwu/workspace/realpde/main/Track_1/fno_deeponet/phase11_train.log
B="--epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15"
G="--lam-grad 1.0"

echo "=== fno_w48 (width 48, modes 8x12) ===" >> "$L"
python train2.py --out fno_w48.pth $B $G --arch fno --width 48 --modes-y 8 --modes-x 12 \
  --n-layers 4 >> "$L" 2>&1
echo "=== w48 saved ===" >> "$L"

echo "=== fno_m1220 (width 24, modes 12x20) ===" >> "$L"
python train2.py --out fno_m1220.pth $B $G --arch fno --width 24 --modes-y 12 --modes-x 20 \
  --n-layers 4 >> "$L" 2>&1
echo "=== m1220 saved ===" >> "$L"

echo "=== fno_w64 (width 64, modes 8x12) ===" >> "$L"
python train2.py --out fno_w64.pth $B $G --arch fno --width 64 --modes-y 8 --modes-x 12 \
  --n-layers 4 >> "$L" 2>&1
echo "=== w64 saved ===" >> "$L"

echo "=== phase 11 done ===" >> "$L"
