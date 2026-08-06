#!/bin/bash
# Phase 13: the last free ensemble member -- an all-data width-64 FNO.
#
# Held-out evidence (the only kind that can size this): adding w64 to a 5-member diverse set
# moved the AoA 15-20 proxy 74.563 -> 74.614, +0.051. Small, but board time is a constant so a
# member costs nothing, and architecture changes have consistently out-earned extra same-width
# seeds (+0.05..0.07 each against +0.01..0.03).
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /home/clockj/RealPDE/AI4PDE/Track_1/unet || exit 1
L=/home/clockj/RealPDE/AI4PDE/Track_1/fno_deeponet/phase13_train.log
echo "=== fno_all_w64 (width 64, all-data) ===" >> "$L"
python train2.py --out fno_all_w64.pth --epochs 150 --stride 5 --var-head --alpha-train 0 \
  --loss mse --lam-v 0.3 --lam-t 0.15 --lam-grad 1.0 --all-data \
  --arch fno --width 64 --modes-y 8 --modes-x 12 --n-layers 4 >> "$L" 2>&1
echo "=== phase 13 done ===" >> "$L"
