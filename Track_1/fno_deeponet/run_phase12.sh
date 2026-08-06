#!/bin/bash
# Phase 12: all-data versions of the DIVERSE architectures, for the shipped ensemble.
#
# Phase 11 asked whether more capacity helps and got a clean no. Individually, on the AoA 15-20
# proxy: width 24 baseline 81.666 / 81.713 / 81.607, width 48 **81.330**, modes (12,20)
# **81.539**. Both bigger models fit the training set far better (w48's field loss 0.044 against
# the baseline's 0.069) and generalize worse. With 74 trajectories and a velocity-gradient loss
# supplying a rich per-pixel signal, extra capacity buys memorization.
#
# But the ENSEMBLE question has a different answer, and it is the one that matters, because
# board time is a constant and members are free. Member sets on the same proxy:
#
#   3x width-24 seeds                               74.418
#   + m1220                                         74.490
#   + w48 + m1220                                   74.563
#   + w48 + m1220 + cp0.25                          74.600
#   + w48 + m1220 + cp0.25 + ag05 + ag15  (8)       74.615
#
# Monotone, +0.197 total, SPS 35.93 -> 36.46. A member that is worse on its own still helps if
# it fails DIFFERENTLY, and an architecture change buys more decorrelation than another seed of
# the same one: adding w48 (individually -0.34) beat adding a fourth width-24 seed.
#
# So the shipped ensemble wants architectural diversity, and every member must be all-data --
# mixing all-data and held-out members would put avoidably weaker models in the average purely
# because of when they happened to be trained.
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /home/clockj/RealPDE/AI4PDE/Track_1/unet || exit 1
L=/home/clockj/RealPDE/AI4PDE/Track_1/fno_deeponet/phase12_train.log
B="--epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15"
G="--lam-grad 1.0 --all-data"

echo "=== fno_all_w48 (width 48, all-data) ===" >> "$L"
python train2.py --out fno_all_w48.pth $B $G --arch fno --width 48 --modes-y 8 --modes-x 12 \
  --n-layers 4 >> "$L" 2>&1
echo "=== all_w48 saved ===" >> "$L"

echo "=== fno_all_m1220 (width 24, modes 12x20, all-data) ===" >> "$L"
python train2.py --out fno_all_m1220.pth $B $G --arch fno --width 24 --modes-y 12 --modes-x 20 \
  --n-layers 4 >> "$L" 2>&1
echo "=== all_m1220 saved ===" >> "$L"

# A fifth width-24 seed costs the same as anything else and the member curve is still rising.
echo "=== fno_all_s5 (width 24, all-data seed 5) ===" >> "$L"
python train2.py --out fno_all_s5.pth $B $G --arch fno --width 24 --modes-y 8 --modes-x 12 \
  --n-layers 4 >> "$L" 2>&1
echo "=== all_s5 saved ===" >> "$L"

echo "=== phase 12 done ===" >> "$L"
