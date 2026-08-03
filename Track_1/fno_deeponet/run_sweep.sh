#!/bin/bash
# Phase 1: size sweep for FNO and DeepONet on REAL data.
#
# Recipe is identical to the wpA2_mse_var_lo control (86.08) in every respect except the
# architecture -- MSE field loss, variance head at lam_v 0.3 / lam_t 0.15, alpha-train 0,
# stride 5, 150 epochs, batch 32 -- so the comparison against the U-Net is on equal footing.
#
# FNO parameter count is dominated by 2 * width^2 * modes_y * modes_x * n_layers, so the sweep
# moves width and modes together. DeepONet is dominated by the rank p and the branch width.
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /home/clockj/RealPDE/AI4PDE/Track_1/unet
BASE="--epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15"
LOG=/home/clockj/RealPDE/AI4PDE/Track_1/fno_deeponet/sweep_train.log

run () {  # name, flags...
  local name=$1; shift
  echo "### $name  $*" >> "$LOG"
  python train2.py --out "$name.pth" $BASE "$@" 2>&1 | tee -a "$LOG"
  echo "### $name DONE" >> "$LOG"
}

run fno_s  --arch fno --width 24 --modes-y  8 --modes-x 12 --n-layers 4
run fno_m  --arch fno --width 32 --modes-y 12 --modes-x 16 --n-layers 4
run fno_l  --arch fno --width 48 --modes-y 16 --modes-x 24 --n-layers 4
run fno_xl --arch fno --width 64 --modes-y 16 --modes-x 32 --n-layers 4
run don_s  --arch deeponet --width 32 --p 128
run don_m  --arch deeponet --width 64 --p 256
run don_l  --arch deeponet --width 96 --p 512
echo "ALL DONE"
