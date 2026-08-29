#!/bin/bash
# Port round 3's decoupled-pair recipe to simulation data.
#   S0  field-only network   (analogue of wpA0_mse_nohead: no variance head at all)
#   S1  variance-only net    (width 32 + history channels, --lam-f 0)
# Paired at inference by eval_pair.py, with the cross-fit sigma calibration.
#
# stride 8 matches wpD_sim (11,132 windows) so the comparison against 84.28 is clean.
# --norm sim is mandatory: sim's scale is ~6x real's and the real constants put the input
# at 7.3 sigma.
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /users/PAS1187/qwu/workspace/realpde/main/Track_1/unet
SIM=/users/PAS1187/qwu/RealPDE/train_sim/train_sim
BASE="--epochs 150 --stride 8 --alpha-train 0 --loss mse --norm sim --data $SIM"

echo "### S0  wpS0_field  (sim, field-only, no variance head)" >> wp_train.log
python train2.py --out wpS0_field.pth $BASE 2>&1 | tee -a wp_train.log
echo "### wpS0_field DONE" >> wp_train.log

echo "### S1  wpS1_varonly  (sim, variance-only, width 32, history)" >> wp_train.log
python train2.py --out wpS1_varonly.pth $BASE --var-head --lam-f 0 --lam-v 1.0 --lam-t 0.5 \
    --width 32 --hist 2>&1 | tee -a wp_train.log
echo "### wpS1_varonly DONE" >> wp_train.log
echo "ALL DONE"
