#!/bin/bash
# Long-context input on sim.
#
# diag_analog.py measures the shedding period at 60-75 frames. The protocol window is 20
# frames -- under a third of a cycle -- so the network cannot resolve the phase or the
# frequency from what it is shown. Phase prediction is what bounds every subscore that is not
# already at a sampling floor, so this is the one lever with a mechanism behind it.
#
# ctx 80 covers roughly one full shedding cycle; ctx 140 covers two. Field-only nets, directly
# comparable to wpS0_field (ctx 20) trained on the identical recipe.
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /home/clockj/RealPDE/AI4PDE/Track_1/unet
SIM=/home/clockj/RealPDE/train_sim/train_sim
BASE="--epochs 150 --stride 8 --alpha-train 0 --loss mse --norm sim --data $SIM"
for n in 80 140; do
  echo "### C$n  wpC${n}_field  (sim, field-only, ctx $n)" >> wp_train.log
  python train2.py --out "wpC${n}_field.pth" $BASE --ctx $n 2>&1 | tee -a wp_train.log
  echo "### wpC${n}_field DONE" >> wp_train.log
done
echo "ALL DONE"
