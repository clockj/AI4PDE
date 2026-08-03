#!/bin/bash
# Variance-reduction runs on sim, queued behind the long-context experiments.
#
# A 3-seed prediction ensemble is worth +2.0 subscore on real (10 sigma) -- the largest single
# effect found in either session -- but costs 3x latency. Two ways to bank it more cheaply:
#   SWA   average the weights of the last 40 epochs; 1x inference. Risk: OneCycleLR has
#         annealed the LR to ~0 by then, so the averaged iterates may be too similar to help.
#   SEED2 a second field net, for a 2-model ensemble at 2x latency (-0.1 final on sim).
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /home/clockj/RealPDE/AI4PDE/Track_1/unet
while pgrep -f 'train2[.]py --out wpC' > /dev/null; do sleep 60; done
SIM=/home/clockj/RealPDE/train_sim/train_sim
BASE="--epochs 150 --stride 8 --alpha-train 0 --loss mse --norm sim --data $SIM"
echo "### SWA  wpS0_swa40  (sim field-only, weights averaged over last 40 epochs)" >> wp_train.log
python train2.py --out wpS0_swa40.pth $BASE --swa 40 2>&1 | tee -a wp_train.log
echo "### wpS0_swa40 DONE" >> wp_train.log
echo "### S0b  wpS0_seed2  (sim field-only, second seed, for ensembling)" >> wp_train.log
python train2.py --out wpS0_seed2.pth $BASE 2>&1 | tee -a wp_train.log
echo "### wpS0_seed2 DONE" >> wp_train.log
echo "ALL DONE"
