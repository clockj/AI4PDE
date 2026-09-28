#!/bin/bash
# Phase 4: exploit the divergence-loss result.
#
# fno_div (0.90M, lam_div 0.3) scored 86.47 -- the best in the project -- improving rel-L2,
# TKE and MVPE simultaneously, with an MVPE of 0.0768 that beats the 33M fno_xl. It has not
# yet been combined with either of the other two things that are known to work: the decoupled
# variance net (+0.15) and the field ensemble (+2.0 subscore, the largest effect measured).
#
#   fdiv1/fdiv2  field-only FNO + divergence loss, two seeds -> decoupled pair AND ensemble
#   fno_div_l    divergence loss on the width-48 FNO -> does the gain grow with capacity?
#   lam sweep    0.3 was a first guess; bracket it
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /users/PAS1187/qwu/workspace/realpde/main/Track_1/unet || exit 1
L=/users/PAS1187/qwu/workspace/realpde/main/Track_1/fno_deeponet/phase4_train.log
S="--arch fno --width 24 --modes-y 8 --modes-x 12 --n-layers 4"
BIG="--arch fno --width 48 --modes-y 16 --modes-x 24 --n-layers 4"
V="--var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15"
F="--alpha-train 0 --loss mse"

echo "=== fdiv1 (field-only FNO + div, seed 1) ===" >> "$L"
python train2.py --out fdiv1.pth --epochs 150 --stride 5 $F $S --lam-div 0.3 >> "$L" 2>&1
echo "=== fdiv2 (field-only FNO + div, seed 2) ===" >> "$L"
python train2.py --out fdiv2.pth --epochs 150 --stride 5 $F $S --lam-div 0.3 >> "$L" 2>&1
echo "=== fno_div_l (div on width-48 FNO) ===" >> "$L"
python train2.py --out fno_div_l.pth --epochs 150 --stride 5 $V $BIG --lam-div 0.3 >> "$L" 2>&1
echo "=== fno_div_lo (lam_div 0.1) ===" >> "$L"
python train2.py --out fno_div_lo.pth --epochs 150 --stride 5 $V $S --lam-div 0.1 >> "$L" 2>&1
echo "=== fno_div_hi (lam_div 1.0) ===" >> "$L"
python train2.py --out fno_div_hi.pth --epochs 150 --stride 5 $V $S --lam-div 1.0 >> "$L" 2>&1
echo "=== fno_div_xhi (lam_div 3.0) ===" >> "$L"
python train2.py --out fno_div_xhi.pth --epochs 150 --stride 5 $V $S --lam-div 3.0 >> "$L" 2>&1
echo "=== unet_div_hi (U-Net, lam_div 1.0) ===" >> "$L"
python train2.py --out unet_div_hi.pth --epochs 150 --stride 5 $V --arch unet --width 64 --lam-div 1.0 >> "$L" 2>&1
echo "=== fdiv3 (field-only FNO + div, seed 3) ===" >> "$L"
python train2.py --out fdiv3.pth --epochs 150 --stride 5 $F $S --lam-div 0.3 >> "$L" 2>&1
echo "ALL DONE" >> "$L"
