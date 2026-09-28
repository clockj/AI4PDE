#!/bin/bash
# Phase 5: does the ROTATIONAL half of the velocity gradient beat the dilatational half?
#
# Divergence matching is worth +0.58 on FNO (86.53). But for a vortex-shedding wake the
# coherent structure is vorticity, not divergence -- the whole signal is a train of alternating
# vortices, and divergence is 76% unobservable out-of-plane dw/dz. If the mechanism is
# "constrain the velocity gradient a mode-truncated operator would otherwise get wrong", then
# vorticity should do at least as well and the full tensor should subsume both.
#
# Baseline to beat: fno_div_hi = 86.53 (lam_div 1.0). Noise floor sigma = 0.040.
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /users/PAS1187/qwu/workspace/realpde/main/Track_1/unet || exit 1
L=/users/PAS1187/qwu/workspace/realpde/main/Track_1/fno_deeponet/phase5_train.log
F="--arch fno --width 24 --modes-y 8 --modes-x 12 --n-layers 4"
B="--epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15"

echo "=== fno_vort (lam_vort 1.0) ===" >> "$L"
python train2.py --out fno_vort.pth $B $F --lam-vort 1.0 >> "$L" 2>&1
echo "=== fno_grad (lam_grad 1.0, full tensor) ===" >> "$L"
python train2.py --out fno_grad.pth $B $F --lam-grad 1.0 >> "$L" 2>&1
echo "=== fno_dv (div 1.0 + vort 1.0) ===" >> "$L"
python train2.py --out fno_dv.pth $B $F --lam-div 1.0 --lam-vort 1.0 >> "$L" 2>&1
echo "=== fno_grad_hi (lam_grad 3.0) ===" >> "$L"
python train2.py --out fno_grad_hi.pth $B $F --lam-grad 3.0 >> "$L" 2>&1
echo "=== fno_vort_hi (lam_vort 3.0) ===" >> "$L"
python train2.py --out fno_vort_hi.pth $B $F --lam-vort 3.0 >> "$L" 2>&1
echo "=== unet_grad (U-Net, lam_grad 1.0) ===" >> "$L"
python train2.py --out unet_grad.pth $B --arch unet --width 64 --lam-grad 1.0 >> "$L" 2>&1
echo "ALL DONE" >> "$L"
