#!/bin/bash
# Phase 2 (physics loss terms, each architecture against its own control) and
# Phase 3 (field-only FNOs for the decoupled pair and a 2-model ensemble).
#
# Written flat -- no helper function, no loop -- because the earlier structured version
# silently skipped its own body.
#
# lam_spec is scaled per architecture: at init the term reads 1.96 for FNO and 81.3 for
# DeepONet, since a rank-p decoder carries almost no high-k energy. One shared lambda would
# swamp DeepONet's field loss the way lam_v/lam_t swamped the trunk in round 2.
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /users/PAS1187/qwu/workspace/realpde/main/Track_1/unet || exit 1
L=/users/PAS1187/qwu/workspace/realpde/main/Track_1/fno_deeponet/phase2_train.log

echo "=== fno_div ===" >> "$L"
python train2.py --out fno_div.pth --epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15 --arch fno --width 24 --modes-y 8 --modes-x 12 --n-layers 4 --lam-div 0.3 >> "$L" 2>&1
echo "=== fno_spec ===" >> "$L"
python train2.py --out fno_spec.pth --epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15 --arch fno --width 24 --modes-y 8 --modes-x 12 --n-layers 4 --lam-spec 0.1 >> "$L" 2>&1
echo "=== don_div ===" >> "$L"
python train2.py --out don_div.pth --epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15 --arch deeponet --width 64 --p 256 --lam-div 0.3 >> "$L" 2>&1
echo "=== don_spec ===" >> "$L"
python train2.py --out don_spec.pth --epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15 --arch deeponet --width 64 --p 256 --lam-spec 0.003 >> "$L" 2>&1
echo "=== unet_div ===" >> "$L"
python train2.py --out unet_div.pth --epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15 --arch unet --width 64 --lam-div 0.3 >> "$L" 2>&1
echo "=== unet_spec ===" >> "$L"
python train2.py --out unet_spec.pth --epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15 --arch unet --width 64 --lam-spec 0.1 >> "$L" 2>&1

echo "=== fno_field1 (field-only, width 48) ===" >> "$L"
python train2.py --out fno_field1.pth --epochs 150 --stride 5 --alpha-train 0 --loss mse --arch fno --width 48 --modes-y 16 --modes-x 24 --n-layers 4 >> "$L" 2>&1
echo "=== fno_field2 (field-only, width 48, second seed) ===" >> "$L"
python train2.py --out fno_field2.pth --epochs 150 --stride 5 --alpha-train 0 --loss mse --arch fno --width 48 --modes-y 16 --modes-x 24 --n-layers 4 >> "$L" 2>&1

echo "ALL DONE" >> "$L"
