#!/bin/bash
# Phase 8 (v9 scoring): close the generalization gap, because v9 made it the only thing left.
#
# The band is done. sps9.optimal_band is the EXACT per-element optimum, and it beats the
# incumbent by 0.6 on val_unseen; sizing it for board-scale errors is worth another ~0.7. Total
# available from intervals: ~1.3 sps. Meanwhile moving from local error scale to board error
# scale costs 8.2 sps on its own. Under v9's linear score every point of generalization pays
# into sps at 4.4x the old rate, ON TOP of rel-L2, TKE and MVPE. So: generalization.
#
# Two things run here, and they answer different questions.
#
# A. all-data (3 seeds). The shipped models hold out 8 of 82 real trajectories, training on 74.
#    Using all 82 is +10.8% data from the same distribution with the same recipe. This is NOT measurable
#    locally -- val_seen and val_unseen both become training data -- so it is a judgement call
#    rather than an experiment, and the checkpoints are for submission only. The band for them
#    is fitted from the HELD-OUT models' val_unseen error field, which is a genuinely
#    out-of-sample error field of the same recipe; fitting it on an all-data model's own
#    training data would give a band far too narrow, which is exactly what happened to the U-Net
#    submission (local rel-L2 0.0721 contaminated vs 0.1179 on the board, band median 0.0046,
#    board sps 36.94).
#
# B. input-noise augmentation (2 strengths, held out). This one IS measurable. The board's
#    errors run 1.46x local, and one candidate cause is that the model is tuned to the noise
#    level of the trajectories it trained on. Perturbing the input window by a fraction of the
#    target's own fluctuation std tests that directly, on val_unseen, against the three-seed
#    fno_grad baseline whose run-to-run sigma is 0.015 final.
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /home/clockj/RealPDE/AI4PDE/Track_1/unet || exit 1
L=/home/clockj/RealPDE/AI4PDE/Track_1/fno_deeponet/phase8_train.log
F="--arch fno --width 24 --modes-y 8 --modes-x 12 --n-layers 4"
B="--epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15"
G="--lam-grad 1.0"

# --- B first: it is the one that can change what we ship, so measure before committing.
echo "=== fno_ag05 (aug-noise 0.05, held out) ===" >> "$L"
python train2.py --out fno_ag05.pth $B $F $G --aug-noise 0.05 >> "$L" 2>&1
echo "=== fno_ag15 (aug-noise 0.15, held out) ===" >> "$L"
python train2.py --out fno_ag15.pth $B $F $G --aug-noise 0.15 >> "$L" 2>&1

# --- A: the submission checkpoints.
for s in 1 2 3; do
  echo "=== fno_all_s$s (all-data seed $s) ===" >> "$L"
  python train2.py --out "fno_all_s$s.pth" $B $F $G --all-data >> "$L" 2>&1
done

echo "=== phase 8 done ===" >> "$L"
