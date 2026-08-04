#!/bin/bash
# Phase 7: get the sim pretraining gain WITHOUT paying for self-normalization.
#
# PHASE 6 FOUND, on val_unseen:
#   fno_grad        (real norm, from scratch) 86.67   <- and 86.70 on a seed repeat
#   fno_grad_self   (self norm, from scratch) 86.47   <- self-norm costs -0.20 by itself
#   fno_s2r         (self norm, sim-pretrained) 86.55 <- pretraining is worth +0.08 vs control
#   fno_s2r_lo      (self norm, lr 5e-5)       86.48  <- too gentle; stage 2 must move
#
# So sim pretraining IS worth something (+0.08 = 2 sigma against its own control -- which is
# exactly why that control was run), but it was bundled with a normalizer that costs -0.20, and
# the bundle nets out below plain real training.
#
# Self-normalization was chosen for stage 1 because raw sim is 3-19x real's scale, so shared
# weights are meaningless in physical units. But each dataset's OWN GaussianNormalizer already
# maps it to ~unit variance, so `--norm sim` for stage 1 and `--norm real` for stage 2 puts both
# stages on a comparable input distribution WITHOUT giving up the real-norm configuration that
# wins. --init copies by name and shape, so the normalizer change does not block the warm start.
#
# Caveat worth keeping visible: CLAUDE.md argues self-normalization's real benefit is Re
# EXTRAPOLATION, which is invisible locally because both val sets sit inside the training Re
# range while the hidden test set does not. So the -0.20 measured for --norm self is a local
# number that may not transfer. This phase does not settle that; it only asks whether the
# pretraining gain survives without it.
#
# Baseline to beat: fno_grad_s2 = 86.70. Noise floor sigma = 0.040.
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /home/clockj/RealPDE/AI4PDE/Track_1/unet || exit 1
L=/home/clockj/RealPDE/AI4PDE/Track_1/fno_deeponet/phase7_train.log
F="--arch fno --width 24 --modes-y 8 --modes-x 12 --n-layers 4"
B="--epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15"
SIM=/home/clockj/RealPDE/train_sim/train_sim

# Stage 1 in sim's own normalizer.
echo "=== fno_simn_pre (stage 1: train_sim, --norm sim) ===" >> "$L"
python train2.py --out fno_simn_pre.pth --epochs 100 --stride 10 --var-head --alpha-train 0 \
  --loss mse --lam-v 0.3 --lam-t 0.15 $F --lam-grad 1.0 --norm sim --data "$SIM" >> "$L" 2>&1

# Stage 2 in real's normalizer -- the configuration that scores 86.67-86.70 from scratch.
# lr 2e-4 is the from-scratch rate: Phase 6 showed the gentler rates underperform, so the
# question here is whether a warm start helps a FULL-strength stage 2 at all.
echo "=== fno_s2rn (stage 2: real, --norm real, lr 2e-4, init from sim) ===" >> "$L"
python train2.py --out fno_s2rn.pth $B $F --lam-grad 1.0 --lr 2e-4 \
  --init fno_simn_pre.pth >> "$L" 2>&1
echo "=== fno_s2rn_lo (stage 2: real, --norm real, lr 1e-4, init from sim) ===" >> "$L"
python train2.py --out fno_s2rn_lo.pth $B $F --lam-grad 1.0 --lr 1e-4 \
  --init fno_simn_pre.pth >> "$L" 2>&1

# Third seed of the winner. Two runs agree at 86.67/86.70; a third makes the mean solid enough
# to package on, and gives a proper 3-sample noise estimate for THIS recipe rather than
# inheriting sigma = 0.040 from a different one.
echo "=== fno_grad_s3 (third seed of fno_grad) ===" >> "$L"
python train2.py --out fno_grad_s3.pth $B $F --lam-grad 1.0 >> "$L" 2>&1

echo "ALL DONE" >> "$L"
