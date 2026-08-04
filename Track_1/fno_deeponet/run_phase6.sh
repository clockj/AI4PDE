#!/bin/bash
# Phase 6: exploit the Phase 5 result, then test the one large untried mechanism.
#
# PHASE 5 FOUND: matching the full velocity-gradient tensor beats matching its divergence.
#   fno_div_hi  (lam_div  1.0) 86.53      <- previous best
#   fno_vort    (lam_vort 1.0) 86.53      <- rotational half ties the dilatational half
#   fno_vort_hi (lam_vort 3.0) 86.57
#   fno_grad    (lam_grad 1.0) 86.68      <- NEW BEST, +0.15 = 3.75 sigma
# and fno_grad improves rel-L2, TKE and MVPE simultaneously (0.0862->0.0839, 0.5082->0.5003,
# 0.0757->0.0726), the same signature that made the divergence term credible. The two invariants
# tie each other and the full tensor beats both, which is what "constrain the velocity gradient
# a mode-truncated spectral operator gets wrong between its retained modes" predicts.
#
# fno_grad_hi (lam_grad 3.0) then ALSO scored 86.68, with slightly better field metrics
# (rel-L2 0.0834, TKE 0.4985). Two independent runs at different lambda reaching the same final
# is effectively a replication of the +0.15, and it says the lam_grad optimum is flat just as
# lam_div's was. It also rules out the obvious follow-up: if the mechanism were "weight high
# wavenumbers harder", lam 3.0 would have beaten lam 1.0 and a Laplacian-matching term (-k^2
# rather than ik) would be next. It did not, so that term does not get built.
#
# Hence the lambda edges are demoted to last here and the untried mechanism leads.
#
# Baseline to beat: fno_grad = 86.68. Noise floor sigma = 0.040, so nothing under +0.08 counts.
# Ordered by expected value: if this is cut short, the good ones have already run.
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate realpde
cd /home/clockj/RealPDE/AI4PDE/Track_1/unet || exit 1
L=/home/clockj/RealPDE/AI4PDE/Track_1/fno_deeponet/phase6_train.log
F="--arch fno --width 24 --modes-y 8 --modes-x 12 --n-layers 4"
B="--epochs 150 --stride 5 --var-head --alpha-train 0 --loss mse --lam-v 0.3 --lam-t 0.15"
SIM=/home/clockj/RealPDE/train_sim/train_sim

# --- sim -> real pretraining: the track's premise, and still untested.
#
# Track 1 IS Sim2Real, yet every model in this project is real-only. train_sim has 100
# trajectories x 1000 frames against real's 82 x 868 and -- the part that matters -- covers
# Re 15225 and 27975, which real does not, while THE HIDDEN TEST SET CONTAINS UNSEEN REYNOLDS
# NUMBERS.
#
# The recorded objection is that sim-only scores 69.0, below free persistence at 79.7. That
# measures zero-shot transfer, not pretraining: the official sim_real_ft checkpoints draw all
# their real skill from stage 2 too, and still beat their own sim_pretrain by 14 points.
#
# Both stages run in SELF-NORMALIZED space (--norm self) per CLAUDE.md: raw sim is 3-19x real's
# scale, so shared weights are meaningless in physical units, while dividing each window by its
# own mean speed collapses sim and real onto one distribution (mean-field rel-L2 between them
# drops from 2.1-17.7 to a condition-independent 0.38-0.47).

# CONTROL FIRST. --norm self changes the input distribution by itself and was only ever measured
# on the U-Net (86.00 vs 86.07, neutral). Without this the comparison would confound
# "pretraining helped" with "self-normalization helped".
echo "=== fno_grad_self (control: lam_grad 1.0, --norm self, real only, from scratch) ===" >> "$L"
python train2.py --out fno_grad_self.pth $B $F --lam-grad 1.0 --norm self >> "$L" 2>&1

# Stage 1: sim. Stride 10 keeps this near 9.6k windows, ~2.8x real, so pretraining does not cost
# more wall clock than the rest of the phase.
echo "=== fno_sim_pre (stage 1: train_sim, --norm self) ===" >> "$L"
python train2.py --out fno_sim_pre.pth --epochs 100 --stride 10 --var-head --alpha-train 0 \
  --loss mse --lam-v 0.3 --lam-t 0.15 $F --lam-grad 1.0 --norm self --data "$SIM" >> "$L" 2>&1

# Stage 2 at two fine-tuning rates. Too high erases stage 1; too low and the real data cannot
# correct sim's different turbulence character (STD_v/MEAN_u is 0.25 sim vs 0.103 real).
echo "=== fno_s2r (stage 2: real, lr 1e-4, init from sim) ===" >> "$L"
python train2.py --out fno_s2r.pth $B $F --lam-grad 1.0 --norm self --lr 1e-4 \
  --init fno_sim_pre.pth >> "$L" 2>&1
echo "=== fno_s2r_lo (stage 2: real, lr 5e-5, init from sim) ===" >> "$L"
python train2.py --out fno_s2r_lo.pth $B $F --lam-grad 1.0 --norm self --lr 5e-5 \
  --init fno_sim_pre.pth >> "$L" 2>&1

# --- lower priority from here: the new best replicated, then the lambda edges.

# Independent repeat of the new best. lam_grad 1.0 and 3.0 already agree, so this is
# confirmation rather than discovery -- but nothing gets packaged on a single seed.
echo "=== fno_grad_s2 (seed repeat of fno_grad) ===" >> "$L"
python train2.py --out fno_grad_s2.pth $B $F --lam-grad 1.0 >> "$L" 2>&1

# Does the divergence term ON TOP still buy anything, or is it already inside the tensor?
# Mathematically it only re-weights the trace, so this tests the loss geometry, not new
# information.
echo "=== fno_gd (lam_grad 1.0 + lam_div 1.0) ===" >> "$L"
python train2.py --out fno_gd.pth $B $F --lam-grad 1.0 --lam-div 1.0 >> "$L" 2>&1

# Edges of the lambda plateau. 1.0 and 3.0 tie; these say how wide the flat region is.
echo "=== fno_grad_03 (lam_grad 0.3) ===" >> "$L"
python train2.py --out fno_grad_03.pth $B $F --lam-grad 0.3 >> "$L" 2>&1
echo "=== fno_grad_10 (lam_grad 10.0) ===" >> "$L"
python train2.py --out fno_grad_10.pth $B $F --lam-grad 10.0 >> "$L" 2>&1

echo "ALL DONE" >> "$L"
