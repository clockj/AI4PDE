#!/bin/bash
#SBATCH -J rp_f27x_ref
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:50:00
# Baseline reference reads of the current system (5 held-out twins) at the operating point.
# Reads only; writes nothing into the worktree except via --out paths on scratch.
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
W=$S/wt_x
cd $W/Track_1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
T0=$(date +%s); stamp() { echo "[STAGE $(date +%T) +$(( $(date +%s) - T0 ))s] $*"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
TW="unet/p2_ffno_l8_sim_s1.pth,unet/p2_ffno_l8_sim_s2.pth,unet/p2_ffno_l8_sim_s3.pth,unet/arch_vit_d384.pth,unet/arch_vit_cmb3sim.pth"
MS="unet/p2_ffno_l8_sim_s1.pth unet/p2_ffno_l8_sim_s2.pth unet/p2_ffno_l8_sim_s3.pth unet/arch_vit_d384.pth unet/arch_vit_cmb3sim.pth"
for f in 1.39 1.29; do
  stamp "ens_v9 AoA15-20 f=$f"
  $PY ens_v9.py --sets "$TW" --alpha 0.80 --f $f --aoa 15 20
  stamp "ens_v9 full f=$f"
  $PY ens_v9.py --sets "$TW" --alpha 0.80 --f $f --aoa 0 5 10 15 20
done
# The SHIPPED band construction (sigma-scaled, p 0.45, centre0, fill) fit on twins and reported on val_unseen AoA 15-20:
for f in 1.39 1.29; do
  stamp "fit_band twins-only centre0 p0.45 inflate=$f"
  $PY submit_fno/fit_band.py --ckpt $MS --err-ckpt $MS --alpha 0.80 --band-p 0.45 --centre-alpha 0 --fill-unscored \
      --inflate $f --out $S/pkg/x_work/ref_band_twins_f$f.npz
done
stamp "ALL DONE"
echo "RP_F27X_REF_END"
