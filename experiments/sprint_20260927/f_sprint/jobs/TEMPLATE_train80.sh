#!/bin/bash
# 80-min training job template (X2 throughput probe, 09-27). Submit with (Ascend A100, many free GPUs):
#   S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
#   DESK=$(squeue -M ascend -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,)
#   sbatch -M ascend -p nextgen -A PAS1186 --gpus-per-node=1 -c 12 -t 01:25:00 ${DESK:+--exclude="$DESK"} \
#     -J rp_f27<dir>_<tag> --chdir $S/wt_<dir> --output $S/logs/rp_f27<dir>_<tag>_%j.log \
#     --export=ALL,WT=$S/wt_<dir>,TAG=<tag>,DIR=<dir>,EPOCHS=130,SEED=1   # optional INIT=<ckpt> $S/jobs/TEMPLATE_train80.sh
# Cardinal H100 (1.65x faster, but only ~6 free GPUs at 16:20): replace the first line with
#   sbatch -M cardinal -p gpu,debug ...   (no --exclude; debug caps at 1 h -> use -p gpu for >1 h, EPOCHS<=220)
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh     # sets PY, OMP threads, HDF5_USE_FILE_LOCKING=FALSE
export REPO=${WT:?set WT to your worktree}
export HDF5_USE_FILE_LOCKING=FALSE
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
O=$S/ckpt/${DIR:?}; mkdir -p $O
U=/users/PAS1187/qwu/workspace/realpde/main/Track_1/unet
cd $REPO/Track_1/unet
echo "start $(date)"
# twin recipe (REPRODUCE.md §1.5); all-data member: drop --sim-exclude-held, add --all-data
$PY train2.py --data /users/PAS1187/qwu/RealPDE/train_real/train_real \
  --arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32 --dropout 0.3 --var-head \
  --loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 \
  --alpha-train 0.0 --stride 5 --batch 32 --lr 2e-4 --wd 1e-5 --seed ${SEED:-1} \
  --sim /users/PAS1187/qwu/RealPDE/train_sim/train_sim --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5 \
  --sim-exclude-held \
  --epochs ${EPOCHS:?} --ckpt-every 10 --resume ${INIT:+--init $INIT} \
  --out $O/${TAG:?}.pth
echo "end $(date)"
