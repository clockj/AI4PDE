#!/bin/bash
# K3 (trick0927): stage-2 "polish" fine-tune of ONE shipped ViT member or held-out ViT twin.
# Stage 1 = REPRODUCE.md 1.2 (members) / 1.3 (twins); stage 2 = the unchanged stage-1 recipe, warm-started
# (--init), OneCycleLR over EPOCHS at peak LR, plus the arm's single change (EXTRA).
# env: ARM NAME PARENT SEED ALLDATA EPOCHS LR VKIND(plain|cmb) EXTRA
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
export REPO=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/wt_3
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
U=/users/PAS1187/qwu/workspace/realpde/main/Track_1/unet
REAL=/users/PAS1187/qwu/RealPDE/train_real/train_real
SIM=/users/PAS1187/qwu/RealPDE/train_sim/train_sim
: "${ARM:?}" "${NAME:?}" "${PARENT:?}" "${SEED:?}" "${ALLDATA:?}" "${EPOCHS:?}" "${LR:?}" "${VKIND:?}"
EXTRA=${EXTRA:-}
O=$K/ckpt/3/$ARM; mkdir -p $O
cd $REPO/Track_1/unet
echo "T_START $(date) ARM=$ARM NAME=$NAME PARENT=$PARENT SEED=$SEED ALLDATA=$ALLDATA EPOCHS=$EPOCHS LR=$LR VKIND=$VKIND EXTRA=[$EXTRA]"
if [ "$VKIND" = "plain" ]; then
  LOSS="--loss mse --lam-f 1.0 --lam-grad 1.0 --lam-v 0.3 --lam-t 0.15"
  if [ "$ALLDATA" = "1" ]; then SPLIT="--all-data"; else SPLIT=""; fi
else
  LOSS="--loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15"
  if [ "$ALLDATA" = "1" ]; then SPLIT="--all-data --sim $SIM --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5"
  else SPLIT="--sim $SIM --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5 --sim-exclude-held"; fi
fi
$PY train2.py --data $REAL --arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 \
    --dropout 0.3 --var-head $LOSS --alpha-train 0.0 \
    --stride 5 --batch 32 --lr $LR --epochs $EPOCHS --wd 1e-5 --seed $SEED $SPLIT \
    $EXTRA --init $U/$PARENT.pth --out $O/$NAME.pth
ls -la $O/$NAME.pth
echo "T_END $(date)"
echo "rp_k273_${ARM}_${NAME} DONE"
