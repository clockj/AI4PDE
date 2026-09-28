#!/bin/bash
# Design A (final0927): stage-2 fine-tune of the cross-stitched F-FNO x ViT hybrid (`ffvit`) or its
# no-link control (`ffvit0`). Parametrised by env: NAME ARCH FPAR VPAR SEED ALLDATA EPOCHS LR.
# Stage 1 = the parents (REPRODUCE.md 1.4 / 1.5 for F-FNO, 1.1-1.3 for the ViTs); stage 2 = this file.
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
export REPO=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_a
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
U=/users/PAS1187/qwu/workspace/realpde/main/Track_1/unet
REAL=/users/PAS1187/qwu/RealPDE/train_real/train_real
SIM=/users/PAS1187/qwu/RealPDE/train_sim/train_sim
: "${NAME:?}" "${ARCH:?}" "${FPAR:?}" "${VPAR:?}" "${SEED:?}" "${ALLDATA:?}" "${EPOCHS:?}" "${LR:?}"
O=$S/ckpt/a; mkdir -p $O/init
INIT=$O/init/init_${NAME}.pth
CK=$O/${NAME}.pth
cd $REPO/Track_1/unet
echo "T_START $(date) NAME=$NAME ARCH=$ARCH F=$FPAR V=$VPAR SEED=$SEED ALLDATA=$ALLDATA EPOCHS=$EPOCHS LR=$LR"
$PY make_ffvit_init.py $U/$FPAR.pth $U/$VPAR.pth $INIT
if [ "$ALLDATA" = "1" ]; then SPLIT="--all-data"; else SPLIT="--sim-exclude-held"; fi
$PY train2.py --data $REAL --arch $ARCH --width 40 --n-layers 8 --modes-y 16 --modes-x 32 \
    --dropout 0.3 --var-head \
    --loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 \
    --lam-v 0.15 --lam-t 0.15 --alpha-train 0.0 \
    --stride 5 --batch 32 --lr $LR --epochs $EPOCHS --wd 1e-5 --seed $SEED \
    --sim $SIM --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5 $SPLIT \
    --init $INIT --out $CK
ls -la $CK
echo "T_END $(date)"
echo "rp_f27a_${NAME} DONE"
