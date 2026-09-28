#!/bin/bash
# Design B cross-fit fold model (from SCRATCH: a warm start from a twin has already seen the fold -> contaminated OOF errors).
# env: WT, FOLD (0..4), ARCH (ffno|vit), EPOCHS, SEED
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
export REPO=${WT:?}; export HDF5_USE_FILE_LOCKING=FALSE
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
O=${OUTDIR:-$S/ckpt/b}; mkdir -p $O
EXCL=$($PY -c "import json;d=json.load(open('$REPO/Track_1/probe_b/folds.json'));print(' '.join(d['folds']['${FOLD:?}']+d['val']))")
echo "start $(date) fold $FOLD arch ${ARCH:?} exclude: $EXCL"
cd $REPO/Track_1/unet
if [ "$ARCH" = ffno ]; then
  $PY train2.py --data /users/PAS1187/qwu/RealPDE/train_real/train_real \
    --arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32 --dropout 0.3 --var-head \
    --loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 \
    --alpha-train 0.0 --stride 5 --batch 32 --lr 2e-4 --wd 1e-5 --seed ${SEED:-1} \
    --sim /users/PAS1187/qwu/RealPDE/train_sim/train_sim --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5 \
    --sim-exclude-held --exclude $EXCL \
    --epochs ${EPOCHS:?} --ckpt-every 10 --resume --out $O/xf_ffno_f${FOLD}.pth
else
  $PY train2.py --data /users/PAS1187/qwu/RealPDE/train_real/train_real \
    --arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 --dropout 0.3 --var-head \
    --loss mse --lam-f 1.0 --lam-grad 1.0 --lam-v 0.3 --lam-t 0.15 --alpha-train 0.0 \
    --stride 5 --batch 32 --lr 2e-4 --wd 1e-5 --seed ${SEED:-1} --exclude $EXCL \
    --epochs ${EPOCHS:?} --ckpt-every 10 --resume --out $O/xf_vit_f${FOLD}.pth
fi
echo "end $(date)"
