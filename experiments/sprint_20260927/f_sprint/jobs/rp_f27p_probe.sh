#!/bin/bash
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
export REPO=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_p
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
CL=${CL:-x}
O=$S/ckpt/probe/$CL; mkdir -p $O
cd $REPO/Track_1/unet
echo "T_JOBSTART $(date +%s.%N) $(date)"
$PY -c "import torch,sys;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else None);sys.stdout.flush()" || echo "TORCH_IMPORT_FAILED"
echo "T_ENVOK $(date +%s.%N)"
SIM=/users/PAS1187/qwu/RealPDE/train_sim/train_sim
DATA=/users/PAS1187/qwu/RealPDE/train_real/train_real
U=/users/PAS1187/qwu/workspace/realpde/main/Track_1/unet
COMMON="--data $DATA --loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 --alpha-train 0.0 --var-head --stride 5 --lr 2e-4 --wd 1e-5 --seed 1 --sim $SIM --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5 --sim-exclude-held"
FF="--arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32 --dropout 0.3"
VIT="--arch vit --width 384 --n-layers 8 --dropout 0.3"
run() { local tag=$1; shift
  echo "ARM_START $tag $(date +%s.%N)"
  $PY train2.py "$@" 2>&1 | while IFS= read -r l; do echo "[$tag $(date +%s.%N)] $l"; done
  echo "ARM_END $tag $(date +%s.%N)"
}
run a_b32   $FF $COMMON --batch 32 --epochs 3 --out $O/a_b32.pth
run b_init  $FF $COMMON --batch 32 --epochs 3 --init $U/p2_ffno_l8_sim_s1.pth --out $O/b_init.pth
run d_b64   $FF $COMMON --batch 64 --epochs 3 --out $O/d_b64.pth
run d_b128  $FF $COMMON --batch 128 --epochs 3 --out $O/d_b128.pth
run c_vit   $VIT $COMMON --batch 32 --epochs 2 --out $O/c_vit.pth
echo "ALL_DONE $(date +%s.%N)"
