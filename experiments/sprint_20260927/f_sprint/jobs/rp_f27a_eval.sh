#!/bin/bash
# Design A evaluation: held-out twins ensembles, paired, one instrument (ens_v9 + compare_v9), ALPHA 0.80.
# env WAVE: "" = wave A (ffvit_*, ffvit0_*), "L" = wave L (ffvitL_*, ffvit0L_*), "S" = the smoke rehearsal.
# submit: sbatch --export=ALL,WAVE=L -A PAS1186 -M ascend -p nextgen --exclude=<desk> --gpus-per-node=1 -c 12 -t 00:40:00 \
#           -J rp_f27a_eval --chdir=$S/wt_a --output=$S/logs/rp_f27a_eval_%j.log $S/jobs/rp_f27a_eval.sh
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
set +e
export REPO=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_a
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
M0=/users/PAS1187/qwu/workspace/realpde/main
U=$M0/Track_1/unet
K=$S/ckpt/a
WAVE=${WAVE:-}
A=$K/ffvit${WAVE}; C=$K/ffvit0${WAVE}
echo "WAVE='$WAVE' arm prefix $A control prefix $C"
cd $REPO/Track_1
for v in val_seen val_unseen; do ln -sf $M0/Track_1/$v.npz $REPO/Track_1/$v.npz; done
BASE="$U/p2_ffno_l8_sim_s1.pth,$U/p2_ffno_l8_sim_s2.pth,$U/p2_ffno_l8_sim_s3.pth,$U/arch_vit_d384.pth,$U/arch_vit_cmb3sim.pth"
# the parents-only ensemble with Y's exact step-0 weighting is what Y must beat: Y at epoch 0 == this by construction
Y_ARM="${A}_t1.pth,${A}_t2.pth,$U/p2_ffno_l8_sim_s3.pth"
Y_CTL="${C}_t1.pth,${C}_t2.pth,$U/p2_ffno_l8_sim_s3.pth"
X_ARM="${A}_t1.pth,${A}_t2.pth,${A}_t3.pth"
X_CTL="${C}_t1.pth,${C}_t2.pth,${C}_t3.pth"
Z_ARM="$BASE,${A}_t1.pth,${A}_t2.pth"
# epoch-0 references (ffvit0Z_* = the untrained merge of the parents): what fine-tuning must beat, same ensemble structure
Y_0="$K/ffvit0Z_t1.pth,$K/ffvit0Z_t2.pth,$U/p2_ffno_l8_sim_s3.pth"
X_0="$K/ffvit0Z_t1.pth,$K/ffvit0Z_t2.pth,$K/ffvit0Z_t3.pth"
SETS=()
for s in "$BASE" "$Y_0" "$Y_ARM" "$Y_CTL" "$X_0" "$X_ARM" "$X_CTL" "$Z_ARM"; do
  ok=1; for f in ${s//,/ }; do [ -f "$f" ] || { echo "MISSING $f -- set skipped"; ok=0; }; done
  [ $ok = 1 ] && SETS+=("$s")
done
echo "=== ens_v9 AoA 15-20 (board proxy), alpha 0.80 ==="
$PY ens_v9.py --alpha 0.80 --aoa 15 20 --sets "${SETS[@]}"
echo "=== ens_v9 FULL split, alpha 0.80 ==="
$PY ens_v9.py --alpha 0.80 --aoa 0 5 10 15 20 --sets "${SETS[@]}"
echo "=== singles (compare_v9), alpha 0.80, inflate 1.39 ==="
CK=(); for f in $U/p2_ffno_l8_sim_s1.pth $U/arch_vit_d384.pth $K/ffvit0Z_t1.pth ${A}_t1.pth ${C}_t1.pth ${A}_t2.pth ${C}_t2.pth ${A}_t3.pth ${C}_t3.pth; do [ -f $f ] && CK+=($f); done
$PY compare_v9.py --alpha 0.80 --inflate 1.39 --aoa 15 20 --ckpt "${CK[@]}"
echo "rp_f27a_eval DONE"
