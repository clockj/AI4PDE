#!/bin/bash
# usage: rp_k273_eval_arm.sh ARM MODE [ENS_ONLY]
#   MODE f  = polished F-FNO (3 twins / 4 members) + the ORIGINAL 2 ViT twins / members
#   MODE fv = polished F-FNO + polished ViTs (arms ctl, hib, hio only)
#   ENS_ONLY=1 (3rd arg) : only the ens_v9 twin reads (needs the twins only, not the members)
# Refuses until every training job it needs has printed its DONE line. Paired reference: TAG=baseline
# (log $K/logs/rp_k273_cr_baseline_*.log), same script, same clone.
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; U=/users/PAS1187/qwu/workspace/realpde/main/Track_1/unet
ARM=$1 MODE=$2 ENS_ONLY=${3:-0}; C=$K/ckpt/3/$ARM
need() { for f in "$@"; do grep -q "DONE" $K/logs/rp_k273_${ARM}_${f}_*.log 2>/dev/null || { echo "NOT READY: rp_k273_${ARM}_$f"; MISSING=1; }; done; }
MISSING=0
# SUBST_M="2" (E3, 17:20): all-data member(s) whose polish job FAILED are replaced by their unpolished parent (band fit only)
MN="1 2 3 4"; for n in ${SUBST_M:-}; do MN=$(echo $MN | tr ' ' '\n' | grep -vx $n | paste -sd' '); done
need t1 t2 t3; [ "$ENS_ONLY" = "1" ] || need $(for n in $MN; do echo -n "m$n "; done)
M="$C/p2_ffno_l8_sim_s1_$ARM.pth $C/p2_ffno_l8_sim_s2_$ARM.pth $C/p2_ffno_l8_sim_s3_$ARM.pth"
A=""; for n in 1 2 3 4; do case " ${SUBST_M:-} " in *" $n "*) A="$A $U/ffno_all_w40l8_sim_s$n.pth";; *) A="$A $C/ffno_all_w40l8_sim_s${n}_$ARM.pth";; esac; done
if [ "$MODE" = "fv" ]; then
  need vtp vtc; [ "$ENS_ONLY" = "1" ] || need vmp vmc
  M="$M $C/arch_vit_d384_$ARM.pth $C/arch_vit_cmb3sim_$ARM.pth"; A="$A $C/arch_vit_all_s1_$ARM.pth $C/arch_vit_all_cmb3sim_s3_$ARM.pth"
else
  M="$M $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"; A="$A $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
fi
[ "$MISSING" = "1" ] && exit 1
TAG=${ARM}_${MODE}; [ "$ENS_ONLY" = "1" ] && TAG=${TAG}_ens
DESK=$(squeue -M ascend -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,)
[ -n "${INFLATE:-}" ] && TAG=${TAG}_i${INFLATE/./}
[ -n "${SUBST_M:-}" ] && TAG=${TAG}_sub${SUBST_M// /}
export CAND_A="$A" CAND_M="$M" TAG W=$K/wt_3 ENS_ONLY INFLATE=${INFLATE:-1.39}
env -u SLURM_JOB_ID -u SLURM_MPI_TYPE -u CUDA_VISIBLE_DEVICES sbatch -M ascend -p nextgen ${DESK:+--exclude=$DESK} \
  -J rp_k273_cr_$TAG --export=ALL --chdir=$K/wt_3 --output=$K/logs/rp_k273_cr_${TAG}_%j.log $K/jobs/rp_k273_candref.sh 2>&1 | tail -1 | sed "s/^/candref $TAG: /"
