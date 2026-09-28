#!/bin/bash
# usage: rp_k273_submit_vit.sh ARM WHICH(tp|tc|mp|mc) CLUSTER EPOCHS LR [EXTRA...]
#   tp = twin plain (arch_vit_d384, seed 1)      tc = twin cmb3sim (arch_vit_cmb3sim, seed 1)
#   mp = member plain (arch_vit_all_s1, seed 1)  mc = member cmb3sim (arch_vit_all_cmb3sim_s3, seed 3)
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
ARM=$1 WH=$2 CL=$3 EPOCHS=$4 LR=$5; shift 5; EXTRA="$*"
case $WH in
  tp) PARENT=arch_vit_d384; SEED=1; ALLDATA=0; VKIND=plain ;;
  tc) PARENT=arch_vit_cmb3sim; SEED=1; ALLDATA=0; VKIND=cmb ;;
  mp) PARENT=arch_vit_all_s1; SEED=1; ALLDATA=1; VKIND=plain ;;
  mc) PARENT=arch_vit_all_cmb3sim_s3; SEED=3; ALLDATA=1; VKIND=cmb ;;
esac
NAME=${PARENT}_${ARM}
if [ "$CL" = "ascend" ]; then
  DESK=$(squeue -M ascend -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,)
  PART="-M ascend -p nextgen ${DESK:+--exclude=$DESK}"
else
  PART="-M cardinal -p gpu"
fi
export ARM NAME PARENT SEED ALLDATA EPOCHS LR VKIND EXTRA
env -u SLURM_JOB_ID -u SLURM_MPI_TYPE -u CUDA_VISIBLE_DEVICES sbatch -A PAS1186 $PART --gpus-per-node=1 -c 12 -t 01:00:00 \
  -J rp_k273_${ARM}_v${WH} --chdir=$K/wt_3 --output=$K/logs/rp_k273_${ARM}_v${WH}_%j.log \
  --export=ALL $K/jobs/rp_k273_polish_vit.sh 2>&1 | tail -1 | sed "s/^/${ARM}_v${WH} $CL: /"
