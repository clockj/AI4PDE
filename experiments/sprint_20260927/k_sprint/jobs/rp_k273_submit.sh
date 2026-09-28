#!/bin/bash
# usage: rp_k273_submit.sh ARM KIND(t|m) N CLUSTER(cardinal|ascend) EPOCHS LR DROPOUT [EXTRA...]
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
ARM=$1 KIND=$2 N=$3 CL=$4 EPOCHS=$5 LR=$6 DROPOUT=$7; shift 7; EXTRA="$*"
if [ "$KIND" = "t" ]; then PARENT=p2_ffno_l8_sim_s$N; ALLDATA=0; else PARENT=ffno_all_w40l8_sim_s$N; ALLDATA=1; fi
NAME=${PARENT}_${ARM}
if [ "$CL" = "ascend" ]; then
  DESK=$(squeue -M ascend -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,)
  PART="-M ascend -p nextgen ${DESK:+--exclude=$DESK}"
else
  PART="-M cardinal -p gpu"
fi
export ARM NAME PARENT SEED=$N ALLDATA EPOCHS LR DROPOUT EXTRA
env -u SLURM_JOB_ID -u SLURM_MPI_TYPE -u CUDA_VISIBLE_DEVICES sbatch -A PAS1186 $PART --gpus-per-node=1 -c 12 -t 01:00:00 \
  -J rp_k273_${ARM}_${KIND}${N} --chdir=$K/wt_3 --output=$K/logs/rp_k273_${ARM}_${KIND}${N}_%j.log \
  --export=ALL $K/jobs/rp_k273_polish.sh 2>&1 | tail -1 | sed "s/^/${ARM}_${KIND}${N} $CL: /"
