#!/bin/bash
# usage: rp_f27a_submit.sh NAME ARCH FPAR VPAR SEED ALLDATA EPOCHS LR [cluster=ascend]
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
NAME=$1 ARCH=$2 FPAR=$3 VPAR=$4 SEED=$5 ALLDATA=$6 EPOCHS=$7 LR=$8 CL=${9:-ascend}
if [ "$CL" = "ascend" ]; then
  DESK=$(squeue -M ascend -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,)
  PART="-M ascend -p nextgen ${DESK:+--exclude=$DESK}"
else
  PART="-M cardinal -p gpu"
fi
env -u SLURM_JOB_ID -u SLURM_MPI_TYPE -u CUDA_VISIBLE_DEVICES sbatch -A PAS1186 $PART --gpus-per-node=1 -c 12 -t 01:30:00 \
  -J rp_f27a_$NAME --chdir=$S/wt_a --output=$S/logs/rp_f27a_${NAME}_%j.log \
  --export=ALL,NAME=$NAME,ARCH=$ARCH,FPAR=$FPAR,VPAR=$VPAR,SEED=$SEED,ALLDATA=$ALLDATA,EPOCHS=$EPOCHS,LR=$LR \
  $S/jobs/rp_f27a_train.sh
