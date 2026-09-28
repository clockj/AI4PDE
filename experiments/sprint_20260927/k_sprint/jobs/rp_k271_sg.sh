#!/bin/bash
#SBATCH -J rp_k271_sg
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:15:00
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; F=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
D=/tmp/e1sg_$SLURM_JOB_ID; mkdir -p $D
unzip -q $F/sub_candidates/track1_x27_a080.zip -d $D/base; unzip -q $K/sub_candidates/track1_k27all_a080.zip -d $D/k27all; unzip -q $K/sub_candidates/track1_k27lat_a080.zip -d $D/k27lat
for v in base k27lat k27all; do /users/PAS1187/qwu/.conda/envs/realpde/bin/python -u $K/e1/small_graph_check.py $D/$v 2>&1 | grep -v "^\[rpde"; done
echo RP_K271_SG_END
