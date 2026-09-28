#!/bin/bash
#SBATCH -J rp_k271_eqt2
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; F=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12} MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}
echo "host=$(hostname) job=$SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"; date
D=/tmp/e1t2_$SLURM_JOB_ID; mkdir -p $D
unzip -q -o $F/sub_candidates/track2_x27_a080.zip -d $D/base
unzip -q -o $F/sub_candidates/track2_x27_d1c_a080.zip -d $D/base_d1c
for t in ${TAGS:-k27lat_a080 k27lat_d1c_a080}; do unzip -q -o $K/sub_candidates/track2_$t.zip -d $D/$t; done
E=$K/e1/t2edge_$SLURM_JOB_ID.jsonl
for v in base ${TAGS:-k27lat_a080} base_d1c k27lat_d1c_a080; do
  [ -d $D/$v ] || continue
  echo "== stream 1100 $v"; $PY $K/e1/t2_edge.py $D/$v $v 1100 $E 2>&1 | grep -v "^\[rpde" | tail -1 | cut -c1-600
done
date; echo RP_K271_EQT2_END
