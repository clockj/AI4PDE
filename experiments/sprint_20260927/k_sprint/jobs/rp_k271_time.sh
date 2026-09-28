#!/bin/bash
#SBATCH -J rp_k271_time
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:35:00
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; F=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12} MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}
echo "host=$(hostname) job=$SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"; date
squeue -M ascend -w $(hostname -s) -h -o '%i %u %j' 2>/dev/null | head
D=/tmp/e1tm_$SLURM_JOB_ID; mkdir -p $D
T1V="base ${T1TAGS:-k27lat_a080 k27latln_a080}"; T2V="base ${T2TAGS:-k27lat_a080}"
unzip -q -o $F/sub_candidates/track1_x27_a080.zip -d $D/t1_base; unzip -q -o $F/sub_candidates/track2_x27_a080.zip -d $D/t2_base
for t in ${T1TAGS:-k27lat_a080 k27latln_a080}; do unzip -q -o $K/sub_candidates/track1_$t.zip -d $D/t1_$t; done
for t in ${T2TAGS:-k27lat_a080}; do unzip -q -o $K/sub_candidates/track2_$t.zip -d $D/t2_$t; done
O1=$K/e1/time_t1_$SLURM_JOB_ID.jsonl; O2=$K/e1/time_t2_$SLURM_JOB_ID.jsonl
for r in 1 2 3; do for v in $T2V; do
  echo "== T2 round $r $v"; $PY $K/k1/k1_t2child.py $D/t2_$v $v $O2 2>&1 | grep -v "^\[rpde" | tail -1 | cut -c1-300
done; done
for r in 1 2; do for n in 99 56 28 14; do for v in $T1V; do
  echo "== T1 round $r coldN $n $v"; $PY $K/k1/k1_child.py $D/t1_$v $v $n f32 3 $O1 2>&1 | grep -v "^\[rpde" | tail -1 | cut -c1-200
done; done; done
date; echo RP_K271_TIME_END
