#!/bin/bash
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12} MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}
echo "host=$(hostname) job=$SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"; date
W=${TMPDIR:-/tmp}/k1t2_$SLURM_JOB_ID; mkdir -p $W
for v in base arena; do mkdir -p $W/$v; unzip -q -o $K/../final0927/sub_candidates/track2_x27_a080.zip -d $W/$v; done
$PY $K/k1/apply_k1_t2patch.py $W/arena 1024
OUT=$K/logs/k1_t2lat_${SLURM_CLUSTER_NAME}_$SLURM_JOB_ID.jsonl
for r in 1 2 3 4; do for v in base arena; do echo "=== round $r $v"; $PY $K/k1/k1_t2child.py $W/$v $v $OUT 2>&1 | tail -2 | cut -c1-500; done; done
date; echo K1_T2LAT_DONE
