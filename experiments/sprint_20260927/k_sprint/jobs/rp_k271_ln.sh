#!/bin/bash
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"; date
W=${TMPDIR:-/tmp}/k1l_$SLURM_JOB_ID; mkdir -p $W/base; unzip -q -o $K/../final0927/sub_candidates/track1_x27_a080.zip -d $W/base
$PY $K/k1/apply_k1_patch.py $W/base --pin
$PY $K/k1/k1_ln.py $W/base 2>&1 | grep -v "^\[rpde-t1\] warm"
date; echo K1_LN_DONE
