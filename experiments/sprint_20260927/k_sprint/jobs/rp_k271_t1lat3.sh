#!/bin/bash
# K1 final A/B: x27 as built (no WARM_TAILS) | ship = x27 + WARM_TAILS | k1b = WARM + PIN 1024 + CAP 56 (bit-identical set) | k1 = k1b + LN_MANUAL
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12} MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}
echo "host=$(hostname) job=$SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) cpus=$OMP_NUM_THREADS"; date
W=${TMPDIR:-/tmp}/k1_$SLURM_JOB_ID; mkdir -p $W
for v in x27 ship k1b k1; do mkdir -p $W/$v; unzip -q -o $K/../final0927/sub_candidates/track1_x27_a080.zip -d $W/$v; done
$PY $K/k1/apply_k1_full.py $W/ship --warm
$PY $K/k1/apply_k1_full.py $W/k1b --warm --pin 1024 --cap 56
$PY $K/k1/apply_k1_full.py $W/k1 --warm --pin 1024 --cap 56 --ln
OUT=$K/logs/k1_t1lat3_${SLURM_CLUSTER_NAME}_$SLURM_JOB_ID.jsonl
r=0
for cfg in "99 f32" "56 f32" "28 f64" "70 f32" "14 f32"; do
  set -- $cfg; r=$((r+1))
  for v in x27 ship k1b k1; do
    echo "=== round $r variant $v cold_N $1 dtype $2"
    $PY $K/k1/k1_child.py $W/$v $v $1 $2 6 $OUT 2>&1 | grep -v "^\[rpde-t1\]" | tail -2 | cut -c1-300
  done
done
date; echo K1_T1LAT3_DONE
