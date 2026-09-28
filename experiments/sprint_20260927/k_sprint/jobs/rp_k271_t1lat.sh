#!/bin/bash
# K1 latency A/B: base vs PAR_MEMBERS vs PIN_OUT vs both, evaluator-like processes, interleaved.
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12} MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}
echo "host=$(hostname) job=$SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) cpus=$OMP_NUM_THREADS"; date
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
W=${TMPDIR:-/tmp}/k1_$SLURM_JOB_ID; mkdir -p $W
for v in base par pin parpin; do mkdir -p $W/$v; unzip -q -o $K/../final0927/sub_candidates/track1_x27_a080.zip -d $W/$v; done
$PY $K/k1/apply_k1_patch.py $W/par --par
$PY $K/k1/apply_k1_patch.py $W/pin --pin
$PY $K/k1/apply_k1_patch.py $W/parpin --par --pin
OUT=$K/logs/k1_t1lat_${SLURM_CLUSTER_NAME}_$SLURM_JOB_ID.jsonl
r=0
for cfg in "56 f32" "14 f32" "99 f64" "28 f32"; do
  set -- $cfg; r=$((r+1))
  for v in base par pin parpin; do
    echo "=== round $r variant $v cold_N $1 dtype $2"; 
    $PY $K/k1/k1_child.py $W/$v $v $1 $2 8 $OUT 2>&1 | grep -v "^\[rpde-t1\] warm" | tail -4
  done
done
date; echo K1_T1LAT_DONE
