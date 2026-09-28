#!/bin/bash
# K1 latency A/B round 2: WARM_TAILS x PIN_OUT x fixed chunk 56, evaluator-like processes, interleaved.
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12} MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}
echo "host=$(hostname) job=$SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) cpus=$OMP_NUM_THREADS"; date
W=${TMPDIR:-/tmp}/k1_$SLURM_JOB_ID; mkdir -p $W
for v in base warm pin warmpin pin56 warmpin56; do mkdir -p $W/$v; unzip -q -o $K/../final0927/sub_candidates/track1_x27_a080.zip -d $W/$v; done
for v in pin warmpin pin56 warmpin56; do $PY $K/k1/apply_k1_patch.py $W/$v --pin; done
for v in warm warmpin warmpin56; do sed -i 's/^WARM_TAILS = False/WARM_TAILS = True/' $W/$v/submission.py; grep -n "^WARM_TAILS" $W/$v/submission.py; done
for v in pin56 warmpin56; do sed -i 's/^    batch = _pick_batch(models, dev)/    batch = 56/' $W/$v/submission.py; grep -n "^    batch = 56" $W/$v/submission.py; done
OUT=$K/logs/k1_t1lat2_${SLURM_CLUSTER_NAME}_$SLURM_JOB_ID.jsonl
r=0
for cfg in "56 f32" "99 f32" "14 f32" "99 f64" "28 f32" "70 f32"; do
  set -- $cfg; r=$((r+1))
  for v in base warm pin warmpin pin56 warmpin56; do
    echo "=== round $r variant $v cold_N $1 dtype $2"
    $PY $K/k1/k1_child.py $W/$v $v $1 $2 6 $OUT 2>&1 | grep -v "^\[rpde-t1\]" | tail -2 | cut -c1-400
  done
done
date; echo K1_T1LAT2_DONE
