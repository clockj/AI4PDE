#!/bin/bash
# Design B2 CPU stream labs on a compute node (12 cores). env: CMDS = ';'-separated "script tag [set]" entries, run in parallel.
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
cd $S/wt_b2/Track_2/stream_b2
PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
echo "host=$(hostname) job=$SLURM_JOB_ID start $(date)"
IFS=';' read -ra A <<< "$CMDS"
NT=$(( 12 / ${#A[@]} ))
for c in "${A[@]}"; do
  tag=$(echo $c | tr ' ' '_' | tr -d '.')
  B2_THREADS=$NT OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT $PY -u $c > $S/logs/b2job_${tag}.log 2>&1 &
done
wait
echo "end $(date)"
