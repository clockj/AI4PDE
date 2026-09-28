#!/bin/bash
# args: pairs of "pkgdir:mode"
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
echo "host=$(hostname) job=$SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
for pm in "$@"; do
  p=${pm%%:*}; m=${pm##*:}; n=$(basename $p)
  echo "=== $n $m $(date)"
  $PY $S/qa/qa_t2_harness.py $S/qa/$p $m $S/qa/out/${n}_$m 2>&1
  echo "=== exit $? $(date)"
done
