#!/bin/bash
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
export REPO=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_c
cd $REPO/Track_2
for d in base d1c d1c_vc stack; do
  echo "=== $d full"; $PY stream_eval.py --submission /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c/$d --split unseen --repeat 3 --causal-check --save /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/se_$d.npz
  echo "=== $d aoa15-20"; $PY stream_eval.py --submission /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c/$d --split unseen --aoa 15 20
done
echo DONE_SMOKE
