#!/bin/bash
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
export REPO=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_c
cd $REPO/Track_2
for rep in 1 2; do for d in d1c d1c_vc50; do
  echo "=== $d full (round $rep)"; $PY stream_eval.py --submission /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c/$d --split unseen --repeat 3 --causal-check --save /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/se_$d.npz
done; done
echo "=== d1c_vc50 aoa15-20"; $PY stream_eval.py --submission /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c/d1c_vc50 --split unseen --aoa 15 20
echo DONE_SMOKE
