#!/bin/bash
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
export REPO=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_c2
cd $REPO/Track_2
for d in h_cons h_s1b; do
  echo "=== $d full"; $PY stream_eval.py --submission /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/$d --split unseen --repeat 2 --causal-check --save /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/c2h_$d.npz
done
$PY - <<'PYEOF'
import numpy as np
for x, y, tag in (("c2f_o_cons", "c2h_h_cons", "knob-off vs original file"), ("c2g_g_s1b", "c2h_h_s1b", "C2 final file vs the verified C2 package")):
    a = np.load("/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/" + x + ".npz"); b = np.load("/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/" + y + ".npz")
    for k in ("pred", "lower", "upper"):
        print("IDENTITY", tag, k, bool(np.array_equal(a[k], b[k])), float(np.abs(a[k] - b[k]).max()))
PYEOF
echo DONE_FIN2
