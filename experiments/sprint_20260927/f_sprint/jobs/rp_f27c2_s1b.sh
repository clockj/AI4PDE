#!/bin/bash
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
export REPO=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_c2
cd $REPO/Track_2
for d in g_cons g_s1b g_cl; do
  echo "=== $d full"; $PY stream_eval.py --submission /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/$d --split unseen --repeat 2 --causal-check --save /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/c2g_$d.npz
  echo "=== $d aoa15-20"; $PY stream_eval.py --submission /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/$d --split unseen --aoa 15 20
done
$PY - <<'PYEOF'
import numpy as np
a = np.load("/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/c2f_o_cons.npz"); b = np.load("/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/c2g_g_cons.npz")
for k in ("pred", "lower", "upper"):
    print("IDENTITY knob-off (final code)", k, bool(np.array_equal(a[k], b[k])), float(np.abs(a[k] - b[k]).max()))
PYEOF
cd $REPO/Track_2/stream_c
$PY interleave_time.py /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/f_base /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/o_cons /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/f_k0 /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/g_s1b /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/g_cl
echo DONE_S1B
