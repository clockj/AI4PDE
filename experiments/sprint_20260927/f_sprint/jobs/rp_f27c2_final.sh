#!/bin/bash
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
export REPO=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_c2
cd $REPO/Track_2
for rep in 1 2; do for d in o_cons f_cons f_s1 f_k0; do
  echo "=== $d full (round $rep)"; $PY stream_eval.py --submission /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/$d --split unseen --repeat 3 --causal-check --save /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/c2f_$d.npz
done; done
for d in f_s1 f_k0 o_cons f_base; do echo "=== $d aoa15-20"; $PY stream_eval.py --submission /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/$d --split unseen --aoa 15 20; done
echo "=== f_base full"; $PY stream_eval.py --submission /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/f_base --split unseen
$PY - <<'PYEOF'
import numpy as np
a = np.load("/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/c2f_o_cons.npz"); b = np.load("/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/c2f_f_cons.npz")
for k in ("pred", "lower", "upper"):
    print("IDENTITY knob-off", k, bool(np.array_equal(a[k], b[k])), float(np.abs(a[k] - b[k]).max()))
PYEOF
echo DONE_REAL
