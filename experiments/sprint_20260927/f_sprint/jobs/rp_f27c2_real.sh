#!/bin/bash
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
export REPO=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_c2
cd $REPO/Track_2
for rep in 1 2; do for d in o_cons n_cons n_k0 n_c2 n_base; do
  echo "=== $d full (round $rep)"; $PY stream_eval.py --submission /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/$d --split unseen --repeat 3 --causal-check --save /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/c2se_$d.npz
done; done
for d in o_cons n_k0 n_c2 n_base; do echo "=== $d aoa15-20"; $PY stream_eval.py --submission /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/$d --split unseen --aoa 15 20; done
$PY - <<'PYEOF'
import numpy as np
a = np.load("/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/c2se_o_cons.npz"); b = np.load("/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/c2se_n_cons.npz")
for k in ("pred", "lower", "upper"):
    print("IDENTITY knob-off", k, bool(np.array_equal(a[k], b[k])), float(np.abs(a[k] - b[k]).max()))
PYEOF
echo DONE_REAL
