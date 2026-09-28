#!/bin/bash
# Design B2 real-path read: the twin T2 package (5 held-out twins, asnp constants) with the TSM/BLK-patched submission.py.
# env: PART = A (bit-identity off==as-copied, D1c30 on/off, causal check) | B (D1c+VARCAL stacks, AoA 15-20, timing); TSM, BN0, BLK
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927; W=$S/wt_b2
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd $W
echo "host=$(hostname) job=$SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date) PART=$PART TSM=$TSM BN0=$BN0 BLK=$BLK"
T=/tmp/rp_f27b2_$SLURM_JOB_ID; mkdir -p $T
mk() {   # mk <dir> <submission.py> <D1C_N0> <VC_N0> <on|off|->
  rm -rf $1; mkdir -p $1; cp $S/pkg/x_work/t2twin_stage/* $1/; cp $2 $1/submission.py
  sed -i "s/^D1C_N0 = [^ ]*/D1C_N0 = $3/; s/^VC_N0 = [^ ]*/VC_N0 = $4/" $1/submission.py
  [ "$5" = "on" ] && sed -i "s/^R1_TSM = [^ ]*/R1_TSM = $TSM/; s/^R1_BLK_N0 = [^ ]*/R1_BLK_N0 = $BN0/; s/^R1_BLK = [^ ]*/R1_BLK = $BLK/" $1/submission.py
  echo "--- $1: $(grep -h '^D1C_N0 = \|^VC_N0 = \|^R1_TSM = \|^R1_BLK_N0 = \|^R1_BLK = \|^R1 = \|^R1_ASYM = \|^ALPHA = ' $1/submission.py | cut -c1-22 | tr '\n' ';')"
}
run() {  # run <name> [extra args]
  local n=$1; shift
  echo "=== $n $*"; $PY Track_2/stream_eval.py --submission $T/$n "$@" 2>&1 | tail -9
}
NEW=$W/Track_2/submit/submission.py; OLD=$S/ckpt/b2/submission_c_asof.py
if [ "$PART" = A ]; then
  mk $T/asof $OLD 30.0 0.0 -;  mk $T/off $NEW 30.0 0.0 off;  mk $T/on $NEW 30.0 0.0 on;  mk $T/vc_on $NEW 30.0 50.0 on
  run asof --save $T/asof.npz; run off --save $T/off.npz; run on --save $T/on.npz
  $PY - <<PYEOF
import numpy as np
a = np.load("$T/asof.npz", allow_pickle=True); b = np.load("$T/off.npz", allow_pickle=True); c = np.load("$T/on.npz", allow_pickle=True)
print("saved keys", a.files)
for k in a.files:
    x, y = a[k], b[k]
    same = x.shape == y.shape and (np.array_equal(x, y) if x.dtype.kind in "US" else np.array_equal(x, y, equal_nan=True))
    print("knobs OFF vs as-copied:", k, x.shape, "IDENTICAL" if same else "DIFFERENT")
f = {k.lower(): k for k in c.files}
lo = next((f[k] for k in ("lower", "lo", "l") if k in f), None); hi = next((f[k] for k in ("upper", "hi", "u") if k in f), None); p = next((f[k] for k in ("pred", "p", "prediction") if k in f), None)
if lo and hi:
    L, U = c[lo], c[hi]
    print("knobs ON bounds: shape", L.shape, "finite", bool(np.isfinite(L).all() and np.isfinite(U).all()), "lower<=upper", bool((L <= U).all()),
          "lower<=pred<=upper", None if p is None else bool(((L <= c[p]) & (c[p] <= U)).all()))
    if p is not None:
        print("knobs ON prediction == knobs OFF prediction (band-only change):", bool(np.array_equal(c[p], b[p])))
PYEOF
  run vc_on --causal-check
else
  mk $T/vc_off $NEW 30.0 50.0 off; mk $T/vc_on $NEW 30.0 50.0 on; mk $T/d10_off $NEW 10.0 50.0 off; mk $T/d10_on $NEW 10.0 50.0 on
  run vc_off; run vc_on; run vc_off --aoa 15 20; run vc_on --aoa 15 20; run d10_off; run d10_on
  echo "=== step time, interleaved, 3 repeats each (shared node: a bracket, not a level)"
  for r in 1 2; do for v in vc_off vc_on; do echo "-- $v"; $PY Track_2/stream_eval.py --submission $T/$v --repeat 3 2>&1 | grep -i -E "ms|step|time" | tail -4; done; done
  run d10_on --causal-check
fi
rm -rf $T
echo "RP_F27B2_REAL_END $(date)"
