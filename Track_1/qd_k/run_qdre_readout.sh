#!/usr/bin/env bash
# Fen / qdre_readout -- inference + band readout for the two Re-extrapolation QD twins (qdre_s1, qdre_s2).
# No training. For each seed: re_readout.py builds the re_split windows from raw h5 (relo/rehi/rein held out,
# the 56 training trajectories for the control band's fit), runs THAT model only, and writes small CSVs.
# The pre-registered gate (re_gate.py) is applied locally afterwards, combining both seeds.
set -uo pipefail
shopt -s nullglob
PY=${RPDE_PY:-/users/PAS1187/qwu/envs/mural_osc/bin/python}
REAL=${RPDE_REAL:-/users/PAS1187/qwu/RealPDE/train_real/train_real}
W=${RPDE_QDRE_WORK:-/work}
OUT=${RPDE_QDRE_OUT:-/out}
fail() { echo "FAIL: $*" >&2; exit 1; }
mkdir -p "$OUT" || fail "mkdir"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HDF5_USE_FILE_LOCKING=FALSE
WHEELS=("$W"/wheels/*.whl)
[ "${#WHEELS[@]}" -ge 1 ] || fail "no h5py wheel"
mkdir -p /tmp/pylib || fail "mkdir /tmp/pylib"
for whl in "${WHEELS[@]}"; do "$PY" -m zipfile -e "$whl" /tmp/pylib || fail "unpack $whl"; done
export PYTHONPATH=/tmp/pylib${PYTHONPATH:+:$PYTHONPATH}
echo "== env $(date)"; hostname; nvidia-smi -L
"$PY" -c "import torch,h5py,numpy;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'h5py',h5py.__version__,'np',numpy.__version__)" || fail "env"
sha256sum "$W"/re_readout.py "$W"/centre_band.py "$W"/ckpt/*.pth
for S in 1 2; do
  CK=$W/ckpt/qdre_arm_s$S.pth
  [ -s "$CK" ] || fail "missing $CK"
  echo "== readout seed $S $(date)"
  "$PY" -u "$W/re_readout.py" --ckpt "$CK" --tag "$S" --data "$REAL" --code "$W/code" --out "$OUT" \
      2>&1 | tee "$OUT/readout_s$S.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "readout seed $S"
  [ -s "$OUT/qdre_delta_$S.csv" ] || fail "no CSV for seed $S"
done
echo "=== out ==="; ls -la "$OUT"
echo "== done $(date)"
echo QDRE_READOUT_OK
