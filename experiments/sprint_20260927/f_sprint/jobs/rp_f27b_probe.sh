#!/bin/bash
# env: B_CKPT, MODE (dump|eval|dumpeval)
set -uo pipefail
export HDF5_USE_FILE_LOCKING=FALSE
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
cd $S/wt_b/Track_1/probe_b
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
echo "host=$(hostname) B_CKPT=$B_CKPT MODE=$MODE $(date)"
case $MODE in *dump*) $PY probe3_xfit.py dump || exit 1;; esac
case $MODE in *eval*) $PY probe3_xfit.py eval;; esac
echo "PROBE_END $(date)"
