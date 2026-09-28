#!/bin/bash
set -euo pipefail
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
echo "host=$(hostname) job=$SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"; date
cd /fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/wt_2/Track_1/probe_k2
$PY dump_jitter.py /fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/dumps/k2_jitter_unseen.npz
date; echo K2_JIT_DONE
