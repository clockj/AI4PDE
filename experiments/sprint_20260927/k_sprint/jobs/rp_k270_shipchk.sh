#!/bin/bash
#SBATCH -J rp_k270_shipchk
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:25:00
set -uo pipefail
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; W=$K/wt_0a; cd $W
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
echo "host=$(hostname) $(date +%T)"
declare -A Z=([sb05]=$K/sub_candidates/track1_k27sb05_a080.zip [sb03]=$K/sub_candidates/track1_k27sb03_a080.zip [base]=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/sub_candidates/track1_x27_a080.zip)
for t in sb05 sb03 base; do rm -rf /tmp/k0s_$t; mkdir -p /tmp/k0s_$t; unzip -q ${Z[$t]} -d /tmp/k0s_$t; done
for t in base sb05 sb03; do echo "=== DUMP $t $(date +%T)"; $PY Track_1/verify/k0a_dump.py /tmp/k0s_$t $K/dumps/k0a/ship_$t.npz 2>&1 | grep -v Warning | tail -3; done
for t in sb05 sb03; do echo "=== EDGE seq ship $t $(date +%T)"; $PY Track_1/verify/k0a_edge.py /tmp/k0s_$t seq 2>&1 | grep -v "Warning\|call2"; done
for t in sb05 sb03; do for n in 1 13 14 99; do $PY Track_1/verify/k0a_edge.py /tmp/k0s_$t one $n float64 2>&1 | grep "N=" | sed "s/^/$t /"; done; done
echo "=== DONE $(date +%T)"
