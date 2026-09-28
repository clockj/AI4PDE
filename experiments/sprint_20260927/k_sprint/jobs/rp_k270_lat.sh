#!/bin/bash
#SBATCH -J rp_k270_lat
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH --exclusive
#SBATCH -t 00:20:00
set -uo pipefail
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; W=$K/wt_0a; cd $W
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T)"
A=${LAT_A:-tw_ship}; B=${LAT_B:-tw_sb05}
for t in $A $B; do rm -rf /tmp/k0l_$t; mkdir -p /tmp/k0l_$t; unzip -q $K/pkg/k0a/track1_$t.zip -d /tmp/k0l_$t; done
$PY Track_1/verify/k0a_lat.py /tmp/k0l_$A /tmp/k0l_$B 20 2>&1 | grep -v Warning
echo "=== DONE $(date +%T)"
