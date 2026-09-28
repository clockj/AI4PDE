#!/bin/bash
#SBATCH -J rp_k270_fast
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH --exclusive
#SBATCH -t 00:25:00
set -uo pipefail
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; W=$K/wt_0a; cd $W
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T) HEAD $(git -C $W log -1 --format=%h)"
for t in tw_ship tw_sb05 tw_sb03; do
  rm -rf /tmp/k0f_$t; mkdir -p /tmp/k0f_$t; unzip -q $K/pkg/k0a/track1_$t.zip -d /tmp/k0f_$t
  git -C $W show 1201ce4:Track_1/submit_fno/_fno.py | sed "s/^D2_DIRECT_PRED = .*/D2_DIRECT_PRED = True/" > /tmp/k0f_$t/_fno.py
  grep -c "_eband_logsig_only" /tmp/k0f_$t/_fno.py
done
for t in tw_sb05 tw_sb03; do echo "=== DUMP fast $t $(date +%T)"; $PY Track_1/verify/k0a_dump.py /tmp/k0f_$t $K/dumps/k0a/${t}f.npz 2>&1 | grep -v Warning | tail -3; done
$PY - <<'PYEOF'
import numpy as np
D = "/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/dumps/k0a/"
for t in ("tw_sb05", "tw_sb03"):
    a, b = np.load(D + t + ".npz"), np.load(D + t + "f.npz")
    print(t, "FAST vs GENERAL (full N=336, real package):", {k: float(np.abs(a[k] - b[k]).max()) for k in ("P", "lo", "hi", "sig", "c0")})
PYEOF
for bs in 112 56; do $PY Track_1/verify/k0a_lat2.py /tmp/k0f_tw_ship /tmp/k0f_tw_sb05 $bs 20 2>&1 | grep -v "Warning\|rpde-t1"; done
echo "=== DONE $(date +%T)"
