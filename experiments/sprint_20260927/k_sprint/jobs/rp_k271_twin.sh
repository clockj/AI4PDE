#!/bin/bash
#SBATCH -J rp_k271_twin
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:20:00
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
D=/tmp/e1tw_$SLURM_JOB_ID; mkdir -p $D
for t in k27twin_a080 k27twinln_a080; do unzip -q -o $K/sub_candidates/track1_$t.zip -d $D/$t
  $PY $K/e1/twin_score.py $D/$t $t $K/e1/tw_$t.npz 2>&1 | grep -v "^\[rpde"; done
$PY - <<PYEOF
import numpy as np
a=np.load("$K/e1/tw_k27twin_a080.npz"); b=np.load("$K/e1/tw_k27twinln_a080.npz")
for k in ("prediction","lower","upper"):
    d=np.abs(a[k]-b[k]); print("twin LN vs noLN", k, "maxabs %.3e"%d.max(), "rel_l2 %.3e"%(np.linalg.norm(a[k]-b[k])/np.linalg.norm(a[k])))
PYEOF
rm -f $K/e1/tw_*.npz; echo RP_K271_TWIN_END
