#!/bin/bash
#SBATCH -J rp_k271_eqt1
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; F=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12} MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}
echo "host=$(hostname) job=$SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"; date
D=/tmp/e1t1_$SLURM_JOB_ID; mkdir -p $D
unzip -q -o $F/sub_candidates/track1_x27_a080.zip -d $D/base
for t in ${TAGS:-k27lat_a080 k27latln_a080}; do unzip -q -o $K/sub_candidates/track1_$t.zip -d $D/$t; done
V="base ${TAGS:-k27lat_a080 k27latln_a080}"
O=$K/e1/t1eq_$SLURM_JOB_ID.jsonl; E=$K/e1/t1edge_$SLURM_JOB_ID.jsonl
for dt in f32 f64; do for v in $V; do
  echo "== md5 $v $dt"; $PY $K/k1/k1_child.py $D/$v $v 56 $dt 1 $O 2>&1 | grep -v "^\[rpde" | tail -1 | cut -c1-400
done; done
for v in $V; do echo "== seq $v"; $PY $K/e1/t1_edge.py $D/$v $v seq $E 2>&1 | grep -v "^\[rpde" | tail -1 | cut -c1-300; done
for n in 1 13 100 1030; do for v in $V; do
  echo "== single $n $v"; $PY $K/e1/t1_edge.py $D/$v $v single:$n:f32 $E 2>&1 | grep -v "^\[rpde" | tail -1 | cut -c1-300
done; done
for v in $V; do $PY $K/e1/t1_edge.py $D/$v $v single:336:f32 $E $D/out_$v.npz >/dev/null 2>&1; done
$PY - <<PYEOF
import numpy as np
b = np.load("$D/out_base.npz")
for v in "$V".split()[1:]:
    c = np.load(f"$D/out_{v}.npz")
    for k in ("prediction","lower","upper"):
        d = np.abs(c[k]-b[k]); print(v, k, "maxabs %.3e"%d.max(), "rel_l2 %.3e"%(np.linalg.norm(c[k]-b[k])/np.linalg.norm(b[k])), "identical", bool((d==0).all()))
PYEOF
date; echo RP_K271_EQT1_END
