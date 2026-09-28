#!/bin/bash
#SBATCH -J rp_k27v_E
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:25:00
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; SC=$K/sub_candidates
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
T=/tmp/rp_k27v_E_$SLURM_JOB_ID; mkdir -p $T; D=$K/ver/out/E; mkdir -p $D
echo "host=$(hostname) $(date +%T)"
unzip -q $SC/track1_k27fin_a080.zip -d $T/fin; unzip -q $SC/track1_k27fin2_a080.zip -d $T/fin2; unzip -q $SC/track1_k27all_a080.zip -d $T/all
unzip -q $SC/track2_k27fin_d1c_a080.zip -d $T/t2fin; unzip -q $SC/track2_k27fin2_a080.zip -d $T/t2fin2
if [ "${PART:-1}" = 1 ]; then
for p in fin fin2; do $PY $K/ver/eseq.py $T/$p $D/seq_$p.npz seq 2>&1 | grep -a "SEQ\|MEM\|rror\|Trace"; done
$PY - <<PYE
import numpy as np
a, b = np.load("$D/seq_fin.npz")["hashes"], np.load("$D/seq_fin2.npz")["hashes"]
print("E1 seq: calls equal fin2 vs fin:", int((a == b).sum()), "/", len(a))
PYE
for N in 14 56 99 336; do for dt in float32 float64; do for p in fin fin2; do $PY $K/ver/eseq.py $T/$p $D/one_${p}_${N}_$dt.npz one $N $dt 2>&1 | grep -a "ONE\|rror" | sed "s/^/$p /"; done; done; done
$PY - <<PYE
import numpy as np
for N in (14, 56, 99, 336):
    for dt in ("float32", "float64"):
        a, b = np.load(f"$D/one_fin_{N}_{dt}.npz"), np.load(f"$D/one_fin2_{N}_{dt}.npz")
        print("E1 fresh", N, dt, "bitwise:", all(np.array_equal(a[k], b[k]) for k in ("P", "lo", "hi")))
PYE
for p in t2fin t2fin2; do $PY $K/ver/t2mem.py $T/$p $D/$p.npz 2>&1 | grep -a "t2run\|MEM\|rror"; done
$PY - <<PYE
import numpy as np
a, b = np.load("$D/t2fin.npz"), np.load("$D/t2fin2.npz")
print("E3 T2 stream fin2 vs fin_d1c bitwise:", {k: bool(np.array_equal(a[k], b[k])) for k in ("P", "L", "U")})
PYE
else
for rep in 1 2 3; do for N in 14 56 99; do for p in all fin fin2; do $PY $K/ver/t1lat.py $T/$p $N 2>&1 | grep -a "^LAT" | sed "s/^LAT /LAT $p /"; done; done; done
fi
rm -rf $T; echo "RP_K27V_E_END $(date +%T)"
