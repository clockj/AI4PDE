#!/bin/bash
#SBATCH -J rp_k27v_A
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:35:00
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
T=/tmp/rp_k27v_A_$SLURM_JOB_ID; mkdir -p $T; D=$K/ver/out/A; mkdir -p $D
echo "host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T)"
unzip -q $K/sub_candidates/track1_k27fin_a080.zip -d $T/fin; unzip -q $K/sub_candidates/track1_k27all_a080.zip -d $T/all; unzip -q $S/sub_candidates/track1_x27_a080.zip -d $T/x27
sha256sum $K/sub_candidates/track1_k27fin_a080.zip
for N in 14 56 99 336; do for p in fin all; do $PY $K/ver/t1chunk.py $T/$p $N $D/${p}_$N.npz 2>&1 | grep -a "t1chunk\|rror\|load:"; done; done
$PY $K/ver/t1chunk.py $T/x27 56 $D/x27_56.npz 2>&1 | grep -a "t1chunk\|rror"
$PY $K/ver/cmpA.py $D
echo "=== A3 band formula $(date +%T)"
$PY $K/wt_fin/Track_1/verify/k0a_dump.py $T/all $D/dump_all.npz 2>&1 | tail -3
$PY $K/wt_fin/Track_1/verify/k0a_dump.py $T/fin $D/dump_fin.npz 2>&1 | tail -3
$PY $K/wt_fin/Track_1/verify/sigblend.py $D/dump_all.npz $D/dump_fin.npz $K/pkg/k0/band_k0_sigblend_lam0.5.npz $K/pkg/k0/band_k0_sigblend_lam0.5.json 2>&1 | tail -20
echo "=== A5 validate_zip $(date +%T)"
cd $K/ver/wt && $PY Track_1/submit_fno/validate_zip.py $K/sub_candidates/track1_k27fin_a080.zip 2>&1 | tail -30
echo "=== A4 edge seq $(date +%T)"
$PY $K/ver/edge.py $T/fin seq 2>&1 | tail -60
echo "=== A4 edge fresh $(date +%T)"
for N in 1 2 13 14 56 99 100 129 336; do for dt in float32 float64; do $PY $K/ver/edge.py $T/fin one $N $dt 2>&1 | grep -a "N=\|RESULT" | tr '\n' ' '; echo; done; done
rm -rf $T; echo "RP_K27V_A_END $(date +%T)"
