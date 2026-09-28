#!/bin/bash
#SBATCH -J rp_k27v_D
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:20:00
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; SC=$K/sub_candidates; S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/sub_candidates
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
T=/tmp/rp_k27v_D_$SLURM_JOB_ID; mkdir -p $T; D=$K/ver/out/D; mkdir -p $D
echo "host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T)"; squeue -w $(hostname) -h -o "%i %u %j" | head
unzip -q $S/track1_x27_a080.zip -d $T/x27; unzip -q $SC/track1_k27all_a080.zip -d $T/k27all; unzip -q $SC/track1_k27fin_a080.zip -d $T/k27fin
unzip -q $S/track2_x27_d1c_a080.zip -d $T/t2x27; unzip -q $SC/track2_k27all_d1c_a080.zip -d $T/t2k27all; unzip -q $SC/track2_k27fin_d1c_a080.zip -d $T/t2k27fin
for rep in 1 2 3; do for N in 14 56 99; do for p in x27 k27all k27fin; do $PY $K/ver/t1lat.py $T/$p $N 2>&1 | grep -a "^LAT"; done; done; done
for rep in 1 2; do for p in t2x27 t2k27all t2k27fin; do $PY $K/ver/t2run.py $T/$p $D/$p_$rep.npz 2>&1 | grep -a "t2run"; done; done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv | head
rm -rf $T; echo "RP_K27V_D_END $(date +%T)"
