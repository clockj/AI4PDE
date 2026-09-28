#!/bin/bash
#SBATCH -J rp_k27v_C
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:25:00
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; SB=$K/ver/sub
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
T=/tmp/rp_k27v_C_$SLURM_JOB_ID; mkdir -p $T; D=$K/ver/out/C; mkdir -p $D
echo "host=$(hostname) $(date +%T)"; ls -la $SB
for t in vtwin_fin vtwin_base; do unzip -q $SB/track1_$t.zip -d $T/t1_$t; grep -h "^EBAND = \|^JOINT_GAIN = \|^SMALL_GRAPH = " $T/t1_$t/submission.py | cut -c1-120; done
$PY $K/ver/t1chunk.py $T/t1_vtwin_fin 56 $D/t1_fin.npz 2>&1 | grep -a "t1chunk\|rror"
$PY $K/ver/t1chunk.py $T/t1_vtwin_base 56 $D/t1_base.npz 2>&1 | grep -a "t1chunk\|rror"
for t in vtwin_fin vtwin_base vtwin_fin_d1c vtwin_base_d1c; do unzip -q $SB/track2_$t.zip -d $T/t2_$t; o=${t#vtwin_}; $PY $K/ver/t2run.py $T/t2_$t $D/t2_$o.npz 2>&1 | grep -a "t2run\|load:\|rror"; done
$PY $K/ver/scoreC.py $D
rm -rf $T; echo "RP_K27V_C_END $(date +%T)"
