#!/bin/bash
#SBATCH -J rp_k27v_B
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:30:00
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; SC=$K/sub_candidates
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
T=/tmp/rp_k27v_B_$SLURM_JOB_ID; mkdir -p $T; D=$K/ver/out/B; mkdir -p $D
echo "host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T)"
for p in fin all sbt2; do for v in "" _d1c; do unzip -q $SC/track2_k27${p}${v}_a080.zip -d $T/$p$v; sha256sum $SC/track2_k27${p}${v}_a080.zip; done; done
for v in "" _d1c; do for p in fin all sbt2; do $PY $K/ver/t2run.py $T/$p$v $D/$p$v.npz 2>&1 | grep -a "t2run\|load:\|rror"; done; done
$PY $K/ver/cmpB.py $D
cd $K/ver/wt
for v in "" _d1c; do
  echo "=== validate_t2 --adaptive fin$v $(date +%T)"; $PY Track_2/validate_t2.py --adaptive --zip $SC/track2_k27fin${v}_a080.zip --extract $T/val$v 2>&1 | grep -a "load:\|SUBMITTABLE\|  - \|ms/step"
  echo "=== causal fin$v"; $PY Track_2/stream_eval.py --submission $T/fin$v --causal-check 2>&1 | grep -a "CAUS\|load:\|errors\|scores"
done
rm -rf $T; echo "RP_K27V_B_END $(date +%T)"
