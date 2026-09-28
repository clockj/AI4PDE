#!/bin/bash
#SBATCH -J rp_k271_mread
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:30:00
# env TAGS (twin tags), W (a clone, for stream_eval.py)
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 HDF5_USE_FILE_LOCKING=FALSE
cd $W; D=/tmp/e1mr_$SLURM_JOB_ID; mkdir -p $D
for t in $TAGS; do
  unzip -q -o $K/sub_candidates/track2_$t.zip -d $D/$t
  grep -H "^JOINT_GAIN = \|^ARENA_STEPS = \|^SIGPOW = \|^D1C_N0 = \|^VC_N0 = \|^FW_N0 = \|^R1_ASYM = \|^ALPHA = " $D/$t/submission.py | cut -c1-120
  $PY Track_2/stream_eval.py --submission $D/$t --save $D/$t.npz 2>&1 | grep -a "load:\|errors\|scores" | cut -c1-60,230-330
  $PY $K/e1/mscore.py $K/e1/mread.jsonl $t=$D/$t.npz
  rm -f $D/$t.npz
done
echo RP_K271_MREAD_END
