#!/bin/bash
#SBATCH -J rp_k272_verify
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:30:00
set -uo pipefail
W=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/wt_2xr; cd $W
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T)"
$PY Track_1/verify/joint_gain.py --t1zip /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/sub_candidates/track1_x27_a080.zip --t2zip /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/sub_candidates/track2_x27_a080.zip --work /tmp/jg_verify_$SLURM_JOB_ID
echo "exit $? $(date +%T)"; rm -rf /tmp/jg_verify_$SLURM_JOB_ID; echo RP_K272_VERIFY_END
