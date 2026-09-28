#!/bin/bash
# Submit ood_run.sh as one Ascend job, off the owner's desktop node.
#
#   bash ood_submit.sh NAME REGIME LAYOUT [TIME]         # env passes through (OOD_BAND, OOD_STAGES, OOD_MEMBERS, ...)
#   OOD_DEPEND=<jobid> bash ood_submit.sh ...           # start after that job ends (afterany), to stay within the job cap
#
# Job name rp_dp_ood_<NAME>; log $OOD_ROOT/logs/<jobname>_<id>.log. Prints "SUBMITTED <jobid> <NAME>".
# The campaign gives this instrument at most 2 running jobs.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NAME=${1:?NAME}; REGIME=${2:?REGIME}; LAYOUT=${3:?LAYOUT}; TIME=${4:-01:15:00}
ROOT=${OOD_ROOT:-/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood}
mkdir -p "$ROOT/logs" "$ROOT/jobs"
J=$ROOT/jobs/ood_$NAME.sh
cat > "$J" <<EOF
#!/bin/bash
#SBATCH -J rp_dp_ood_$NAME
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t $TIME
bash $HERE/ood_run.sh $NAME $REGIME $LAYOUT
EOF
DESK=$(squeue -M ascend -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,)
ID=$(env -u SLURM_JOB_ID -u SLURM_MPI_TYPE -u CUDA_VISIBLE_DEVICES \
  sbatch --parsable ${DESK:+--exclude="$DESK"} ${OOD_DEPEND:+--dependency=afterany:$OOD_DEPEND} \
         --chdir="$ROOT" --output="$ROOT/logs/%x_%j.log" "$J") || exit 1
ID=${ID%%;*}
echo "SUBMITTED $ID $NAME"
