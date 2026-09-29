#!/bin/bash
# Gate R2 on ONE card for ONE package pair (inside a GPU job):
#   bash r2_run.sh CARD NAME TRACK2.zip TRACK1.zip "MODES"
# -> $R2ROOT/CARD/NAME/: validate_t2 --adaptive log; stream_eval of two trajectories (saved); the sprint's QA harness in each
#    MODE (graph = the package's own choices; eager / fp32 / eagerfp32 / pinfail = forced fallbacks), each saving the whole
#    val_unseen stream; Track 1: t1_check.py on the archive (and with --fp32 when MODES holds fp32).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CARD=$1; NAME=$2; Z2=$3; Z1=$4; MODES=${5:-graph}
R2ROOT=${R2ROOT:-/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/r2}
WT=${OOD_WT:-/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/wt_ood}
PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
O=$R2ROOT/$CARD/$NAME; rm -rf "$O"; mkdir -p "$O/pkg"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HDF5_USE_FILE_LOCKING=FALSE OOD_WT=$WT R2_WORK=$O
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
echo "card $CARD host $(hostname) job ${SLURM_JOB_ID:-} gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
sha256sum "$Z2" "$Z1"
$PY -m zipfile -e "$Z2" "$O/pkg"
echo "== validate_t2 --adaptive"
$PY "$WT/Track_2/validate_t2.py" --adaptive --zip "$Z2" --extract "$O/vt2" > "$O/validate_t2.log" 2>&1
tr -d '\000' < "$O/validate_t2.log" | grep -a "SUBMITTABLE\|NOT SUBMITTABLE\|load:\|FAIL" | cut -c1-400
echo "== stream_eval, two trajectories"
$PY "$WT/Track_2/stream_eval.py" --submission "$O/pkg" --traj 3750_0 21600_15 --save "$O/stream.npz" > "$O/stream_eval.log" 2>&1
tr -d '\000' < "$O/stream_eval.log" | grep -a "load:\|scores\|rep 0" | cut -c1-600
for m in $MODES; do
  echo "== QA harness mode $m"
  $PY -u "$HERE/t2_harness.py" "$O/pkg" "$m" "$O/qa_$m" > "$O/qa_$m.log" 2>&1
  tr -d '\000' < "$O/qa_$m.log" | grep -a "^LOAD\|RED" | cut -c1-400
done
echo "== Track 1"
$PY "$HERE/t1_check.py" "$Z1" "$O/t1.npz" > "$O/t1.log" 2>&1; grep -a '"submittable"\|"amp_active"\|"load_line"' "$O/t1.log" | cut -c1-400
case " $MODES " in *" fp32 "*) $PY "$HERE/t1_check.py" "$Z1" "$O/t1_fp32.npz" --fp32 > "$O/t1_fp32.log" 2>&1
  grep -a '"submittable"\|"amp_active"' "$O/t1_fp32.log";; esac
echo RP_DP_OOD_END
