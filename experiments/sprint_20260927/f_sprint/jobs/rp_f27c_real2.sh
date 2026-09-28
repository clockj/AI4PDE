#!/bin/bash
#SBATCH -J rp_f27c_real2
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:45:00
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927; cd $S/wt_c_p2/Track_2
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 HDF5_USE_FILE_LOCKING=FALSE
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
Z=/tmp/rp_f27c_r2_zip; rm -rf $Z; mkdir -p $Z; $PY -m zipfile -e $S/pkg/c_work2/track2_p2off_d1c30.zip $Z
mk() { # src name "K=V K=V"
  d=/tmp/rp_f27c_r2_$2; rm -rf $d; cp -a $1 $d; cp $Z/submission.py $Z/_fno.py $d/
  for kv in $3; do k=${kv%%=*}; v=${kv#*=}; sed -i "s/^$k = [^ ]*/$k = $v/" $d/submission.py; done; echo $d; }
F='s/amp .*r1_asym/r1_asym/; s/sigema.*graph /graph /; s/graph_worst.*members/members/'
for spec in $ARMS; do n=${spec%%|*}; kv=${spec#*|}; kv=${kv//;/ }
  d_=$(mk $S/pkg/c/base in_$n "$kv")
  echo "=== IN full $n"; $PY stream_eval.py --submission $d_ 2>&1 | grep -E "load:|errors|scores|Traceback|Error" | sed "$F"
  echo "=== IN a15 $n"; $PY stream_eval.py --submission $d_ --aoa 15 20 2>&1 | grep -E "errors|scores|Traceback|Error"
  d_=$(mk $S/pkg/c_work/ood_base ood_$n "$kv")
  echo "=== OOD $n"; $PY stream_eval.py --submission $d_ --split all --aoa 20 2>&1 | grep -E "load:|errors|scores|Traceback|Error" | sed "$F"
done
echo RP_F27C_REAL2_END
