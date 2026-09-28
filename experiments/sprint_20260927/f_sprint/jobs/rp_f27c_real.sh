#!/bin/bash
#SBATCH -J rp_f27c_real
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:45:00
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927; cd $S/wt_c/Track_2
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 HDF5_USE_FILE_LOCKING=FALSE
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
mk() { # src name d1c vc fw
  d=/tmp/rp_f27c_r_$2; rm -rf $d; cp -a $1 $d
  sed -i "s/^D1C_N0 = [^ ]*/D1C_N0 = $3/; s/^VC_N0 = [^ ]*/VC_N0 = $4/; s/^FW_N0 = [^ ]*/FW_N0 = $5/" $d/submission.py; echo $d; }
for spec in $OODARMS; do IFS=: read n d v f <<< "$spec"; d_=$(mk $S/pkg/c_work/ood_base ood_$n $d $v $f)
  echo "=== OOD $n"; $PY stream_eval.py --submission $d_ --split all --aoa 20 2>&1 | grep -E "load:|errors|scores|Traceback|Error" | sed 's/amp .*d1c_n0/d1c_n0/; s/sigema.*members/members/'
done
for spec in $INARMS; do IFS=: read n d v f <<< "$spec"; d_=$(mk $S/pkg/c/base in_$n $d $v $f)
  echo "=== IN full $n"; $PY stream_eval.py --submission $d_ 2>&1 | grep -E "load:|errors|scores|Traceback|Error" | sed 's/amp .*d1c_n0/d1c_n0/; s/sigema.*members/members/'
  echo "=== IN a15 $n"; $PY stream_eval.py --submission $d_ --aoa 15 20 2>&1 | grep -E "errors|scores|Traceback|Error"
done
echo RP_F27C_REAL_END
