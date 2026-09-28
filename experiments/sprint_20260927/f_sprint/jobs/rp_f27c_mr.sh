#!/bin/bash
#SBATCH -J rp_f27c_mr
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927; cd $S/wt_c_p2/Track_2
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 HDF5_USE_FILE_LOCKING=FALSE
PYB=/users/PAS1187/qwu/.conda/envs/realpde/bin/python; PY="$PYB -u"
AO=/fs/scratch/PAS1187/qwu/tmp/realpde/screen/aoa
for z in $ZIPS; do t=$(basename $z .zip); t=${t#track2_}
  d=/tmp/rpc_tw_$t; rm -rf $d; mkdir -p $d; $PYB -m zipfile -e $z $d
  o=/tmp/rpc_ood_$t; rm -rf $o; cp -a $d $o; rm -f $o/fno*.pth; ln -s $AO/aoa20_ffnosim_s1.pth $o/fno.pth; ln -s $AO/aoa20_ffnosim_s2.pth $o/fno_2.pth; ln -s $AO/aoa20_fno_s1.pth $o/fno_3.pth
  echo "=== IN full $t"; $PY stream_eval.py --submission $d 2>&1 | grep -aE "scores|Traceback|Error"
  [ -n "${NOA15:-}" ] || { echo "=== IN a15 $t"; $PY stream_eval.py --submission $d --aoa 15 20 2>&1 | grep -aE "scores|Traceback|Error"; }
  echo "=== OOD $t"; $PY stream_eval.py --submission $o --split all --aoa 20 2>&1 | grep -aE "scores|Traceback|Error"
done
echo RP_F27C_MR_END
