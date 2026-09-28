#!/bin/bash
#SBATCH -J rp_f27c_time
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927; W=$S/wt_c; cd $W/Track_2
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 HDF5_USE_FILE_LOCKING=FALSE
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
echo "host=$(hostname) aff=$(nproc)"
for p in ref:$S/pkg/fallback/track2_x27asnp_d1c30_a080.zip $(for t in $TAGS; do z=$S/pkg/c_work/track2_$t.zip; [ -f $z ] || z=$S/pkg/c_work2/track2_$t.zip; echo $t:$z; done); do
  t=${p%%:*}; z=${p#*:}; d=/tmp/rp_f27c_t_$t; rm -rf $d; mkdir -p $d; $PY -m zipfile -e $z $d; done
for rep in 1 2 3 4; do
  for t in ref $TAGS; do
    echo "TIME $t r$rep $($PY stream_eval.py --submission /tmp/rp_f27c_t_$t --repeat 3 2>&1 | grep 'rep [0-9]:' | sed 's/.*median \([0-9.]*\),.*/\1/' | paste -sd' ')"
  done
done
echo RP_F27C_TIME_END
