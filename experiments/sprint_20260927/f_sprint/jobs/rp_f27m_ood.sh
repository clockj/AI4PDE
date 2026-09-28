#!/bin/bash
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927; cd $S/wt_c_p2/Track_2
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 HDF5_USE_FILE_LOCKING=FALSE
PYB=/users/PAS1187/qwu/.conda/envs/realpde/bin/python; PY="$PYB -u"
AO=/fs/scratch/PAS1187/qwu/tmp/realpde/screen/aoa
for z in $S/pkg/c_work3/track2_tw_p2.zip $S/pkg/c_work3/track2_tw_mp2.zip $S/pkg/c_work3/track2_tw_mp2nj.zip $S/pkg/c_work3/track2_tw_mcons.zip; do t=$(basename $z .zip); t=${t#track2_}
  d=/tmp/rpm_tw_$t; rm -rf $d; mkdir -p $d; $PYB -m zipfile -e $z $d
  o=/tmp/rpm_ood_$t; rm -rf $o; cp -a $d $o; rm -f $o/fno*.pth; ln -s $AO/aoa20_ffnosim_s1.pth $o/fno.pth; ln -s $AO/aoa20_ffnosim_s2.pth $o/fno_2.pth; ln -s $AO/aoa20_fno_s1.pth $o/fno_3.pth
  echo "=== OOD $t"; $PY stream_eval.py --submission $o --split all --aoa 20 > $S/logs/m_ood_$t.txt 2>&1; tr -d '\000' < $S/logs/m_ood_$t.txt | grep -a -E "^ +(errors|scores)" | tr -s ' '
  echo "=== INFULL $t"; $PY stream_eval.py --submission $d > $S/logs/m_in_$t.txt 2>&1; tr -d '\000' < $S/logs/m_in_$t.txt | grep -a -E "^ +(errors|scores)" | tr -s ' '
done
echo RP_F27M_END
