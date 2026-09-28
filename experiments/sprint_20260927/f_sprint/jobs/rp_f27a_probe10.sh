#!/bin/bash
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927; cd $S/ckpt/a/p2/stream_c
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
ARMS=(D1c30 "CONS(D1c30+VC50)" "T2_AGGR(D1c10+VC50)" "C2:EBG30(ebn1)+r1+VC30")
case $MODE in
  f100) $PY probe2.py --regime ood10 --f 1.0 --armfile arms_ea.py --arms "${ARMS[@]}" --ref D1c30 --procs 6 --json $S/ckpt/a/p10_f100.json ;;
  t14)  $PY probe2.py --regime ood10 --f 1.0 --trunc 14 --armfile arms_ea.py --arms "${ARMS[@]}" --ref D1c30 --procs 6 --json $S/ckpt/a/p10_t14.json ;;
  f129) $PY probe2.py --regime ood10 --f 1.29 --armfile arms_ea.py --arms "${ARMS[@]}" --ref D1c30 --procs 6 --json $S/ckpt/a/p10_f129.json ;;
  simonly) $PY probe2.py --regime ood10 --famf 0 --famv 2 --f 1.0 --armfile arms_ea.py --arms "${ARMS[@]}" --ref D1c30 --procs 6 ;;
  b2d) B2_THREADS=12 $PY b2_ea.py D1c30 1.0 ;;
  b2c) B2_THREADS=12 $PY b2_ea.py C2 1.0 ;;
esac
echo PROBE_DONE
