#!/bin/bash
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
export REPO=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_c
cd $REPO/Track_2/stream_c
$PY dump_twins.py --split aoa_20 --out /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/ood_aoa20.npz --ckpt /fs/scratch/PAS1187/qwu/tmp/realpde/screen/aoa/aoa20_ffnosim_s1.pth /fs/scratch/PAS1187/qwu/tmp/realpde/screen/aoa/aoa20_ffnosim_s2.pth /fs/scratch/PAS1187/qwu/tmp/realpde/screen/aoa/aoa20_ffno_s1.pth /fs/scratch/PAS1187/qwu/tmp/realpde/screen/aoa/aoa20_ffno_s2.pth /fs/scratch/PAS1187/qwu/tmp/realpde/screen/aoa/aoa20_fno_s1.pth
echo DONE_DUMP
