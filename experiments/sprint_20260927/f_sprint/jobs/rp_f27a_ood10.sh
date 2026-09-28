#!/bin/bash
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
export REPO=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_a
cd /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_a/Track_2/stream_c
$PY dump_twins.py --split aoa_10 --out /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/a/ood_aoa10.npz --ckpt /fs/scratch/PAS1187/qwu/tmp/realpde/screen/aoa/aoa10_ffnosim_s1.pth /fs/scratch/PAS1187/qwu/tmp/realpde/screen/aoa/aoa10_ffno_s1.pth /fs/scratch/PAS1187/qwu/tmp/realpde/screen/aoa/aoa10_fno_s1.pth /fs/scratch/PAS1187/qwu/tmp/realpde/screen/aoa/aoa10_nosim_s1.pth
echo DONE_DUMP
