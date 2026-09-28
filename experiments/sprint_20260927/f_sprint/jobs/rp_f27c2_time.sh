#!/bin/bash
source /fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh
export REPO=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_c2
cd $REPO/Track_2/stream_c
$PY interleave_time.py /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/f_base /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/o_cons /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/f_k0 /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/c2/f_s1
echo DONE_TIME
