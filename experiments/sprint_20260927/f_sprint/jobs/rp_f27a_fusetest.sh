#!/bin/bash
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
cd $S/pkg/a/fusetree
$PY $S/jobs/rp_f27a_fusetest.py $S/pkg/a/fusetree $S/ckpt/a/smoke.pth 
