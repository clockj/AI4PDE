#!/bin/bash
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927; export OMP_NUM_THREADS=8; PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
echo "host=$(hostname) ulimit-l=$(ulimit -l) $(date)"
$PY $S/qa/qa_long.py $S/qa/$1 2>&1 | grep -v "^\[rpde-t2\] first"
echo "=== ulimit -l 65536 run $(date)"
( ulimit -l 65536; echo "ulimit-l now $(ulimit -l)"; $PY $S/qa/qa_t2_harness.py $S/qa/$1 graph $S/qa/out3/${1}_ulimitl 2>&1 | grep -E "LOAD|SUMMARY|RED|MEM_END|OUTFMT|Traceback|Error" )
echo "=== exit $(date)"
