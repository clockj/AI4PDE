#!/bin/bash
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927; export OMP_NUM_THREADS=8
for r in 1 2 3; do for p in T2_CONS_track2_c27cons_d1c30vc50 W_p2_consb2 W_p2_c2 W_p2_c2b2 FALLBACK; do
/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u $S/qa/qa_time.py $S/qa/$p 2>&1 | grep TIME; done; done
