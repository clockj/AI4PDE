#!/bin/bash
#SBATCH -J rp_k27v_load
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:15:00
export OMP_NUM_THREADS=12
PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
for r in 1 2; do
$PY -u /fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/ver2/loadtime.py /fs/scratch/PAS1187/qwu/tmp/realpde/final0927/sub_candidates/track1_x27_a080.zip x27
$PY -u /fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/ver2/loadtime.py /fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/sub_candidates/track1_k27fin_a080.zip k27fin
$PY -u /fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/ver2/loadtime.py /fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/sub_candidates/track1_k27fin2_a080.zip k27fin2
done
echo LOAD_END
