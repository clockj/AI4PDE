#!/bin/bash
#SBATCH -J rp_k272_t1twin
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:20:00
cd /fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/wt_2xr; export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u /fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/jobs/rp_k272_t1twin.py 2>&1 | grep -v "^\s*$" | tee /fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/pkg/candref_k272/t1twin_table.txt
echo RP_K272_T1TWIN_END
