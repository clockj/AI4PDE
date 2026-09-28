#!/bin/bash
#SBATCH -J rp_f27c_mx
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 28
#SBATCH --mem=220G
#SBATCH -t 00:50:00
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927; E=$S/pkg/c_work/probe_env/stream_c; O=$S/pkg/c_work/mx3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
BAND=${BAND:-$S/pkg/fallback/band_r38_a080_c0_fill.npz}
IN=$S/ckpt/c/twins_unseen.npz; OOD=$S/ckpt/c/ood_aoa20.npz
cd $E
jobs_list() {
  for a in $ARMS; do
    echo "--dump $IN --f 1.0 --arm $a --out $O/in_f1.00_$a.json"
    echo "--dump $OOD --fam-f 0 1 --fam-v 4 --f 1.0 --arm $a --out $O/ood_f1.00_$a.json"
    [ -n "${F129:-}" ] && echo "--dump $IN --f 1.29 --arm $a --out $O/in_f1.29_$a.json"
  done
  [ -n "${GAUGE:-}" ] && echo "--dump $OOD --fam-f 0 1 --fam-v 4 --f 1.29 --arm base --out $O/ood_f1.29_base.json"
}
jobs_list | xargs -P 26 -I{} sh -c "$PY -u ec_matrix.py --band $BAND {} 2>&1 | tail -2"
echo RP_F27C_MX_END
