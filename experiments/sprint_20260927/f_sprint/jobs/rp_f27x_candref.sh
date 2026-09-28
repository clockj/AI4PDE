#!/bin/bash
#SBATCH -J rp_f27x_candref
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:30:00
# PAIRED held-out read of a candidate member set, the same instruments as $S/notes/baseline_ref.md.
# Usage (paths absolute or relative to $W; space-separated):
#   sbatch --export=ALL,CAND_A="<all-data members>",CAND_M="<twins>",TAG=<tag>,W=<your worktree> \
#          ${DESK:+--exclude=$DESK} --output=$S/logs/rp_f27<dir>_candref_%j.log $S/jobs/rp_f27x_candref.sh
# Reads only from W (does not sed anything in it). Do NOT run while a build_round13.sh runs in the same W.
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
W=${W:-$S/wt_x}; cd $W
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
PYB=/users/PAS1187/qwu/.conda/envs/realpde/bin/python; PY="$PYB -u"
T0=$(date +%s); stamp() { echo "[STAGE $(date +%T) +$(( $(date +%s) - T0 ))s] $*"; }
O=$S/pkg/candref_$TAG; mkdir -p $O
abs() { for c in $1; do case $c in /*) echo -n "$c ";; *) echo -n "$W/$c ";; esac; done; }
A=$(abs "$CAND_A"); M=$(abs "$CAND_M"); MC=$(echo $M | tr ' ' ',')
echo "A: $A"; echo "M: $M"
for f in 1.39 1.29; do
  stamp "ens_v9 twins AoA15-20 f=$f"; (cd Track_1 && $PY ens_v9.py --sets "$MC" --alpha 0.80 --f $f --aoa 15 20)
  stamp "ens_v9 twins full f=$f";     (cd Track_1 && $PY ens_v9.py --sets "$MC" --alpha 0.80 --f $f --aoa 0 5 10 15 20)
done
stamp "fit_band --ckpt A --err-ckpt M (shipped construction: 0.80, p 0.45, centre0, fill, 1.39)"
$PY Track_1/submit_fno/fit_band.py --ckpt $A --err-ckpt $M --alpha 0.80 --band-p 0.45 --centre-alpha 0 --fill-unscored \
    --out $O/band.npz | grep -v "^\s*$"
stamp "twin T2 package (members = twins) with the asnp constants"
$PY Track_2/build_t2.py --band $O/band.npz --int8 --linear tensor --ckpt $M --stage $O/stage --out $O/twin.zip | tail -2
cp $S/pkg/x_work/t2twin_stage/submission.py $O/stage/submission.py     # baked: ALPHA 0.80 fp16 bmm dense R1 asym numpy
for v in static asym asym_d1c; do
  d=/tmp/rp_f27x_cand_${TAG}_$v; rm -rf $d; cp -r $O/stage $d
  case $v in
    static)   sed -i "s/^R1 = .*/R1 = False/; s/^R1_ASYM = .*/R1_ASYM = False/" $d/submission.py ;;
    asym)     : ;;
    asym_d1c) sed -i "s/^D1C_N0 = [^ ]*/D1C_N0 = 30.0/" $d/submission.py ;;
  esac
  stamp "stream twin $v all";      $PY Track_2/stream_eval.py --submission $d 2>&1 | grep -E "errors|scores|final|members"
  stamp "stream twin $v aoa15-20"; $PY Track_2/stream_eval.py --submission $d --aoa 15 20 2>&1 | grep -E "errors|scores|final"
done
stamp "ALL DONE"; echo RP_F27X_CANDREF_END
