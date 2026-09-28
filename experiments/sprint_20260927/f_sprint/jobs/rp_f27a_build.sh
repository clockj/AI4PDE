#!/bin/bash
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
# EA (design A executor): build + gate a package in the BUILD worktree wt_a_bld (a snapshot of wt_a's model code),
# optionally the smoke round trip, optionally interleaved latency vs the fallback baseline packages.
# env: A_LIST M_LIST (space-separated abs paths) TAG  RT=0/1  TIMING=0/1  D1C=0/1
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
W=$S/wt_a_bld; WA=$S/wt_a; cd $W
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
PYB=/users/PAS1187/qwu/.conda/envs/realpde/bin/python; PY="$PYB -u"
T0=$(date +%s); stamp() { echo "[STAGE $(date +%T) +$(( $(date +%s) - T0 ))s] $*"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
OUT=$S/pkg/a; mkdir -p $OUT/snap_$TAG
# snapshot the design agent's model code into the build tree
for f in Track_1/submit_fno/_fno.py Track_1/unet/models_alt.py Track_1/unet/train2.py Track_1/unet/make_ffvit_init.py; do
  cp $WA/$f $W/$f; cp $WA/$f $OUT/snap_$TAG/; done
if [ -n "${FUSE:-}" ]; then sed -i "s/^FUSE_FFVIT = .*/FUSE_FFVIT = $FUSE/" $W/Track_1/submit_fno/_fno.py; cp $W/Track_1/submit_fno/_fno.py $OUT/snap_$TAG/_fno.py; fi
grep -H "^FUSE_FFVIT" $W/Track_1/submit_fno/_fno.py
md5sum $W/Track_1/submit_fno/_fno.py
restore() { git -C $W checkout -- Track_1/submit_fno/submission.py Track_2/submit/submission.py; cp $OUT/snap_$TAG/_fno.py $W/Track_1/submit_fno/_fno.py; }
if [ "${RT:-0}" = 1 ]; then stamp "packed round trip on smoke"; $PY $S/jobs/rp_f27a_rt.py $W $S/ckpt/a/smoke.pth; fi
COMMON=(RPDE_FORCE=1 RPDE_SUB=$OUT RPDE_BAND_FILL=1 RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor 'RPDE_SHIP_AMP="fp16"'
        RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense)
T1K=(RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=True RPDE_SHIP_SMALL_N=2 RPDE_SHIP_SMALL_GRAPH_LAZY=True)
T2K=(RPDE_SHIP_R1=True RPDE_SHIP_R1_ASYM=True RPDE_SHIP_NUMPY_OUT=True)
stamp "build $TAG (band refit)"
env "${COMMON[@]}" "${T1K[@]}" "${T2K[@]}" RPDE_A="$A_LIST" RPDE_M="$M_LIST" RPDE_TAG=$TAG \
    bash Track_1/submit_fno/build_round13.sh 0.80 2>&1 | tee $S/logs/build_$TAG.log | grep -E "SUBMITTABLE|FAIL|Error|error|rpde-t1\] load|rpde-t2\] load|sps board|^ALPHA|^AMP|Traceback|ms/|MB"
cp Track_2/submit/band8.npz $OUT/band_$TAG.npz
restore
if [ "${D1C:-0}" = 1 ]; then
  stamp "build ${TAG}_d1c (band reused)"
  env "${COMMON[@]}" "${T1K[@]}" "${T2K[@]}" RPDE_SHIP_D1C_N0=30.0 RPDE_BAND=$OUT/band_$TAG.npz \
      RPDE_A="$A_LIST" RPDE_M="$M_LIST" RPDE_TAG=${TAG}_d1c \
      bash Track_1/submit_fno/build_round13.sh 0.80 2>&1 | tee $S/logs/build_${TAG}_d1c.log | grep -E "SUBMITTABLE|FAIL|Traceback|rpde-t2\] load"
  restore; rm -f $OUT/track1_${TAG}_d1c.zip
fi
for z in track2_$TAG $( [ "${D1C:-0}" = 1 ] && echo track2_${TAG}_d1c ); do
  stamp "validate_t2 --adaptive $z"
  $PY Track_2/validate_t2.py --adaptive --zip $OUT/$z.zip --extract /tmp/rp_f27a_$z 2>&1 | tail -15
  stamp "causal-check $z"
  $PY Track_2/stream_eval.py --submission /tmp/rp_f27a_$z --causal-check 2>&1 | grep -iE "causal|errors|local|FAIL"
done
git -C $W status --short
if [ "${TIMING:-0}" = 1 ]; then
  stamp "interleaved latency: baseline fallback vs $TAG (shared node, rough)"
  for p in track1_x27lz_a080:$S/pkg/fallback track2_x27asnp_a080:$S/pkg/fallback track1_$TAG:$OUT track2_$TAG:$OUT; do
    n=${p%%:*}; dd=${p#*:}; rm -rf /tmp/rp_f27a_tm_$n; mkdir -p /tmp/rp_f27a_tm_$n; $PYB -m zipfile -e $dd/$n.zip /tmp/rp_f27a_tm_$n; done
  for r in 1 2 3; do
    for n in track1_x27lz_a080 track1_$TAG; do $PY $S/jobs/rp_f27a_time.py t1 /tmp/rp_f27a_tm_$n $W/Track_1/val_unseen.npz 2>&1 | grep "^T1"; done
    for n in track2_x27asnp_a080 track2_$TAG; do $PY $S/jobs/rp_f27a_time.py t2 /tmp/rp_f27a_tm_$n $W/Track_1/val_unseen.npz 2>&1 | grep "^T2"; done
  done
fi
cd $OUT && sha256sum track?_${TAG}*.zip; ls -la track?_${TAG}*.zip
stamp "ALL DONE"; echo RP_F27A_BUILD_END
