#!/bin/bash
#SBATCH -J rp_f27c_build2
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 01:00:00
# EC: knob-off identity + candidate T2 builds (stream stack) + gates + interleaved latency.
# BUILDS env: space-separated "tag:D1C:VC:FW" (FW/VC 0 = unset/off)
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
W=${W:-$S/wt_c_p2}
cd $W
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
PYB=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
PY="$PYB -u"
OUT=$S/pkg/c_work2; mkdir -p $OUT
T0=$(date +%s)
stamp() { echo "[STAGE $(date +%T) +$(( $(date +%s) - T0 ))s] $*"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
U=Track_1/unet
A="$U/ffno_all_w40l8_sim_s1.pth $U/ffno_all_w40l8_sim_s2.pth $U/ffno_all_w40l8_sim_s3.pth $U/ffno_all_w40l8_sim_s4.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
M="$U/p2_ffno_l8_sim_s1.pth $U/p2_ffno_l8_sim_s2.pth $U/p2_ffno_l8_sim_s3.pth $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"
BAND=${BAND:-$S/pkg/fallback/band_r38_a080_c0_fill.npz}
COMMON=(RPDE_FORCE=1 RPDE_SUB=$OUT RPDE_BAND_FILL=1 RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor 'RPDE_SHIP_AMP="fp16"'
        RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense)
T1K=(RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=True RPDE_SHIP_SMALL_N=2 RPDE_SHIP_SMALL_GRAPH_LAZY=True)
T2K=(RPDE_SHIP_R1=True RPDE_SHIP_R1_ASYM=True RPDE_SHIP_NUMPY_OUT=True)
restore() { git -C $W checkout -- Track_1/submit_fno/submission.py Track_2/submit/submission.py Track_1/submit_fno/_fno.py 2>/dev/null; \
            git -C $W stash list >/dev/null; }
# NOTE: Track_2/submit/submission.py carries design C's UNCOMMITTED patch; `git checkout` would revert it. Save + restore by copy instead.
cp Track_2/submit/submission.py $OUT/submission_patched.py
cp Track_1/submit_fno/submission.py $OUT/t1_submission_orig.py
cp Track_1/submit_fno/_fno.py $OUT/fno_orig.py
restore() { cp $OUT/submission_patched.py Track_2/submit/submission.py; cp $OUT/t1_submission_orig.py Track_1/submit_fno/submission.py; cp $OUT/fno_orig.py Track_1/submit_fno/_fno.py; }

for spec in $BUILDS; do
  tag=${spec%%|*}; kv=${spec#*|}; IFS=";" read -ra K <<< "$kv"
  stamp "build $tag  ${K[*]}"
  env "${COMMON[@]}" "${T1K[@]}" "${T2K[@]}" "${K[@]}" RPDE_BAND=$BAND RPDE_A="$A" RPDE_M="$M" RPDE_TAG=$tag \
      bash Track_1/submit_fno/build_round13.sh 0.80 > $S/logs/rp_f27c_build_$tag.log 2>&1
  grep -H "^R1_TSM = \|^R1_BLK_N0 = \|^D1C_EB = \|^D1C_EB_N0 = \|^D1C_LEAD_N0 = \|^ALPHA = \|^D1C_N0 = \|^VC_N0 = \|^FW_N0 = \|^R1 = \|^R1_ASYM = \|^NUMPY_OUT = \|^AMP = " Track_2/submit/submission.py
  restore
  cmp Track_2/submit/submission.py $OUT/submission_patched.py && echo restored-ok
done

for spec in $BUILDS; do
  tag=${spec%%|*}
  z=$OUT/track2_$tag.zip; X=/tmp/rp_f27c2_v_$tag; rm -rf $X
  stamp "validate_t2 --adaptive $tag"
  $PY Track_2/validate_t2.py --adaptive --zip $z --extract $X 2>&1 | tail -6
  stamp "stream_eval --causal-check $tag (all-data, contaminated scores)"
  $PY Track_2/stream_eval.py --submission $X --causal-check 2>&1 | grep -v "rep " | tail -9
done

if [ -n "${IDENT:-}" ]; then
  stamp "knob-off identity: $IDENT vs fallback track2_x27asnp_d1c30_a080"
  F=/tmp/rp_f27c2_fb; rm -rf $F; mkdir -p $F; $PYB -m zipfile -e $S/pkg/fallback/track2_x27asnp_d1c30_a080.zip $F
  X=/tmp/rp_f27c2_v_$IDENT
  (cd $F && find . -type f ! -name '*.pyc' | sort | xargs md5sum) > /tmp/rp_f27c2_md5_a.txt
  (cd $X && find . -type f ! -name '*.pyc' ! -path '*__pycache__*' | sort | xargs md5sum) > /tmp/rp_f27c2_md5_b.txt
  diff /tmp/rp_f27c2_md5_a.txt /tmp/rp_f27c2_md5_b.txt && echo "IDENT all files md5 equal"
  diff $F/submission.py $X/submission.py | head -80
  $PY Track_2/stream_eval.py --submission $F --save /tmp/rp_f27c2_se_fb.npz 2>&1 | tail -5
  $PY Track_2/stream_eval.py --submission $X --save /tmp/rp_f27c2_se_off.npz 2>&1 | tail -5
  $PYB - <<'EOF'
import numpy as np
a=np.load('/tmp/rp_f27c2_se_fb.npz'); b=np.load('/tmp/rp_f27c2_se_off.npz')
for k in ['pred','lower','upper']:
    print('IDENT', k, a[k].shape, 'bit-equal' if np.array_equal(a[k],b[k]) else f'DIFF max {np.abs(a[k]-b[k]).max():.3e}')
EOF
fi

if [ -n "${TIMING:-}" ]; then
  F=/tmp/rp_f27c2_fb2; rm -rf $F; mkdir -p $F; $PYB -m zipfile -e $S/pkg/fallback/track2_x27asnp_d1c30_a080.zip $F
  for rep in 1 2 3; do
    stamp "timing round $rep: ref d1c30"
    $PY Track_2/stream_eval.py --submission $F --repeat 3 2>&1 | grep "local .*ms/step"
    for spec in $BUILDS; do tag=${spec%%|*}
      echo -n "TIME $tag r$rep: "; $PY Track_2/stream_eval.py --submission /tmp/rp_f27c2_v_$tag --repeat 3 2>&1 | grep "local .*ms/step"
    done
    echo -n "TIME ref r$rep (again): "; $PY Track_2/stream_eval.py --submission $F --repeat 3 2>&1 | grep "local .*ms/step"
  done
fi
cd $OUT && sha256sum track2_*.zip
stamp "ALL DONE"
echo "RP_F27C_BUILD_END"
