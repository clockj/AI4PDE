#!/bin/bash
#SBATCH -J rp_k271_bint
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
# E1: build + gate K1 latency packages in K/wt_1 (committed 7fc7317). Band reused.
set -uo pipefail
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
W=${W:-$K/wt_int}
cd $W
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
OUT=$K/sub_candidates; mkdir -p $OUT $K/pkg/1
T0=$(date +%s)
stamp() { echo "[STAGE $(date +%T) +$(( $(date +%s) - T0 ))s] $*"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
git -C $W log -1 --format='%h %s' | cut -c1-120
BAND=${BAND:-/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/x_work/band_r38_a080_c0_fill.npz}
U=Track_1/unet
V="$U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
M="$U/p2_ffno_l8_sim_s1.pth $U/p2_ffno_l8_sim_s2.pth $U/p2_ffno_l8_sim_s3.pth $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"
A="$U/ffno_all_w40l8_sim_s1.pth $U/ffno_all_w40l8_sim_s2.pth $U/ffno_all_w40l8_sim_s3.pth $U/ffno_all_w40l8_sim_s4.pth $V"
COMMON=(RPDE_FORCE=1 RPDE_SUB=$OUT RPDE_BAND_FILL=1 RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor 'RPDE_SHIP_AMP="fp16"'
        RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense RPDE_BAND=$BAND)
T1K=(RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=True RPDE_SHIP_SMALL_N=2 RPDE_SHIP_SMALL_GRAPH_LAZY=True)
T2K=(RPDE_SHIP_R1=True RPDE_SHIP_R1_ASYM=True RPDE_SHIP_NUMPY_OUT=True)
LAT=(RPDE_SHIP_WARM_TAILS=True RPDE_SHIP_PIN_OUT=1024 RPDE_SHIP_BATCH_CAP=56 RPDE_SHIP_ARENA_STEPS=1024)
restore() { git -C $W checkout -- Track_1/submit_fno/submission.py Track_2/submit/submission.py Track_1/submit_fno/_fno.py; }
build() {  # tag members extra...
  local tag=$1 mem=$2; shift 2
  stamp "build $tag"
  local lat=("${LAT[@]}"); [ "${NOLAT:-0}" = 1 ] && lat=()
  env "${COMMON[@]}" "${T1K[@]}" "${T2K[@]}" "${lat[@]}" "$@" RPDE_A="$mem" RPDE_M="$M" RPDE_TAG=$tag \
      bash Track_1/submit_fno/build_round13.sh 0.80 2>&1 | tee $K/logs/build_$tag.log | grep -a "SUBMITTABLE\|load:\|^Track\|FAIL\|Error\|error" | head -30
  grep -H "^JOINT_GAIN = \|^ALPHA = \|^AMP = \|^WARM_TAILS = \|^PIN_OUT = \|^BATCH_CAP = \|^LN_MANUAL = \|^SMALL_N = \|^SMALL_GRAPH = \|^SMALL_GRAPH_LAZY = " Track_1/submit_fno/submission.py
  grep -H "^D2_DIRECT_PRED = \|^JOINT_GAIN = " Track_1/submit_fno/_fno.py
  grep -H "^ARENA_STEPS = \|^R1 = \|^R1_ASYM = \|^NUMPY_OUT = \|^D1C_N0 = \|^ALPHA = " Track_2/submit/submission.py
  restore
  stamp "build $tag done"
}
TAGS=${TAGS:-"k27lat_a080 k27latln_a080 k27lat_d1c_a080 k27twin_a080 k27twinln_a080"}
for t in $TAGS; do
  case $t in
    k27int_off_a080) NOLAT=1 build $t "$A" ;;
    k27int_lat_a080) build $t "$A" ;;
    k27all_a080) build $t "$A" RPDE_SHIP_JOINT_GAIN=True ;;
    k27all_d1c_a080) build $t "$A" RPDE_SHIP_JOINT_GAIN=True RPDE_SHIP_D1C_N0=30.0 ;;
    k27all_ln_a080) build $t "$A" RPDE_SHIP_JOINT_GAIN=True RPDE_SHIP_LN_MANUAL=True ;;
    k27lat_a080) build $t "$A" ;;
    k27latln_a080) build $t "$A" RPDE_SHIP_LN_MANUAL=True ;;
    k27lat_d1c_a080) build $t "$A" RPDE_SHIP_D1C_N0=30.0 ;;
    k27twin_a080) build $t "$M" ;;
    k27twinln_a080) build $t "$M" RPDE_SHIP_LN_MANUAL=True ;;
  esac
done
for t in $TAGS; do
  case $t in k27twin*|k27int_off*|k27int_lat*) continue ;; esac
  z=track2_$t
  stamp "validate_t2 --adaptive $z"
  $PY Track_2/validate_t2.py --adaptive --zip $OUT/$z.zip --extract /tmp/rp_k271_$z 2>&1 | tail -25
  stamp "stream_eval --causal-check $z"
  $PY Track_2/stream_eval.py --submission /tmp/rp_k271_$z --causal-check 2>&1 | tail -12
done
cd $OUT && sha256sum track?_k27*.zip
stamp "ALL DONE"
echo "RP_K271_BUILD_END"
