#!/bin/bash
#SBATCH -J rp_k27_final
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
# Commander: FINAL integration builds in K/wt_fin (K1 latency + K2 JOINT_GAIN + K0 SIGBLEND band: T1 EBAND, T2 SIGPOW).
# env: TAGS (space separated), LAM (0.5 | 0.3), optional A_OVERRIDE / M_OVERRIDE (member / twin lists)
set -uo pipefail
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
W=${W:-$K/wt_fin}
cd $W
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
PYB=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
PY="$PYB -u"
OUT=$K/sub_candidates; mkdir -p $OUT $K/pkg/fin
T0=$(date +%s)
stamp() { echo "[STAGE $(date +%T) +$(( $(date +%s) - T0 ))s] $*"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
git -C $W log -1 --format='%h %s' | cut -c1-120
git -C $W status --short | head -5
LAM=${LAM:-0.5}
SHIPBAND=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/x_work/band_r38_a080_c0_fill.npz
SBBAND=$K/pkg/k0/band_k0_sigblend_lam$LAM.npz
EBAND=$($PYB -c "import json;print(json.load(open('$K/pkg/k0/band_k0_sigblend_lam$LAM.json'))['EBAND'])")
SIGPOW=$($PYB -c "import json;q=json.load(open('$K/pkg/k0/band_k0_sigblend_lam$LAM.json'))['T2_sigpow'];print('(%r, %r)' % (float(q[0]), float(q[1])))")
echo "LAM $LAM  EBAND $EBAND  SIGPOW $SIGPOW"
U=Track_1/unet
V="$U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
M=${M_OVERRIDE:-"$U/p2_ffno_l8_sim_s1.pth $U/p2_ffno_l8_sim_s2.pth $U/p2_ffno_l8_sim_s3.pth $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"}
A=${A_OVERRIDE:-"$U/ffno_all_w40l8_sim_s1.pth $U/ffno_all_w40l8_sim_s2.pth $U/ffno_all_w40l8_sim_s3.pth $U/ffno_all_w40l8_sim_s4.pth $V"}
COMMON=(RPDE_FORCE=1 RPDE_SUB=$OUT RPDE_BAND_FILL=1 RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor 'RPDE_SHIP_AMP="fp16"'
        RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense)
# T1: EBAND needs the GPU-band path for every N -> SMALL_GRAPH off (the batch-1 graph only ever served N = 1 calls)
T1K_SB=(RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=False RPDE_SHIP_SMALL_GRAPH_LAZY=False)
T1K_27=(RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=True RPDE_SHIP_SMALL_N=2 RPDE_SHIP_SMALL_GRAPH_LAZY=True)
T2K=(RPDE_SHIP_R1=True RPDE_SHIP_R1_ASYM=True RPDE_SHIP_NUMPY_OUT=True)
LAT=(RPDE_SHIP_WARM_TAILS=True RPDE_SHIP_PIN_OUT=${PIN_OUT:-128} RPDE_SHIP_BATCH_CAP=56 RPDE_SHIP_ARENA_STEPS=${ARENA_STEPS:-1024})
restore() { git -C $W checkout -- Track_1/submit_fno/submission.py Track_2/submit/submission.py Track_1/submit_fno/_fno.py; }
build() {  # tag band extra...
  local tag=$1 band=$2; shift 2
  stamp "build $tag (band $band)"
  env "${COMMON[@]}" "${T2K[@]}" "${LAT[@]}" RPDE_BAND=$band "$@" RPDE_A="$A" RPDE_M="$M" RPDE_TAG=$tag \
      bash Track_1/submit_fno/build_round13.sh 0.80 2>&1 | tee $K/logs/build_$tag.log | grep -a "SUBMITTABLE\|load:\|^Track\|FAIL\|Error\|error\|REFUS\|EBAND" | head -40
  grep -H "^EBAND = \|^JOINT_GAIN = \|^ALPHA = \|^AMP = \|^WARM_TAILS = \|^PIN_OUT = \|^BATCH_CAP = \|^LN_MANUAL = \|^SMALL_N = \|^SMALL_GRAPH = \|^SMALL_GRAPH_LAZY = \|^GPU_BAND = \|^CENTRE_ALPHA = " Track_1/submit_fno/submission.py
  grep -H "^D2_DIRECT_PRED = \|^JOINT_GAIN = " Track_1/submit_fno/_fno.py
  grep -H "^SIGPOW = \|^JOINT_GAIN = \|^ARENA_STEPS = \|^R1 = \|^R1_ASYM = \|^NUMPY_OUT = \|^D1C_N0 = \|^ALPHA = \|^SIGEMA = " Track_2/submit/submission.py
  restore
  stamp "build $tag done"
}
TAGS=${TAGS:-"k27fin_a080"}
for t in $TAGS; do
  case $t in
    # both tracks valid: SIGBLEND band + T1 EBAND + T2 SIGPOW, latency knobs, JOINT_GAIN
    k27fin2_a080)     build $t $SBBAND "${T1K_SB[@]}" RPDE_SHIP_JOINT_GAIN=True "RPDE_SHIP_EBAND=$EBAND" "RPDE_SHIP_SIGPOW=$SIGPOW" RPDE_SHIP_D1C_N0=30.0 ;;
    k27fin_a080)      build $t $SBBAND "${T1K_SB[@]}" RPDE_SHIP_JOINT_GAIN=True "RPDE_SHIP_EBAND=$EBAND" "RPDE_SHIP_SIGPOW=$SIGPOW" ;;
    k27fin_d1c_a080)  build $t $SBBAND "${T1K_SB[@]}" RPDE_SHIP_JOINT_GAIN=True "RPDE_SHIP_EBAND=$EBAND" "RPDE_SHIP_SIGPOW=$SIGPOW" RPDE_SHIP_D1C_N0=30.0 ;;
    # same without JOINT_GAIN (one lever fewer)
    k27finnj_a080)    build $t $SBBAND "${T1K_SB[@]}" "RPDE_SHIP_EBAND=$EBAND" "RPDE_SHIP_SIGPOW=$SIGPOW" ;;
    k27finnj_d1c_a080) build $t $SBBAND "${T1K_SB[@]}" "RPDE_SHIP_EBAND=$EBAND" "RPDE_SHIP_SIGPOW=$SIGPOW" RPDE_SHIP_D1C_N0=30.0 ;;
    # shipped band (T2 fallback if SIGPOW does not pay under R1): latency + JOINT_GAIN only
    k27shp_a080)      build $t $SHIPBAND "${T1K_27[@]}" RPDE_SHIP_JOINT_GAIN=True ;;
    k27shp_d1c_a080)  build $t $SHIPBAND "${T1K_27[@]}" RPDE_SHIP_JOINT_GAIN=True RPDE_SHIP_D1C_N0=30.0 ;;
    *) echo "unknown tag $t"; continue ;;
  esac
done
for t in $TAGS; do
  z=track2_$t
  stamp "validate_t2 --adaptive $z"
  $PY Track_2/validate_t2.py --adaptive --zip $OUT/$z.zip --extract /tmp/rp_k27fin_$z 2>&1 | tail -25
  stamp "stream_eval --causal-check $z"
  $PY Track_2/stream_eval.py --submission /tmp/rp_k27fin_$z --causal-check 2>&1 | tail -12
done
cd $OUT && for t in $TAGS; do sha256sum track1_$t.zip track2_$t.zip; done
stamp "ALL DONE"
echo "RP_K27_FINAL_END"
