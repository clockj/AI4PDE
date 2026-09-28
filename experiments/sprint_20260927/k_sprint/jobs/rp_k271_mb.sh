#!/bin/bash
#SBATCH -J rp_k271_mb
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
# E1 phase 3: merged builds. env W (clone), TAGS.
set -uo pipefail
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
cd $W
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PYB=/users/PAS1187/qwu/.conda/envs/realpde/bin/python; PY="$PYB -u"
OUT=$K/sub_candidates; T0=$(date +%s)
stamp() { echo "[STAGE $(date +%T) +$(( $(date +%s) - T0 ))s] $*"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-} W=$W"; git -C $W log -1 --format='%h %s' | cut -c1-100; git -C $W status --short | head -3
SHIPBAND=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/x_work/band_r38_a080_c0_fill.npz
SBBAND=$K/pkg/k0/band_k0_sigblend_lam0.5.npz
EBAND=$($PYB -c "import json;print(json.load(open('$K/pkg/k0/band_k0_sigblend_lam0.5.json'))['EBAND'])")
SIGPOW=$($PYB -c "import json;q=json.load(open('$K/pkg/k0/band_k0_sigblend_lam0.5.json'))['T2_sigpow'];print('(%r, %r)' % (float(q[0]), float(q[1])))")
echo "EBAND $EBAND SIGPOW $SIGPOW"
U=Track_1/unet
M="$U/p2_ffno_l8_sim_s1.pth $U/p2_ffno_l8_sim_s2.pth $U/p2_ffno_l8_sim_s3.pth $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"
A="$U/ffno_all_w40l8_sim_s1.pth $U/ffno_all_w40l8_sim_s2.pth $U/ffno_all_w40l8_sim_s3.pth $U/ffno_all_w40l8_sim_s4.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
COMMON=(RPDE_FORCE=1 RPDE_SUB=$OUT RPDE_BAND_FILL=1 RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor 'RPDE_SHIP_AMP="fp16"'
        RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense)
T1K_SB=(RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=False RPDE_SHIP_SMALL_GRAPH_LAZY=False)
T1K_27=(RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=True RPDE_SHIP_SMALL_N=2 RPDE_SHIP_SMALL_GRAPH_LAZY=True)
T2K=(RPDE_SHIP_R1=True RPDE_SHIP_R1_ASYM=True RPDE_SHIP_NUMPY_OUT=True)
LAT=(RPDE_SHIP_WARM_TAILS=True RPDE_SHIP_PIN_OUT=128 RPDE_SHIP_BATCH_CAP=56 RPDE_SHIP_ARENA_STEPS=1024)
OURS=("${T1K_SB[@]}" "${LAT[@]}" RPDE_SHIP_JOINT_GAIN=True "RPDE_SHIP_EBAND=$EBAND" "RPDE_SHIP_SIGPOW=$SIGPOW")
restore() { git -C $W checkout -- Track_1/submit_fno/submission.py Track_2/submit/submission.py Track_1/submit_fno/_fno.py; }
build() {  # tag members band extra...
  local tag=$1 mem=$2 band=$3; shift 3
  stamp "build $tag"
  env "${COMMON[@]}" "${T2K[@]}" RPDE_BAND=$band "$@" RPDE_A="$mem" RPDE_M="$M" RPDE_TAG=$tag \
      bash Track_1/submit_fno/build_round13.sh 0.80 > $K/logs/build_$tag.log 2>&1
  grep -a "SUBMITTABLE\|rpde-t2\] load:\|REFUS\|Error\|error" $K/logs/build_$tag.log | cut -c1-80,200-520 | head -12
  grep -H "^JOINT_GAIN = \|^ARENA_STEPS = \|^SIGPOW = \|^D1C_N0 = \|^VC_N0 = \|^FW_N0 = \|^R1 = \|^R1_ASYM = \|^NUMPY_OUT = \|^ALPHA = " Track_2/submit/submission.py | cut -c1-110
  grep -H "^EBAND = \|^SMALL_GRAPH = \|^JOINT_GAIN = \|^PIN_OUT = " Track_1/submit_fno/submission.py | cut -c1-110
  restore
  stamp "build $tag done"
}
for t in $TAGS; do
  case $t in
    k27mrg_eqA)        build $t "$A" $SHIPBAND "${T1K_27[@]}" RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_VC_N0=50.0 ;;
    k27mrg_eqB)        build $t "$A" $SBBAND "${OURS[@]}" RPDE_SHIP_D1C_N0=30.0 ;;
    k27mrg_cons_a080)  build $t "$A" $SBBAND "${OURS[@]}" RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_VC_N0=50.0 ;;
    k27mrg_aggr_a080)  build $t "$A" $SBBAND "${OURS[@]}" RPDE_SHIP_D1C_N0=10.0 RPDE_SHIP_VC_N0=50.0 ;;
    k27mrg_consnj_a080) build $t "$A" $SBBAND "${T1K_SB[@]}" "${LAT[@]}" "RPDE_SHIP_EBAND=$EBAND" "RPDE_SHIP_SIGPOW=$SIGPOW" RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_VC_N0=50.0 ;;
    tw_R0)    build $t "$M" $SHIPBAND "${T1K_27[@]}" RPDE_SHIP_D1C_N0=30.0 ;;
    tw_R1)    build $t "$M" $SHIPBAND "${T1K_27[@]}" RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_VC_N0=50.0 ;;
    tw_R2)    build $t "$M" $SHIPBAND "${T1K_27[@]}" "${LAT[@]}" RPDE_SHIP_JOINT_GAIN=True RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_VC_N0=50.0 ;;
    tw_R3)    build $t "$M" $SBBAND "${OURS[@]}" RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_VC_N0=50.0 ;;
    tw_R4)    build $t "$M" $SHIPBAND "${T1K_27[@]}" RPDE_SHIP_D1C_N0=10.0 RPDE_SHIP_VC_N0=50.0 ;;
    tw_R5)    build $t "$M" $SBBAND "${OURS[@]}" RPDE_SHIP_D1C_N0=10.0 RPDE_SHIP_VC_N0=50.0 ;;
    tw_R3nj)  build $t "$M" $SBBAND "${T1K_SB[@]}" "${LAT[@]}" "RPDE_SHIP_EBAND=$EBAND" "RPDE_SHIP_SIGPOW=$SIGPOW" RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_VC_N0=50.0 ;;
    tw_R0add) build $t "$M" $SBBAND "${OURS[@]}" RPDE_SHIP_D1C_N0=30.0 ;;
    tw_R1j)   build $t "$M" $SHIPBAND "${T1K_27[@]}" RPDE_SHIP_JOINT_GAIN=True RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_VC_N0=50.0 ;;
    *) echo "unknown tag $t" ;;
  esac
done
cd $OUT && for t in $TAGS; do sha256sum track2_$t.zip; done
stamp "ALL DONE"; echo RP_K271_MB_END
