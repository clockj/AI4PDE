#!/bin/bash
#SBATCH -J rp_k273_build
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
# K3: build + gate BOTH tracks from a polished member set. env: ARM, MODE (f|fv), [BAND=<band.npz>], [D1C=1], [TAGX=<suffix>]
#   members  = $K/ckpt/3/$ARM/ffno_all_w40l8_sim_s{1..4}_$ARM.pth + ViT members (original for MODE f, polished for fv)
#   twins    = $K/ckpt/3/$ARM/p2_ffno_l8_sim_s{1..3}_$ARM.pth   + ViT twins   (original for MODE f, polished for fv)
#   band     = BAND if given (e.g. $K/pkg/3/candref_<tag>/band.npz, the shipped construction at the chosen inflate),
#              else refit by build_round13.sh (shipped construction, inflate 1.39)
# Runs ONLY in the build clone $K/wt_3b (build_round13.sh seds constants into submission.py); one build at a time.
set -uo pipefail
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; W=$K/wt_3b; U=/users/PAS1187/qwu/workspace/realpde/main/Track_1/unet
: "${ARM:?}" "${MODE:?}"
cd $W
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
PYB=/users/PAS1187/qwu/.conda/envs/realpde/bin/python; PY="$PYB -u"
OUT=$K/sub_candidates; mkdir -p $OUT $K/pkg/3
T0=$(date +%s); stamp() { echo "[STAGE $(date +%T) +$(( $(date +%s) - T0 ))s] $*"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"; git -C $W log -1 --format='%h' ; git -C $W status --short | head
C=$K/ckpt/3/$ARM
A="$C/ffno_all_w40l8_sim_s1_$ARM.pth $C/ffno_all_w40l8_sim_s2_$ARM.pth $C/ffno_all_w40l8_sim_s3_$ARM.pth $C/ffno_all_w40l8_sim_s4_$ARM.pth"
M="$C/p2_ffno_l8_sim_s1_$ARM.pth $C/p2_ffno_l8_sim_s2_$ARM.pth $C/p2_ffno_l8_sim_s3_$ARM.pth"
if [ "$MODE" = "fv" ]; then A="$A $C/arch_vit_all_s1_$ARM.pth $C/arch_vit_all_cmb3sim_s3_$ARM.pth"; M="$M $C/arch_vit_d384_$ARM.pth $C/arch_vit_cmb3sim_$ARM.pth"
else A="$A $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"; M="$M $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"; fi
for f in $A $M; do [ -f $f ] || { echo "MISSING $f"; exit 2; }; done
COMMON=(RPDE_FORCE=1 RPDE_SUB=$OUT RPDE_BAND_FILL=1 RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor 'RPDE_SHIP_AMP="fp16"'
        RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense)
T1K=(RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=True RPDE_SHIP_SMALL_N=2 RPDE_SHIP_SMALL_GRAPH_LAZY=True)
T2K=(RPDE_SHIP_R1=True RPDE_SHIP_R1_ASYM=True RPDE_SHIP_NUMPY_OUT=True)
[ "${D1C:-0}" = "1" ] && T2K+=(RPDE_SHIP_D1C_N0=30.0)
TAG=k273_${ARM}_${MODE}${TAGX:-}_a080; [ "${D1C:-0}" = "1" ] && TAG=k273_${ARM}_${MODE}${TAGX:-}_d1c_a080
restore() { git -C $W checkout -- Track_1/submit_fno/submission.py Track_2/submit/submission.py Track_1/submit_fno/_fno.py; }
restore
stamp "build $TAG  (BAND=${BAND:-refit})"
env "${COMMON[@]}" "${T1K[@]}" "${T2K[@]}" ${BAND:+RPDE_BAND=$BAND} RPDE_A="$A" RPDE_M="$M" RPDE_TAG=$TAG \
    bash Track_1/submit_fno/build_round13.sh 0.80 2>&1
cp Track_2/submit/band8.npz $K/pkg/3/band_$TAG.npz
restore
stamp "validate_t2 --adaptive"
$PY Track_2/validate_t2.py --adaptive --zip $OUT/track2_$TAG.zip --extract /tmp/rp_k273_v_$TAG 2>&1 | tail -25
stamp "stream_eval --causal-check"
$PY Track_2/stream_eval.py --submission /tmp/rp_k273_v_$TAG --causal-check 2>&1 | tail -12
cd $OUT && sha256sum track1_$TAG.zip track2_$TAG.zip
stamp "ALL DONE"; echo RP_K273_BUILD_END
