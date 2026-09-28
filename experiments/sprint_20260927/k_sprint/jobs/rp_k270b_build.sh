#!/bin/bash
#SBATCH -J rp_k270b_build
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
# Builds 4 staging packages (lam 0.3 / 0.5 x asnp / asnp+D1c) with the all-data members into $K/pkg/k0b/stage_sub;
# the chosen pair is COPIED to $K/sub_candidates afterwards under tags k27sbt2_a080 / k27sbt2_d1c_a080 (T2 only).
set -uo pipefail
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; W=$K/wt_0b2; cd $W
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PYB=/users/PAS1187/qwu/.conda/envs/realpde/bin/python; PY="$PYB -u"
OUT=$K/pkg/k0b/stage_sub; mkdir -p $OUT
T0=$(date +%s); stamp() { echo "[STAGE $(date +%T) +$(( $(date +%s) - T0 ))s] $*"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"; git log -1 --oneline
U=Track_1/unet
M="$U/p2_ffno_l8_sim_s1.pth $U/p2_ffno_l8_sim_s2.pth $U/p2_ffno_l8_sim_s3.pth $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"
A="$U/ffno_all_w40l8_sim_s1.pth $U/ffno_all_w40l8_sim_s2.pth $U/ffno_all_w40l8_sim_s3.pth $U/ffno_all_w40l8_sim_s4.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
COMMON=(RPDE_FORCE=1 RPDE_SUB=$OUT RPDE_BAND_FILL=1 RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor 'RPDE_SHIP_AMP="fp16"'
        RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense)
T1K=(RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=True RPDE_SHIP_SMALL_N=2 RPDE_SHIP_SMALL_GRAPH_LAZY=True)
T2K=(RPDE_SHIP_R1=True RPDE_SHIP_R1_ASYM=True RPDE_SHIP_NUMPY_OUT=True)
restore() { git -C $W checkout -- Track_1/submit_fno/submission.py Track_2/submit/submission.py Track_1/submit_fno/_fno.py; }
for lam in 0.5 0.3; do
  if [ $lam = 0.5 ]; then SP='(0.24232558823017827, 0.24535682217400656)'; L=l5; else SP='(0.14539535293810696, 0.14721409330440394)'; L=l3; fi
  B=$K/pkg/k0/band_k0_sigblend_lam$lam.npz
  for d1c in 0 1; do
    TAG=k27sbt2_${L}$([ $d1c = 1 ] && echo _d1c)_a080
    X=(); [ $d1c = 1 ] && X=(RPDE_SHIP_D1C_N0=30.0)
    stamp "build $TAG"
    env "${COMMON[@]}" "${T1K[@]}" "${T2K[@]}" "${X[@]}" RPDE_SHIP_SIGPOW="$SP" RPDE_BAND=$B RPDE_A="$A" RPDE_M="$M" RPDE_TAG=$TAG \
      bash Track_1/submit_fno/build_round13.sh 0.80 > $K/logs/rp_k270b_build_$TAG.log 2>&1
    rm -f $OUT/track1_$TAG.zip          # T1 zip carries the SIGBLEND npz without EBAND: ~3x too wide, invalid
    grep -H "^SIGPOW = \|^D1C_N0 = \|^R1 = \|^R1_ASYM = \|^ALPHA = " Track_2/submit/submission.py | cut -c1-140
    restore
    grep -n "SUBMITTABLE" $K/logs/rp_k270b_build_$TAG.log | tail -2
    stamp "gate $TAG"
    $PY Track_2/validate_t2.py --adaptive --zip $OUT/track2_$TAG.zip --extract /tmp/rp_k270b_v_$TAG 2>&1 | grep -v "^\s*$" | tail -12
    grep -H "^SIGPOW = \|^D1C_N0 = " /tmp/rp_k270b_v_$TAG/submission.py | cut -c1-120
    cmp $B /tmp/rp_k270b_v_$TAG/band.npz && echo "band.npz == $B"
    $PY Track_2/stream_eval.py --submission /tmp/rp_k270b_v_$TAG --causal-check 2>&1 | grep "CAUSAL\|load:\|scores\|errors" | cut -c1-500
    sha256sum $OUT/track2_$TAG.zip
  done
done
ls -la $OUT; stamp "ALL DONE"; echo RP_K270B_BUILD_END
