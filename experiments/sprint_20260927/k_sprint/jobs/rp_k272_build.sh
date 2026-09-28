#!/bin/bash
#SBATCH -J rp_k272_build
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
set -uo pipefail
W=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/wt_2x; cd $W
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T)"
U=Track_1/unet
A="$U/ffno_all_w40l8_sim_s1.pth $U/ffno_all_w40l8_sim_s2.pth $U/ffno_all_w40l8_sim_s3.pth $U/ffno_all_w40l8_sim_s4.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
M="$U/p2_ffno_l8_sim_s1.pth $U/p2_ffno_l8_sim_s2.pth $U/p2_ffno_l8_sim_s3.pth $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"
BAND=/users/PAS1187/qwu/workspace/realpde/main/Track_1/bands/band_r38_a080_c0_fill.npz
[ -f $BAND ] || BAND=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/x_work/band_r38_a080_c0_fill.npz
echo "band $BAND"
COMMON=(RPDE_FORCE=1 RPDE_SUB=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/sub_candidates RPDE_BAND=$BAND RPDE_BAND_FILL=1 RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor
        'RPDE_SHIP_AMP="fp16"' RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense RPDE_SHIP_JOINT_GAIN=True)
T1K=(RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=True RPDE_SHIP_SMALL_N=2 RPDE_SHIP_SMALL_GRAPH_LAZY=True)
T2K=(RPDE_SHIP_R1=True RPDE_SHIP_R1_ASYM=True RPDE_SHIP_NUMPY_OUT=True)
for TAG in k27jg_a080 k27jg_d1c_a080; do
  X=(); [ $TAG = k27jg_d1c_a080 ] && X=(RPDE_SHIP_D1C_N0=30.0)
  echo "===== BUILD $TAG $(date +%T)"
  env "${COMMON[@]}" "${T1K[@]}" "${T2K[@]}" "${X[@]}" RPDE_A="$A" RPDE_M="$M" RPDE_TAG=$TAG       bash Track_1/submit_fno/build_round13.sh 0.80 2>&1 | tee /fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/logs/build_$TAG.log | grep -E "SUBMITTABLE|JOINT_GAIN|^Track|ALPHA = |D1C_N0|rpde-t1\] load|rpde-t2\] load|REFUS|Error|error" 
  git checkout -- Track_1/submit_fno/submission.py Track_2/submit/submission.py Track_1/submit_fno/_fno.py
  echo "----- T2 --adaptive gate $TAG"
  $PY Track_2/validate_t2.py --adaptive --zip /fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/sub_candidates/track2_$TAG.zip --extract /tmp/v_k272_$TAG 2>&1 | grep -E "SUBMITTABLE|FAIL|load:" 
  echo "----- T2 causal check $TAG"
  $PY Track_2/stream_eval.py --submission /tmp/v_k272_$TAG --causal-check 2>&1 | grep -E "CAUSALITY|load:|errors|scores"
  rm -rf /tmp/v_k272_$TAG
  sha256sum /fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/sub_candidates/track{1,2}_$TAG.zip
done
git status --short; echo "RP_K272_BUILD_END $(date +%T)"
