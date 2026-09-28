#!/bin/bash
#SBATCH -J rp_k270_ship2
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
set -uo pipefail
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; W=$K/wt_0a; cd $W
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PY="/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date +%T)"
U=Track_1/unet
TW="$U/p2_ffno_l8_sim_s1.pth $U/p2_ffno_l8_sim_s2.pth $U/p2_ffno_l8_sim_s3.pth $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"
AD="$U/ffno_all_w40l8_sim_s1.pth $U/ffno_all_w40l8_sim_s2.pth $U/ffno_all_w40l8_sim_s3.pth $U/ffno_all_w40l8_sim_s4.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
COMMON=(RPDE_FORCE=1 RPDE_BAND_FILL=1 RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor 'RPDE_SHIP_AMP="fp16"' RPDE_SPEC_PATH=bmm
        RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=False RPDE_SHIP_SMALL_N=2
        RPDE_SHIP_SMALL_GRAPH_LAZY=True RPDE_SHIP_WARM_TAILS=True)
SHIPB=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/x_work/band_r38_a080_c0_fill.npz
EB05=$(python3 -c "import json;print(json.load(open('$K/pkg/k0/band_k0_sigblend_lam0.5.json'))['EBAND'])")
EB03=$(python3 -c "import json;print(json.load(open('$K/pkg/k0/band_k0_sigblend_lam0.3.json'))['EBAND'])")
restore() { git checkout -- Track_1/submit_fno/submission.py Track_2/submit/submission.py Track_1/submit_fno/_fno.py; }
build() {  # tag members sub band eband
  local tag=$1 mem=$2 sub=$3 band=$4 eb=$5
  echo "=== BUILD $tag $(date +%T)"
  if [ -n "$eb" ]; then
    env "${COMMON[@]}" RPDE_SUB=$sub RPDE_A="$mem" RPDE_M="$TW" RPDE_BAND=$band RPDE_SHIP_EBAND="$eb" RPDE_TAG=$tag \
      bash Track_1/submit_fno/build_round13.sh 0.80 > $K/logs/k0a_build_$tag.log 2>&1
  else
    env "${COMMON[@]}" RPDE_SUB=$sub RPDE_A="$mem" RPDE_M="$TW" RPDE_BAND=$band RPDE_TAG=$tag \
      bash Track_1/submit_fno/build_round13.sh 0.80 > $K/logs/k0a_build_$tag.log 2>&1
  fi
  grep -H "EBAND baked\|SUBMITTABLE\|^ALPHA = \|REUSING band\|\[rpde-t1\] load" $K/logs/k0a_build_$tag.log | head -20
  grep -n "^EBAND = \|^SMALL_GRAPH = \|^WARM_TAILS = \|^ALPHA = \|^GPU_BAND = " Track_1/submit_fno/submission.py
  restore
  rm -f $sub/track2_$tag.zip
}
echo "HEAD $(git log -1 --format=%h)"
S=$K/sub_candidates
build k27sb05_a080 "$AD" $S $K/pkg/k0/band_k0_sigblend_lam0.5.npz "$EB05"
build k27sb03_a080 "$AD" $S $K/pkg/k0/band_k0_sigblend_lam0.3.npz "$EB03"
ls -la $S/track*_k27sb0* ; sha256sum $S/track1_k27sb0*.zip
for t in k27sb05_a080 k27sb03_a080; do unzip -p $S/track1_$t.zip _fno.py | grep -c 'eb_mode == "mult" and eb_clip is None'; done
declare -A Z=([sb05]=$K/sub_candidates/track1_k27sb05_a080.zip [sb03]=$K/sub_candidates/track1_k27sb03_a080.zip [base]=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/sub_candidates/track1_x27_a080.zip)
for t in sb05 sb03 base; do rm -rf /tmp/k0s_$t; mkdir -p /tmp/k0s_$t; unzip -q ${Z[$t]} -d /tmp/k0s_$t; done
for t in base sb05 sb03; do echo "=== DUMP $t $(date +%T)"; $PY Track_1/verify/k0a_dump.py /tmp/k0s_$t $K/dumps/k0a/ship_$t.npz 2>&1 | grep -v Warning | tail -3; done
for t in sb05 sb03; do echo "=== EDGE seq ship $t $(date +%T)"; $PY Track_1/verify/k0a_edge.py /tmp/k0s_$t seq 2>&1 | grep -v "Warning\|call2"; done
for t in sb05 sb03; do for n in 1 13 14 99; do $PY Track_1/verify/k0a_edge.py /tmp/k0s_$t one $n float64 2>&1 | grep "N=" | sed "s/^/$t /"; done; done
echo "=== DONE $(date +%T)"
