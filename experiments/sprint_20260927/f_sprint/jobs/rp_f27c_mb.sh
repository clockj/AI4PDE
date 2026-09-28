#!/bin/bash
#SBATCH -J rp_f27c_mb
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927; K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
W=${W:-$S/wt_mrgp2}; cd $W
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PYB=/users/PAS1187/qwu/.conda/envs/realpde/bin/python; PY="$PYB -u"
OUT=${OUT:-$S/pkg/c_work3}; mkdir -p $OUT; T0=$(date +%s)
stamp() { echo "[STAGE $(date +%T) +$(( $(date +%s) - T0 ))s] $*"; }
SHIPBAND=$S/pkg/x_work/band_r38_a080_c0_fill.npz
SBBAND=$K/pkg/k0/band_k0_sigblend_lam0.5.npz
EBAND=$($PYB -c "import json;print(json.load(open('$K/pkg/k0/band_k0_sigblend_lam0.5.json'))['EBAND'])")
SIGPOW=$($PYB -c "import json;q=json.load(open('$K/pkg/k0/band_k0_sigblend_lam0.5.json'))['T2_sigpow'];print('(%r, %r)' % (float(q[0]), float(q[1])))")
echo "EBAND $EBAND SIGPOW $SIGPOW"; cmp $SHIPBAND $S/pkg/fallback/band_r38_a080_c0_fill.npz && echo shipband==fallback
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
NJ=("${T1K_SB[@]}" "${LAT[@]}" "RPDE_SHIP_EBAND=$EBAND" "RPDE_SHIP_SIGPOW=$SIGPOW")
P2K=(RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_D1C_EB=True RPDE_SHIP_D1C_EB_N0=1.0 RPDE_SHIP_D1C_LEAD_N0=5.0 RPDE_SHIP_VC_N0=30.0 RPDE_SHIP_R1_TSM=3 RPDE_SHIP_R1_BLK_N0=20.0 RPDE_SHIP_R1_BLK=4)
cp Track_2/submit/submission.py $OUT/sub_orig.py; cp Track_1/submit_fno/submission.py $OUT/t1_orig.py; cp Track_1/submit_fno/_fno.py $OUT/fno_orig.py
restore() { cp $OUT/sub_orig.py Track_2/submit/submission.py; cp $OUT/t1_orig.py Track_1/submit_fno/submission.py; cp $OUT/fno_orig.py Track_1/submit_fno/_fno.py; }
build() { local tag=$1 mem=$2 band=$3; shift 3; stamp "build $tag"
  env "${COMMON[@]}" "${T2K[@]}" RPDE_BAND=$band "$@" RPDE_A="$mem" RPDE_M="$M" RPDE_TAG=$tag bash Track_1/submit_fno/build_round13.sh 0.80 > $S/logs/rp_f27c_mb_$tag.log 2>&1
  grep -H "^JOINT_GAIN = \|^ARENA_STEPS = \|^SIGPOW = \|^D1C_N0 = \|^D1C_EB = \|^D1C_EB_N0 = \|^D1C_LEAD_N0 = \|^VC_N0 = \|^FW_N0 = \|^R1_TSM = \|^R1_BLK_N0 = \|^R1_BLK = \|^SIGEMA = " Track_2/submit/submission.py | cut -c1-90
  restore; }
case ${PHASE:-A} in
A)
build m_cons "$A" $SBBAND "${OURS[@]}" RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_VC_N0=50.0
build m_p2arena "$A" $SHIPBAND "${T1K_27[@]}" "${LAT[@]}" "${P2K[@]}"
build MRGP2 "$A" $SBBAND "${OURS[@]}" "${P2K[@]}"
build MRGP2nj "$A" $SBBAND "${NJ[@]}" "${P2K[@]}"
build tw_mcons "$M" $SBBAND "${OURS[@]}" RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_VC_N0=50.0
build tw_p2 "$M" $SHIPBAND "${T1K_27[@]}" "${P2K[@]}"
build tw_mp2 "$M" $SBBAND "${OURS[@]}" "${P2K[@]}"
build tw_mp2nj "$M" $SBBAND "${NJ[@]}" "${P2K[@]}"
build tw_mp2nosp "$M" $SHIPBAND "${T1K_27[@]}" "${LAT[@]}" RPDE_SHIP_JOINT_GAIN=True "${P2K[@]}"
ls -la $OUT/*.zip; cd $OUT && sha256sum track2_*.zip
stamp "identity"
cd $W/Track_2
ex() { rm -rf $2; mkdir -p $2; $PYB -m zipfile -e $1 $2; }
ex $K/FINAL/track2_k27mrg_cons_a080.zip /tmp/rpc_kcons; ex $OUT/track2_m_cons.zip /tmp/rpc_mcons
ex $S/pkg/c_final/T2_P2_track2_p2_c2b2.zip /tmp/rpc_p2; ex $OUT/track2_m_p2arena.zip /tmp/rpc_mp2a
for d in kcons mcons p2 mp2a; do $PY stream_eval.py --submission /tmp/rpc_$d --save /tmp/rpc_se_$d.npz 2>&1 | grep -a "load:\|scores" | cut -c1-60,150-600; done
$PYB - <<'PE'
import numpy as np
for a,b in (("kcons","mcons"),("p2","mp2a")):
    A=np.load(f"/tmp/rpc_se_{a}.npz"); B=np.load(f"/tmp/rpc_se_{b}.npz")
    for k in ("pred","lower","upper"):
        print("IDENT", a, "vs", b, k, "bit-equal" if np.array_equal(A[k],B[k]) else f"DIFF max {np.abs(A[k]-B[k]).max():.3e}")
PE
for t in MRGP2 MRGP2nj; do stamp "gates $t"; X=/tmp/rpc_v_$t
  $PY validate_t2.py --adaptive --zip $OUT/track2_$t.zip --extract $X 2>&1 | grep -a "load:\|FAIL\|SUBMITTABLE\|finite\|lower <=" | cut -c1-80,150-700
  $PY stream_eval.py --submission $X --causal-check 2>&1 | grep -a "CAUSAL\|scores" | cut -c1-160
  $PY $S/pkg/c_work/t2_edge.py $X $t 1100 $OUT/edge.jsonl 2>&1 | grep -v "^\[rpde" | tail -1 | cut -c1-600
  $PYB -c "
import sys,os,resource; sys.path.insert(0,'$X'); os.chdir('$X'); import submission; m=submission.get_ttt_model('$X','cuda')
import re; print('RSS_after_load_GB', [int(l.split()[1])/1e6 for l in open('/proc/self/status') if l.startswith('VmRSS')][0])" 2>&1 | grep RSS
done
stamp "A DONE"; echo RP_F27C_MB_A_END ;;
B)
C2K=(RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_D1C_EB=True RPDE_SHIP_D1C_EB_N0=1.0 RPDE_SHIP_D1C_LEAD_N0=5.0 RPDE_SHIP_VC_N0=30.0)
build tw_mc2 "$M" $SBBAND "${OURS[@]}" "${C2K[@]}"
build tw_c2 "$M" $SHIPBAND "${T1K_27[@]}" "${C2K[@]}"
build tw_mp2nosp_nj "$M" $SHIPBAND "${T1K_27[@]}" "${LAT[@]}" "${P2K[@]}"
echo RP_F27C_MB_B_END ;;
esac
