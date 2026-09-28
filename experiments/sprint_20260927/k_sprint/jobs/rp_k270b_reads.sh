#!/bin/bash
#SBATCH -J rp_k270b_reads
#SBATCH -A PAS1186
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
set -uo pipefail
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; W=$K/wt_0b; cd $W
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PYB=/users/PAS1187/qwu/.conda/envs/realpde/bin/python; PY="$PYB -u"
O=$K/pkg/k0b; D=$K/dumps/k0b; mkdir -p $O/sub_ref $D
T0=$(date +%s); stamp() { echo "[STAGE $(date +%T) +$(( $(date +%s) - T0 ))s] $*"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"; git log -1 --oneline
U=Track_1/unet
M="$U/p2_ffno_l8_sim_s1.pth $U/p2_ffno_l8_sim_s2.pth $U/p2_ffno_l8_sim_s3.pth $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"
A="$U/ffno_all_w40l8_sim_s1.pth $U/ffno_all_w40l8_sim_s2.pth $U/ffno_all_w40l8_sim_s3.pth $U/ffno_all_w40l8_sim_s4.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
BA=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/pkg/x_work/band_r38_a080_c0_fill.npz
B3=$K/pkg/k0/band_k0_sigblend_lam0.3.npz; B5=$K/pkg/k0/band_k0_sigblend_lam0.5.npz
SP3='(0.14539535293810696, 0.14721409330440394)'; SP5='(0.24232558823017827, 0.24535682217400656)'
COMMON=(RPDE_FORCE=1 RPDE_SUB=$O/sub_ref RPDE_BAND_FILL=1 RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor 'RPDE_SHIP_AMP="fp16"'
        RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense)
T1K=(RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=True RPDE_SHIP_SMALL_N=2 RPDE_SHIP_SMALL_GRAPH_LAZY=True)
T2K=(RPDE_SHIP_R1=True RPDE_SHIP_R1_ASYM=True RPDE_SHIP_NUMPY_OUT=True)
restore() { git -C $W checkout -- Track_1/submit_fno/submission.py Track_2/submit/submission.py Track_1/submit_fno/_fno.py; }
stamp "sed asnp constants via build_round13 (band reused, SIGPOW off)"
env "${COMMON[@]}" "${T1K[@]}" "${T2K[@]}" RPDE_BAND=$BA RPDE_A="$A" RPDE_M="$M" RPDE_TAG=k0b_ref \
    bash Track_1/submit_fno/build_round13.sh 0.80 > $K/logs/rp_k270b_refbuild.log 2>&1
grep -n "SUBMITTABLE" $K/logs/rp_k270b_refbuild.log | tail -3
for b in a:$BA l3:$B3 l5:$B5; do
  n=${b%%:*}; f=${b#*:}
  $PY Track_2/build_t2.py --band $f --int8 --linear tensor --ckpt $M --stage $O/stage_$n --out $O/t2twin_$n.zip 2>&1 | tail -2
done
grep -H "^ALPHA = \|^R1 = \|^R1_ASYM = \|^NUMPY_OUT = \|^D1C_N0 = \|^AMP = \|^FFNO_PATH = \|^SPEC_PATH = \|^SIGPOW = " $O/stage_a/submission.py
restore; git status --short | head
stamp "stream reads"
for n in a l3 l5; do for v in static asym asym_d1c; do
  d=/tmp/rp_k270b_${n}_$v; rm -rf $d; mkdir -p $d; $PYB -m zipfile -e $O/t2twin_$n.zip $d
  case $v in
    static)   sed -i "s/^R1 = .*/R1 = False/; s/^R1_ASYM = .*/R1_ASYM = False/" $d/submission.py ;;
    asym)     : ;;
    asym_d1c) sed -i "s/^D1C_N0 = [^ ]*/D1C_N0 = 30.0/" $d/submission.py ;;
  esac
  [ $n = l3 ] && sed -i "s|^SIGPOW = None |SIGPOW = $SP3 |" $d/submission.py
  [ $n = l5 ] && sed -i "s|^SIGPOW = None |SIGPOW = $SP5 |" $d/submission.py
  grep -H "^R1 = \|^R1_ASYM = \|^D1C_N0 = \|^SIGPOW = " $d/submission.py | cut -c1-120
  stamp "stream $n $v"
  $PY Track_2/stream_eval.py --submission $d --save $D/s_${n}_$v.npz 2>&1 | grep -v "^\s*$" | tail -6
done; done
stamp "unpatched control (static, asym) on band a"
for v in static asym; do
  d=/tmp/rp_k270b_unp_$v; rm -rf $d; cp -r /tmp/rp_k270b_a_$v $d; $PYB $K/jobs/k0b/unpatch.py $d/submission.py
  $PY Track_2/stream_eval.py --submission $d --save $D/s_unp_$v.npz 2>&1 | tail -3
done
$PYB - <<PYEOF
import numpy as np
D="$D"
for v in ("static","asym"):
    a=np.load(f"{D}/s_a_{v}.npz"); b=np.load(f"{D}/s_unp_{v}.npz")
    print(f"knob-off vs unpatched [{v}]: " + " ".join(f"{k} {np.array_equal(a[k],b[k])}" for k in ("pred","lower","upper")), len(a["pred"]), "steps")
for n in ("l3","l5"):
  for v in ("static","asym","asym_d1c"):
    a=np.load(f"{D}/s_a_{v}.npz"); b=np.load(f"{D}/s_{n}_{v}.npz"); P,L,U=b["pred"],b["lower"],b["upper"]
    print(f"knob-on [{n} {v}]: pred==off {np.array_equal(a['pred'],P)}  finite {bool(np.isfinite(L).all() and np.isfinite(U).all() and np.isfinite(P).all())}  "
          f"L<=P {bool((L<=P).all())} P<=U {bool((P<=U).all())}  width uv mean {float((U-L)[...,:2].mean()):.5f} (off {float((a['upper']-a['lower'])[...,:2].mean()):.5f})")
PYEOF
stamp "score"
$PY $K/jobs/k0b/score_saves.py a_static a_static=$D/s_a_static.npz l3_static=$D/s_l3_static.npz l5_static=$D/s_l5_static.npz
$PY $K/jobs/k0b/score_saves.py a_asym a_asym=$D/s_a_asym.npz l3_asym=$D/s_l3_asym.npz l5_asym=$D/s_l5_asym.npz
$PY $K/jobs/k0b/score_saves.py a_asym_d1c a_asym_d1c=$D/s_a_asym_d1c.npz l3_asym_d1c=$D/s_l3_asym_d1c.npz l5_asym_d1c=$D/s_l5_asym_d1c.npz
stamp "verifier (static, SIGPOW on)"
for n in l5 l3; do $PY $K/jobs/k0b/verify_sigpow.py /tmp/rp_k270b_${n}_static $([ $n = l5 ] && echo $B5 || echo $B3) 2>&1 | grep -v "^\s*$" | tail -5; done
stamp "causal check on twin asym+sigpow l5, d1c l5"
for v in asym asym_d1c; do $PY Track_2/stream_eval.py --submission /tmp/rp_k270b_l5_$v --causal-check 2>&1 | grep "CAUSAL\|load:" | cut -c1-400; done
stamp "ALL DONE"; echo RP_K270B_READS_END
