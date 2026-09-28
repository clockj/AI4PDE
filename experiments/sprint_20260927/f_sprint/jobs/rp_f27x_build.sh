#!/bin/bash
#SBATCH -J rp_f27x_build
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 01:00:00
# Builds the local equivalents of the 09-27 planned packages (T1 27lz, T2 asnp_a080, T2 asnp+D1c)
# in the X1 worktree, gates them, runs the T2 causal check, and builds a TWIN-member T2 package with the
# same constants for the held-out stream reads (static / asym R1 / asym R1 + D1c).
set -uo pipefail
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
W=$S/wt_x
cd $W
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
PYB=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
PY="$PYB -u"
OUT=$S/sub_candidates; mkdir -p $OUT $S/pkg/x_work
T0=$(date +%s)
stamp() { echo "[STAGE $(date +%T) +$(( $(date +%s) - T0 ))s] $*"; }
tsp() { awk '{print strftime("%H:%M:%S"), $0; fflush()}'; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
git -C $W log -1 --format='%h %s' | cut -c1-120

U=Track_1/unet
V="$U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
M="$U/p2_ffno_l8_sim_s1.pth $U/p2_ffno_l8_sim_s2.pth $U/p2_ffno_l8_sim_s3.pth $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"
A="$U/ffno_all_w40l8_sim_s1.pth $U/ffno_all_w40l8_sim_s2.pth $U/ffno_all_w40l8_sim_s3.pth $U/ffno_all_w40l8_sim_s4.pth $V"
# recipe A common knobs (REPRODUCE.md block) -- RPDE_FORCE=1 because sprint rp_f27* jobs are running: timing is NOT valid
COMMON=(RPDE_FORCE=1 RPDE_SUB=$OUT RPDE_BAND_FILL=1 RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor 'RPDE_SHIP_AMP="fp16"'
        RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense)
# T1 27lz: D2 latency path + lazy batch-1 graph for N=1   |   T2 asnp: R1 asym + numpy return (dual stream + graph are defaults)
T1K=(RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=True RPDE_SHIP_SMALL_N=2 RPDE_SHIP_SMALL_GRAPH_LAZY=True)
T2K=(RPDE_SHIP_R1=True RPDE_SHIP_R1_ASYM=True RPDE_SHIP_NUMPY_OUT=True)
restore() { git -C $W checkout -- Track_1/submit_fno/submission.py Track_2/submit/submission.py Track_1/submit_fno/_fno.py; }

stamp "build 1: fit band @0.80 centre0 fill + T1 27lz + T2 asnp (tag x27_a080)"
env "${COMMON[@]}" "${T1K[@]}" "${T2K[@]}" RPDE_A="$A" RPDE_M="$M" RPDE_TAG=x27_a080 \
    bash Track_1/submit_fno/build_round13.sh 0.80 2>&1 | tsp
cp Track_2/submit/band8.npz $S/pkg/x_work/band_r38_a080_c0_fill.npz
stamp "build 1 done"

stamp "twin T2 package (same baked constants as asnp, members = the 5 twins) for held-out stream reads"
$PY Track_2/build_t2.py --band Track_2/submit/band8.npz --int8 --linear tensor --ckpt $M \
    --stage $S/pkg/x_work/t2twin_stage --out $S/pkg/x_work/t2twin_asnp.zip 2>&1 | tail -5
grep -H "^ALPHA = \|^R1 = \|^R1_ASYM = \|^NUMPY_OUT = \|^D1C_N0 = \|^AMP = \|^FFNO_PATH = \|^SPEC_PATH = " $S/pkg/x_work/t2twin_stage/submission.py
restore

stamp "build 2: asnp + D1c (D1C_N0=30), band REUSED"
env "${COMMON[@]}" "${T1K[@]}" "${T2K[@]}" RPDE_SHIP_D1C_N0=30.0 RPDE_BAND=$S/pkg/x_work/band_r38_a080_c0_fill.npz \
    RPDE_A="$A" RPDE_M="$M" RPDE_TAG=x27_d1c_a080 bash Track_1/submit_fno/build_round13.sh 0.80 2>&1 | tsp
restore
stamp "build 2 done"

for z in track2_x27_a080 track2_x27_d1c_a080; do
  stamp "validate_t2 --adaptive $z"
  $PY Track_2/validate_t2.py --adaptive --zip $OUT/$z.zip --extract /tmp/rp_f27x_$z 2>&1 | tsp | tail -25
  stamp "stream_eval --causal-check $z"
  $PY Track_2/stream_eval.py --submission /tmp/rp_f27x_$z --causal-check 2>&1 | tsp | tail -12
done

stamp "held-out T2 stream reads on the twin package (val_unseen; all 8 and AoA 15-20)"
for v in static asym asym_d1c; do
  d=/tmp/rp_f27x_twin_$v; rm -rf $d; mkdir -p $d; $PYB -m zipfile -e $S/pkg/x_work/t2twin_asnp.zip $d
  case $v in
    static)   sed -i "s/^R1 = .*/R1 = False/; s/^R1_ASYM = .*/R1_ASYM = False/" $d/submission.py ;;
    asym)     : ;;
    asym_d1c) sed -i "s/^D1C_N0 = [^ ]*/D1C_N0 = 30.0/" $d/submission.py ;;
  esac
  grep -H "^R1 = \|^R1_ASYM = \|^D1C_N0 = " $d/submission.py
  stamp "stream twin $v all"
  $PY Track_2/stream_eval.py --submission $d 2>&1 | tail -8
  stamp "stream twin $v aoa15-20"
  $PY Track_2/stream_eval.py --submission $d --aoa 15 20 2>&1 | tail -8
done
cd $OUT && sha256sum *.zip
stamp "ALL DONE"
echo "RP_F27X_BUILD_END"
