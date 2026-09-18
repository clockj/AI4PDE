#!/bin/bash
# Rebuild both packages of record from the released data. READ REPRODUCE.md FIRST.
#
#   RPDE_REPRODUCE_GO=1 bash reproduce.sh [t1|t2|both]        # default: both
#
# ~43 A100-hours for `both` (29.5 for t1 alone, 24.6 for t2 alone; REPRODUCE.md section 5).
# It runs the stages SEQUENTIALLY in one process and is meant to be launched from a single
# long job, or split across the 4-job cap by running the `train_*` functions separately.
#
# NEVER RUN: without the guard below, so it cannot be started by a stray shell.
# NEVER BEEN RUN end to end. This is REPRODUCE.md written as a script, not a tested pipeline.
#
# Every stage SKIPS if its output already exists, so it is restartable. Delete the .pth to redo
# one member. Checkpoints go to Track_1/unet/ -- the documented exception to "no generated trees
# under $HOME" (CLAUDE.md), a few large files. Everything else goes to scratch.
set -uo pipefail
cd "$(dirname "$0")"

[ "${RPDE_REPRODUCE_GO:-0}" = "1" ] || {
  echo "REFUSING: this is a ~43 A100-hour pipeline. Re-run with RPDE_REPRODUCE_GO=1."
  echo "See REPRODUCE.md. It needs a GPU node -- run it from an sbatch job, not a login shell."
  exit 1; }

PY=${PY:-/users/PAS1187/qwu/.conda/envs/realpde/bin/python}
RPDE=${RPDE:-/users/PAS1187/qwu/RealPDE}
REAL=$RPDE/train_real/train_real
SIM=$RPDE/train_sim/train_sim
U=Track_1/unet
P2=${P2:-$U}                 # where the three F-FNO twins live; see REPRODUCE.md 6.2
WHICH=${1:-both}
export HDF5_USE_FILE_LOCKING=FALSE

for d in "$REAL" "$SIM"; do [ -d "$d" ] || { echo "MISSING data dir $d"; exit 1; }; done
$PY $U/train2.py --help > /dev/null || { echo "PREFLIGHT FAILED: train2.py"; exit 1; }
echo "reproduce.sh: $WHICH   data $RPDE   git $(git rev-parse --short HEAD 2>/dev/null || echo '?')"

# --- recipe fragments, read back from the shipped checkpoints' config (REPRODUCE.md section 1) ---
LOSS="--loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 \
      --lam-v 0.15 --lam-t 0.15 --alpha-train 0.0"
OPT="--stride 5 --batch 32 --lr 2e-4 --epochs 400 --wd 1e-5"
FNO="--arch fno --width 96 --n-layers 8 --modes-y 8 --modes-x 12 --dropout 0.45 \
     --var-head --var-mlp --lift-mlp $LOSS $OPT"
FFNO="--arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32 --dropout 0.3 \
      --var-head $LOSS $OPT"
VITC="--arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 --dropout 0.3 \
      --var-head $LOSS $OPT"
VITP="--arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 --dropout 0.3 --var-head \
      --loss mse --lam-f 1.0 --lam-grad 1.0 --lam-v 0.3 --lam-t 0.15 --alpha-train 0.0 $OPT"
SIMF="--sim $SIM --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5"

run() {                      # run <out.pth> <flags...>
  local out="$1"; shift
  if [ -f "$out" ]; then echo "  SKIP  $(basename "$out") (exists)"; return 0; fi
  echo "  TRAIN $(basename "$out")  $(date '+%F %T')"
  ( cd "$U" && $PY train2.py --data "$REAL" "$@" --out "$(basename "$out")" ) \
      > "${out%.pth}.trainlog" 2>&1
  local rc=$?
  [ -f "$out" ] || { echo "  FAILED $out (rc $rc); see ${out%.pth}.trainlog"; exit 1; }
  tail -2 "${out%.pth}.trainlog"
}

# ============================== stage 1: TWINS (the band's error field) =====================
# First on purpose: a package whose band came from all-data models scored sps 36.94 (R6), so the
# twins are the part you cannot skip. Held out = model.VAL_TRAJ (no --exclude, no --all-data).
twins_t1() {
  echo "== twins, Track 1 (5) =="
  for s in 1 2 3; do
    o=$U/fno_r18_cmb3sim$([ "$s" = 1 ] || echo _s$s).pth
    run "$o" $FNO --seed $s $SIMF --sim-exclude-held
  done
  run $U/arch_vit_d384.pth    $VITP --seed 1
  run $U/arch_vit_cmb3sim.pth $VITC --seed 1 $SIMF --sim-exclude-held
}
twins_t2() {
  echo "== twins, Track 2 (3 F-FNO; the two ViT twins are shared with Track 1) =="
  for s in 1 2 3; do run $P2/p2_ffno_l8_s$s.pth $FFNO --seed $s; done
}

# ============================== stage 2: ALL-DATA MEMBERS (what ships) ======================
members_t1() {
  echo "== all-data members, Track 1 (6) =="
  for s in 1 2 3 4; do
    run $U/fno_all_w96L8cmb3sim_s$s.pth $FNO --seed $s --all-data $SIMF
  done
  run $U/arch_vit_all_s1.pth           $VITP --seed 1 --all-data
  run $U/arch_vit_all_cmb3sim_s3.pth   $VITC --seed 3 --all-data $SIMF
}
members_t2() {
  echo "== all-data members, Track 2 (4 F-FNO; the two ViTs are shared with Track 1) =="
  for s in 1 2 3 4; do run $U/ffno_all_w40l8_s$s.pth $FFNO --seed $s --all-data; done
}

# ============================== stage 3 + 4: BAND, BUILD, GATES =============================
# build_round13.sh does the band and both tracks' archives in one pass. RPDE_BAND reuses a
# committed band instead of refitting -- which is what a reproduction of a SCORED package must
# do, because a refit is not bit-reproducible (Track_1/bands/MANIFEST.md).
build_t1() {
  echo "== band + build, Track 1 (R25 xfam6fp16) =="
  local A="$U/fno_all_w96L8cmb3sim_s1.pth $U/fno_all_w96L8cmb3sim_s2.pth \
           $U/fno_all_w96L8cmb3sim_s3.pth $U/fno_all_w96L8cmb3sim_s4.pth \
           $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
  local M="$U/fno_r18_cmb3sim.pth $U/fno_r18_cmb3sim_s2.pth $U/fno_r18_cmb3sim_s3.pth \
           $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"
  RPDE_FORCE=1 RPDE_A="$A" RPDE_M="$M" \
    RPDE_BAND="${RPDE_BAND_T1:-Track_1/bands/r24_xfam6.npz}" \
    RPDE_INT8=1 RPDE_LINEAR=tensor RPDE_SHIP_AMP='"fp16"' RPDE_SPEC_PATH=bmm \
    RPDE_HOST_WRITE=torch RPDE_TAG="r25_xfam6fp16_a060" \
    bash Track_1/submit_fno/build_round13.sh 0.60
}
build_t2() {
  echo "== band + build, Track 2 (T2R13 P2-L8) =="
  # FFNO_PATH must be baked dense on BOTH tracks first. build_round13.sh does NOT do this --
  # REPRODUCE.md 6.4. The script that did it is not in the repo; until it is, edit the two
  # submission.py files by hand and check the value back out of the archive.
  grep -H '^FFNO_PATH' Track_1/submit_fno/submission.py Track_2/submit/submission.py || true
  echo "  ^ both must read FFNO_PATH = \"dense\" before this build (REPRODUCE.md 6.4)"
  local A="$U/ffno_all_w40l8_s1.pth $U/ffno_all_w40l8_s2.pth $U/ffno_all_w40l8_s3.pth \
           $U/ffno_all_w40l8_s4.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth"
  local M="$P2/p2_ffno_l8_s1.pth $P2/p2_ffno_l8_s2.pth $P2/p2_ffno_l8_s3.pth \
           $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth"
  RPDE_FORCE=1 RPDE_A="$A" RPDE_M="$M" \
    ${RPDE_BAND_T2:+RPDE_BAND="$RPDE_BAND_T2"} \
    RPDE_INT8=1 RPDE_LINEAR=tensor RPDE_SHIP_AMP='"fp16"' RPDE_SPEC_PATH=bmm \
    RPDE_HOST_WRITE=torch RPDE_TAG="r27_p2ffno4_a060" \
    bash Track_1/submit_fno/build_round13.sh 0.60
}

case "$WHICH" in
  t1)   twins_t1;            members_t1;            build_t1 ;;
  t2)   twins_t1; twins_t2;  members_t1; members_t2; build_t2 ;;   # t2 needs T1's four ViTs
  both) twins_t1; twins_t2;  members_t1; members_t2; build_t1; build_t2 ;;
  *) echo "usage: $0 [t1|t2|both]"; exit 2 ;;
esac

echo
echo "DONE. Acceptance is a SCORE test, not a checksum (REPRODUCE.md section 7): the strict gates"
echo "above re-score each archive through the kit's scoring.py, and the four error subscores must"
echo "land inside the seed spread. Then compare against the uploaded archive with"
echo "  $PY Track_1/verify/pkg_identity.py --track t1 --a <extracted old> --b <extracted new>"
