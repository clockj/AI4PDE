#!/bin/bash
# Rebuild both packages of record from the released data. READ REPRODUCE.md FIRST.
#
#   RPDE_REPRODUCE_GO=1 bash reproduce.sh [t1|t2|both|r25|r28]    # default: both
#
# t1, t2 and both are the method of record (R30 / T2R17, tag r34): ONE member set, both archives.
# r25 and r28 rebuild the superseded packages the board actually scored (REPRODUCE.md section 4).
#
# ~37 A100-hours for `both`: since 09-20 the two tracks ship the SAME member set, so `t1`, `t2`
# and `both` differ only in which archive is built at the end (REPRODUCE.md section 5).
# It runs the stages SEQUENTIALLY in one process and is meant to be launched from a single
# long job, or split across the job cap with RPDE_REPRO_STAGE (below).
#
# NEVER RUN: without the guard below, so it cannot be started by a stray shell.
# Run end to end in a 2-epoch SMOKE configuration on 2026-09-20 (job pz_repro_smoke, Pitzer
# V100); the full-fidelity F-FNO twin stage was re-run at 400 epochs and scored against the
# shipped checkpoint the same day. REPRODUCE.md section 8 is the procedure and the result.
#
# Every stage SKIPS if its output already exists, so it is restartable. Delete the .pth to redo
# one member. Checkpoints go to Track_1/unet/ -- the documented exception to "no generated trees
# under $HOME" (CLAUDE.md), a few large files. Everything else goes to scratch.
#
# Five knobs, all no-ops when unset, so the default path is exactly the pipeline of record:
#   RPDE_REPRO_OUT=<dir>     write (and read back) every checkpoint there instead of Track_1/unet
#   RPDE_REPRO_EPOCHS=<n>    --epochs for every training stage (400 = the recipe). Anything else
#                            is a SMOKE: the guard below then refuses to write into the repo.
#   RPDE_REPRO_STAGE="a b"   run only these stage functions, in this order, and no build
#   RPDE_REPRO_SEEDS="1 2"   narrow every seed loop (default 1 2 3 for twins, 1 2 3 4 for members)
#   RPDE_REPRO_FITBAND=1     FIT the band from the twins (RPDE_BAND_FILL=1) instead of reusing the
#                            committed one -- what a NEW member set needs, and what the smoke
#                            exercises. A refit is not bit-reproducible across twin sets.
set -uo pipefail
cd "$(dirname "$0")"

[ "${RPDE_REPRODUCE_GO:-0}" = "1" ] || {
  echo "REFUSING: this is a ~37 A100-hour pipeline. Re-run with RPDE_REPRODUCE_GO=1."
  echo "See REPRODUCE.md. It needs a GPU node -- run it from an sbatch job, not a login shell."
  exit 1; }

PY=${PY:-/users/PAS1187/qwu/.conda/envs/realpde/bin/python}
RPDE=${RPDE:-/users/PAS1187/qwu/RealPDE}
REAL=$RPDE/train_real/train_real
SIM=$RPDE/train_sim/train_sim
U=Track_1/unet                          # the CODE: train2.py and model.py live here
OUT=${RPDE_REPRO_OUT:-$U}               # the CHECKPOINTS: default = the code dir, as shipped
P2=${P2:-$OUT}               # where the three F-FNO twins live; see REPRODUCE.md 6.2
EPOCHS=${RPDE_REPRO_EPOCHS:-400}
WHICH=${1:-both}
export HDF5_USE_FILE_LOCKING=FALSE

for d in "$REAL" "$SIM"; do [ -d "$d" ] || { echo "MISSING data dir $d"; exit 1; }; done
mkdir -p "$OUT" || exit 1
$PY $U/train2.py --help > /dev/null || { echo "PREFLIGHT FAILED: train2.py"; exit 1; }
echo "reproduce.sh: $WHICH   data $RPDE   out $OUT   epochs $EPOCHS   git $(git rev-parse --short HEAD 2>/dev/null || echo '?')"

# A reduced-epoch run produces checkpoints that are NOT the method of record, and
# build_round13.sh unconditionally stages its two archives into ../sub_candidates (its `SUB` is
# not overridable). So a smoke must run from a COPY of the repository on scratch, never in the
# working tree -- REPRODUCE.md section 8 has the three-line recipe.
if [ "$EPOCHS" != "400" ]; then
  echo "SMOKE MODE: --epochs $EPOCHS. These checkpoints are NOT the method of record."
  if [ "$(cd "$OUT" && pwd)" = "$(pwd)/$U" ]; then
    echo "REFUSING: a smoke would overwrite the shipped checkpoints in $U. Set RPDE_REPRO_OUT."
    exit 1; fi
  if [ -d ../sub_candidates ]; then
    echo "REFUSING: ../sub_candidates exists, so this is the working tree and the build would"
    echo "stage smoke archives beside the uploads. Run the smoke from a scratch copy of the repo."
    exit 1; fi
fi

# --- recipe fragments, read back from the shipped checkpoints' config (REPRODUCE.md section 1) ---
LOSS="--loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 \
      --lam-v 0.15 --lam-t 0.15 --alpha-train 0.0"
OPT="--stride 5 --batch 32 --lr 2e-4 --epochs $EPOCHS --wd 1e-5"
FNO="--arch fno --width 96 --n-layers 8 --modes-y 8 --modes-x 12 --dropout 0.45 \
     --var-head --var-mlp --lift-mlp $LOSS $OPT"
FFNO="--arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32 --dropout 0.3 \
      --var-head $LOSS $OPT"
VITC="--arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 --dropout 0.3 \
      --var-head $LOSS $OPT"
VITP="--arch vit --width 384 --n-layers 8 --modes-y 12 --modes-x 16 --dropout 0.3 --var-head \
      --loss mse --lam-f 1.0 --lam-grad 1.0 --lam-v 0.3 --lam-t 0.15 --alpha-train 0.0 $OPT"
SIMF="--sim $SIM --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5"

SEEDS_TW=${RPDE_REPRO_SEEDS:-"1 2 3"}       # twins:   seeds 1..3
SEEDS_ME=${RPDE_REPRO_SEEDS:-"1 2 3 4"}     # members: seeds 1..4

run() {                      # run <out.pth> <flags...>
  local out="$1"; shift
  if [ -f "$out" ]; then echo "  SKIP  $(basename "$out") (exists)"; return 0; fi
  # train2.py runs FROM $U (it imports model.py as a sibling) but must write to $OUT, which is
  # not always the same directory -- so --out is the ABSOLUTE path, not a basename. With
  # basename, an $OUT (or $P2) outside $U silently wrote the checkpoint into $U and the
  # existence check below then failed on a file that had in fact been trained.
  mkdir -p "$(dirname "$out")"
  local abs; abs="$(cd "$(dirname "$out")" && pwd)/$(basename "$out")"
  echo "  TRAIN $abs  $(date '+%F %T')"
  ( cd "$U" && $PY train2.py --data "$REAL" "$@" --out "$abs" ) \
      > "${out%.pth}.trainlog" 2>&1
  local rc=$?
  [ -f "$out" ] || { echo "  FAILED $out (rc $rc); see ${out%.pth}.trainlog"; exit 1; }
  tail -2 "${out%.pth}.trainlog"
}

# ============================== stage 1: TWINS (the band's error field) =====================
# First on purpose: a package whose band came from all-data models scored sps 36.94 (R6), so the
# twins are the part you cannot skip. Held out = model.VAL_TRAJ (no --exclude, no --all-data),
# and wherever --sim is present the twin adds --sim-exclude-held.
twins_vit() {
  echo "== ViT twins (2), shared by every package below =="
  run $OUT/arch_vit_d384.pth    $VITP --seed 1
  run $OUT/arch_vit_cmb3sim.pth $VITC --seed 1 $SIMF --sim-exclude-held
}
twins_ffno() {
  echo "== F-FNO + sim twins (3) -- the band's error field on BOTH tracks =="
  for s in $SEEDS_TW; do
    run $P2/p2_ffno_l8_sim_s$s.pth $FFNO $SIMF --sim-exclude-held --seed $s
  done
}

# ============================== stage 2: ALL-DATA MEMBERS (what ships) ======================
members_vit() {
  echo "== all-data ViT members (2), shared by every package below =="
  run $OUT/arch_vit_all_s1.pth         $VITP --seed 1 --all-data
  run $OUT/arch_vit_all_cmb3sim_s3.pth $VITC --seed 3 --all-data $SIMF
}
members_ffno() {
  echo "== all-data F-FNO + sim members (4) =="
  for s in $SEEDS_ME; do run $OUT/ffno_all_w40l8_sim_s$s.pth $FFNO $SIMF --seed $s --all-data; done
}

# ---- SUPERSEDED recipes, kept because they are what the board actually scored ---------------
# R25 (Track 1, 81.344644): FNO w96 L8 cmb3sim x4 + the same ViT pair, band r24_xfam6.npz.
# R28 / T2R13 (81.401196 / 81.314785): the F-FNO members and twins WITHOUT the sim flags.
# Reachable as `bash reproduce.sh r25` and `bash reproduce.sh r28`; not on the default path.
twins_fno_w96() {
  echo "== SUPERSEDED: FNO w96 L8 cmb3sim twins (3), R25 =="
  for s in $SEEDS_TW; do
    o=$OUT/fno_r18_cmb3sim$([ "$s" = 1 ] || echo _s$s).pth
    run "$o" $FNO --seed $s $SIMF --sim-exclude-held
  done
}
members_fno_w96() {
  echo "== SUPERSEDED: all-data FNO w96 L8 cmb3sim members (4), R25 =="
  for s in $SEEDS_ME; do run $OUT/fno_all_w96L8cmb3sim_s$s.pth $FNO --seed $s --all-data $SIMF; done
}
twins_ffno_nosim() {
  echo "== SUPERSEDED: F-FNO twins WITHOUT sim (3), R28 / T2R13 =="
  for s in $SEEDS_TW; do run $P2/p2_ffno_l8_s$s.pth $FFNO --seed $s; done
}
members_ffno_nosim() {
  echo "== SUPERSEDED: all-data F-FNO members WITHOUT sim (4), R28 / T2R13 =="
  for s in $SEEDS_ME; do run $OUT/ffno_all_w40l8_s$s.pth $FFNO --seed $s --all-data; done
}

# ============================== stage 3 + 4: BAND, BUILD, GATES =============================
# build_round13.sh does the band and both tracks' archives in one pass. RPDE_BAND reuses a
# committed band instead of refitting -- which is what a reproduction of a SCORED package must
# do, because a refit is not bit-reproducible (Track_1/bands/MANIFEST.md).
#
# It does NOT propagate its gates' exit codes (CLAUDE.md), so `build` below runs it through a
# log and reads the two SUBMITTABLE lines itself -- one per track. Without this the script
# printed DONE over a failed gate.
build() {                    # build <tag> <env assignment>...   -- last arg wins as usual
  local log; log="${RPDE_REPRO_LOG:-${TMPDIR:-/tmp}}/build_$1.log"; shift
  mkdir -p "$(dirname "$log")"
  env "$@" bash Track_1/submit_fno/build_round13.sh 0.60 2>&1 | tee "$log"
  local n; n=$(grep -c '^SUBMITTABLE$' "$log")
  echo "gates: $n/2 SUBMITTABLE   (log $log)"
  [ "$n" = "2" ] || { echo "FAILED: a gate did not pass -- read $log"; exit 1; }
}

# The band: RPDE_REPRO_FITBAND=1 fits it from the twins this run just trained (the build's
# RPDE_BAND_FILL=1 path, --fill-unscored, which is the method of record); unset reuses the
# committed band, which is what reproducing a SCORED package must do.
band_env() {                 # band_env <committed band path>
  if [ "${RPDE_REPRO_FITBAND:-0}" = "1" ]; then echo "RPDE_BAND_FILL=1"; else echo "RPDE_BAND=$1"; fi
}
build_r34() {
  echo "== band + build, BOTH tracks (R30 / T2R17, r34_p2ffnoL8sim_fill) =="
  # One call builds and gates both archives from one member set. FFNO_PATH is baked by the build
  # itself since 09-18 (RPDE_FFNO_PATH below, REPRODUCE.md 6.4 CLOSED): bake_ffno_path.py writes
  # the same block T2R13 shipped, byte for byte.
  # RPDE_BAND reuses the COMMITTED band, which is what reproducing a scored package must do -- a
  # refit is not bit-reproducible with a different twin set (REPRODUCE.md 3, bands/MANIFEST.md).
  # To fit it instead: drop RPDE_BAND and pass RPDE_BAND_FILL=1.
  local A="$OUT/ffno_all_w40l8_sim_s1.pth $OUT/ffno_all_w40l8_sim_s2.pth \
           $OUT/ffno_all_w40l8_sim_s3.pth $OUT/ffno_all_w40l8_sim_s4.pth \
           $OUT/arch_vit_all_s1.pth $OUT/arch_vit_all_cmb3sim_s3.pth"
  local M="$P2/p2_ffno_l8_sim_s1.pth $P2/p2_ffno_l8_sim_s2.pth $P2/p2_ffno_l8_sim_s3.pth \
           $OUT/arch_vit_d384.pth $OUT/arch_vit_cmb3sim.pth"
  build r34 RPDE_FORCE=1 RPDE_A="$A" RPDE_M="$M" \
    "$(band_env "${RPDE_BAND_R34:-Track_1/bands/r34_p2ffnoL8sim_fill.npz}")" \
    RPDE_INT8=1 RPDE_LINEAR=tensor RPDE_SHIP_AMP='"fp16"' RPDE_SPEC_PATH=bmm \
    RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense RPDE_TAG="r34_p2ffnoL8sim_fill"
}
build_r25() {
  echo "== SUPERSEDED: band + build, Track 1 (R25 xfam6fp16) =="
  local A="$OUT/fno_all_w96L8cmb3sim_s1.pth $OUT/fno_all_w96L8cmb3sim_s2.pth \
           $OUT/fno_all_w96L8cmb3sim_s3.pth $OUT/fno_all_w96L8cmb3sim_s4.pth \
           $OUT/arch_vit_all_s1.pth $OUT/arch_vit_all_cmb3sim_s3.pth"
  local M="$OUT/fno_r18_cmb3sim.pth $OUT/fno_r18_cmb3sim_s2.pth $OUT/fno_r18_cmb3sim_s3.pth \
           $OUT/arch_vit_d384.pth $OUT/arch_vit_cmb3sim.pth"
  build r25 RPDE_FORCE=1 RPDE_A="$A" RPDE_M="$M" \
    "$(band_env "${RPDE_BAND_T1:-Track_1/bands/r24_xfam6.npz}")" \
    RPDE_INT8=1 RPDE_LINEAR=tensor RPDE_SHIP_AMP='"fp16"' RPDE_SPEC_PATH=bmm \
    RPDE_HOST_WRITE=torch RPDE_TAG="r25_xfam6fp16_a060"
}
build_r28() {
  echo "== SUPERSEDED: band + build, R28 / T2R13 (P2-L8, no sim) =="
  local A="$OUT/ffno_all_w40l8_s1.pth $OUT/ffno_all_w40l8_s2.pth $OUT/ffno_all_w40l8_s3.pth \
           $OUT/ffno_all_w40l8_s4.pth $OUT/arch_vit_all_s1.pth $OUT/arch_vit_all_cmb3sim_s3.pth"
  local M="$P2/p2_ffno_l8_s1.pth $P2/p2_ffno_l8_s2.pth $P2/p2_ffno_l8_s3.pth \
           $OUT/arch_vit_d384.pth $OUT/arch_vit_cmb3sim.pth"
  build r28 RPDE_FORCE=1 RPDE_A="$A" RPDE_M="$M" \
    "$(band_env "${RPDE_BAND_T2:-Track_1/bands/t2r13_p2ffnoL8x4.npz}")" \
    RPDE_INT8=1 RPDE_LINEAR=tensor RPDE_SHIP_AMP='"fp16"' RPDE_SPEC_PATH=bmm \
    RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense RPDE_TAG="r27_p2ffno4_a060"
}

# ---------------------------------------------------------------------------------------------
# OPTIONAL: the distilled single-model student (STUDENT=1; OFF by default).
# Not a package of record -- no student has scored -- but a two-stage recipe is exactly the kind
# of method the re-training step cannot reproduce from one train2.py line, so its driver lives
# here beside the others rather than in a handoff. It reuses the teachers this script just
# trained (stage 1 of the pipeline skips whatever is already present), labels every training
# window with them, and trains one model against those labels: ~6 GPU-h on top of `both`.
#   STUDENT=1 RPDE_REPRODUCE_GO=1 bash reproduce.sh both
student() {
  echo "== distilled student (heldout = the evidence pipeline; pass STUDENT_MODE=alldata to ship) =="
  bash Track_1/unet/distill_pipeline.sh \
      "${STUDENT_OUT:-/fs/scratch/PAS1187/qwu/tmp/realpde/distill/reproduce}" \
      "${STUDENT_MODE:-heldout}"
}

# RPDE_REPRO_STAGE runs a subset -- the header's "split across the job cap", which needed a way
# to call one function. It trains and stops: no band, no build, no gates.
if [ -n "${RPDE_REPRO_STAGE:-}" ]; then
  for fn in $RPDE_REPRO_STAGE; do
    declare -F "$fn" > /dev/null || { echo "no such stage: $fn"; exit 2; }
    "$fn"
  done
  echo; echo "STAGES DONE: $RPDE_REPRO_STAGE  (no build -- rerun without RPDE_REPRO_STAGE)"; exit 0
fi

# t1, t2 and both are the SAME pipeline since 09-20: one member set, one build, two archives.
case "$WHICH" in
  t1|t2|both) twins_vit; twins_ffno; members_vit; members_ffno; build_r34 ;;
  r25)        twins_vit; twins_fno_w96;    members_vit; members_fno_w96;    build_r25 ;;
  r28)        twins_vit; twins_ffno_nosim; members_vit; members_ffno_nosim; build_r28 ;;
  *) echo "usage: $0 [t1|t2|both|r25|r28]   (r25 and r28 are the superseded packages)"; exit 2 ;;
esac
[ "${STUDENT:-0}" = "1" ] && student

echo
echo "DONE. Acceptance is a SCORE test, not a checksum (REPRODUCE.md section 7): the strict gates"
echo "above re-score each archive through the kit's scoring.py, and the four error subscores must"
echo "land inside the seed spread. Then compare against the uploaded archive with"
echo "  $PY Track_1/verify/pkg_identity.py --track t1 --a <extracted old> --b <extracted new>"
