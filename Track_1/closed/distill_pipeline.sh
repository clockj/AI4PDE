#!/bin/bash
# =====================================================================================
# REPRODUCIBLE PIPELINE for a distilled single-model package (forge, 2026-09-17).
#
# The organizers shortlist the top 10 from the development board and then RE-TRAIN each
# shortlisted method FROM SCRATCH on their own cluster before scoring the private test
# set at novel angles of attack and Reynolds numbers; a method they cannot reproduce is
# disqualified. A distilled student is a TWO-STAGE recipe, so "the recipe" is not one
# train2.py line -- it is this script. Everything below is seeded, every split is
# derived rather than typed, and no stage reads a file another stage did not write.
#
#   bash Track_1/unet/distill_pipeline.sh <outdir> [heldout|alldata]
#
#   heldout  the EVIDENCE pipeline: teachers and student both exclude model.VAL_TRAJ,
#            so every number it produces is honest.
#   alldata  the SHIPPING pipeline: teachers and student see all 82 trajectories.
#            All-data checkpoints supply weights, never evidence.
#
# Stage 1 trains the six teachers (~16 GPU-h), stage 2 labels every training window with
# them (~15 min), stage 3 trains the student (~6 h). Stage 1 is skipped if the teacher
# checkpoints are already present, which is how it is run here -- they were trained for
# rounds R19/R21 and their recipes are reproduced verbatim below so a third party does
# not have to take them on trust.
#
# WHY THE STUDENT EXISTS. The shipped package is six members in two invocations; the
# board charges ~1.15 ms per invocation and 1.75 board ms per 4060-equivalent local ms.
# One model that reproduces the ensemble's mean field AND its sigma ships as one
# invocation at a quarter of the FNO group's FLOPs.
# =====================================================================================
set -euo pipefail

OUT=${1:?usage: distill_pipeline.sh <outdir> [heldout|alldata]}
MODE=${2:-heldout}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-/users/PAS1187/qwu/.conda/envs/realpde/bin/python}
REAL=${REAL:-/users/PAS1187/qwu/RealPDE/train_real/train_real}
SIM=${SIM:-/users/PAS1187/qwu/RealPDE/train_sim/train_sim}
U=$REPO/Track_1/unet
mkdir -p "$OUT"

case "$MODE" in
  heldout) SPLIT="--sim-exclude-held"; LBLFLAG=""; TAG=heldout ;;
  alldata) SPLIT="--all-data";         LBLFLAG="--all-data"; TAG=alldata ;;
  *) echo "mode must be heldout or alldata"; exit 2 ;;
esac

# The recipe every FNO teacher member was trained with (CLAUDE.md, "What ships, and why").
# --lam-f 0.0 --lam-cov 2.0 replaces the MSE field term with a curvature-matched Welsch
# "coverage" loss that bends only the tail; --alpha-train 0.0 keeps compose out of training.
FNO_RECIPE="--arch fno --width 96 --n-layers 8 --modes-y 8 --modes-x 12 --var-head --var-mlp \
 --lift-mlp --loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 \
 --lam-v 0.15 --lam-t 0.15 --alpha-train 0.0 --dropout 0.45 --lr 2e-4 --batch 32 --epochs 400 \
 --stride 5 --sim $SIM --lam-sim 0.3 --sim-loss full"
# The two ViT teachers differ from each other by design: one plain-MSE, one on the cmb3sim
# losses with sim. A second family is what buys the ensemble its containment.
VIT_A="--arch vit --width 384 --n-layers 8 --dropout 0.3 --var-head --loss mse --lam-f 1.0 \
 --lam-grad 1.0 --lam-v 0.3 --lam-t 0.15 --alpha-train 0.0 --lr 2e-4 --batch 32 --epochs 400 --stride 5"
VIT_B="--arch vit --width 384 --n-layers 8 --dropout 0.3 --var-head --loss mse --lam-f 0.0 \
 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 \
 --alpha-train 0.0 --lr 2e-4 --batch 32 --epochs 400 --stride 5 --sim $SIM --lam-sim 0.3 --sim-loss full"

echo "### stage 1/3  teachers ($TAG)"
if [ "$TAG" = alldata ]; then
  T=($U/fno_all_w96L8cmb3sim_s1.pth $U/fno_all_w96L8cmb3sim_s2.pth $U/fno_all_w96L8cmb3sim_s3.pth
     $U/fno_all_w96L8cmb3sim_s4.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth)
  SEEDS=(1 2 3 4); VSEED_A=1; VSEED_B=3
else
  T=($U/fno_r18_cmb3sim.pth $U/fno_r18_cmb3sim_s2.pth $U/fno_r18_cmb3sim_s3.pth
     $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth)
  SEEDS=(1 2 3); VSEED_A=1; VSEED_B=1
fi
MISSING=0; for c in "${T[@]}"; do [ -f "$c" ] || MISSING=1; done
if [ $MISSING = 1 ]; then
  echo "    training the teachers from scratch (this is the ~16 GPU-h stage)"
  i=0
  for s in "${SEEDS[@]}"; do
    ( cd "$U" && $PY train2.py --data "$REAL" $FNO_RECIPE $SPLIT --seed "$s" --out "${T[$i]}" )
    i=$((i + 1))
  done
  ( cd "$U" && $PY train2.py --data "$REAL" $VIT_A $SPLIT --seed $VSEED_A --out "${T[$i]}" )
  ( cd "$U" && $PY train2.py --data "$REAL" $VIT_B $SPLIT --seed $VSEED_B --out "${T[$((i + 1))]}" )
else
  echo "    teachers already present; their recipes are the FNO_RECIPE / VIT_A / VIT_B lines above"
fi
for c in "${T[@]}"; do md5sum "$c"; done

echo "### stage 2/3  teacher labels"
# Stores what _fno.predict_fno averages BEFORE compose -- the member-averaged raw field and
# the member-averaged sigma, float16, in normalized space, in window_index order with the
# index saved beside them. --verify re-runs predict_fno at alpha 0 and refuses labels that do
# not reproduce the inference path. The split is derived, never typed: [] with --all-data,
# model.VAL_TRAJ otherwise, and every teacher's own config is checked against it.
$PY "$REPO/Track_1/closed/distill_labels.py" --members "${T[@]}" --out "$OUT/labels" --data "$REAL" \
    --stride 2 $LBLFLAG --sim "$SIM" --sim-stride 5 ${SPLIT/--all-data/} --batch 64 --verify 16

echo "### stage 3/3  student"
# One model, the same cmb3sim recipe, at stride 2 (2.5x the pool -- the teacher labels any
# window, so the student is not limited to the stride the members were trained at), with the
# target blended half true / half teacher and the sigma head distilled outright.
( cd "$U" && $PY train2.py --data "$REAL" $FNO_RECIPE $SPLIT --stride 2 \
    --teacher "$OUT/labels" --teacher-mix 0.5 --seed 1 --out "$OUT/student_s1.pth" )
md5sum "$OUT/student_s1.pth"
echo "### done: $OUT/student_s1.pth"
