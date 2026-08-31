#!/bin/bash
# Round 13: build BOTH tracks from the same 8 members, gate them, and stage them for upload.
#
#   bash Track_1/submit_fno/build_round13.sh [ALPHA]     # run from the repo root, IDLE GPU
#
# The two tracks ship the identical model on purpose: whether they score the same hidden
# trajectories is unknown, and one model on both boards answers it. `Track_2/verify_space.py`
# checks the two code paths agree (3.0e-6 physical, which is float32 batch-size noise).
#
# ALPHA is an argument because it is NOT settled. It was tuned in round 11 under the round-9
# subscore weights, which priced rel_l2 at 0.266; the eight-round refit prices it at 0.545 and
# tke at 0.170, and ALPHA is exactly a trade between those two. `alpha_board.py` re-swept on
# the held-out twins is what decides it. Changing ALPHA REQUIRES refitting the band -- both
# `sigma_ref` and the error field the band is solved against must be measured at the alpha the
# package actually deploys -- which is why this script does the two together and nothing else
# should build a package by hand.
set -uo pipefail
cd "$(dirname "$0")/../.."
ALPHA="${1:-0.90}"
PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
U=Track_1/unet
SUB=../sub_candidates

# SHIPPED: 8 all-data members, one fused invocation (7 w24 seeds + m1220, mode-padded).
A="$U/fno_all_vl_s1.pth $U/fno_all_vl_s2.pth $U/fno_all_vl_s3.pth $U/fno_all_vl_s4.pth \
   $U/fno_all_vl_s5.pth $U/fno_all_vl_s6.pth $U/fno_all_vl_s7.pth $U/fno_all_vl_m1220.pth"
# HELD-OUT twins of the same recipes, for the ERROR FIELD only. A band fitted to the all-data
# models' own memorized errors is what produced sps 36.94 in round 6.
M="$U/fno_vl_s1.pth $U/fno_vl_s2.pth $U/fno_vl_s3.pth $U/fno_vl_s4.pth $U/fno_vl_m1220.pth"

if pgrep -f "train2\.py" > /dev/null; then
  echo "REFUSING: train2.py is running -- validate_strict's ms/window would be inflated and"
  echo "the recorded board prediction would be wrong. Wait for the queue to drain."
  exit 1
fi
for c in $A $M; do [ -f "$c" ] || { echo "MISSING $c"; exit 1; }; done

echo "=== 1/5  band for ALPHA=$ALPHA ============================================"
$PY Track_1/submit_fno/fit_band.py --ckpt $A --err-ckpt $M --alpha "$ALPHA" --band-p 0.45 \
    --out Track_2/submit/band8.npz || exit 1

echo "=== 2/5  ALPHA into both submission.py files ============================="
sed -i "s/^ALPHA = .*/ALPHA = $ALPHA/" Track_1/submit_fno/submission.py Track_2/submit/submission.py
grep -H "^ALPHA = " Track_1/submit_fno/submission.py Track_2/submit/submission.py

echo "=== 3/5  Track 1: build + strict gate ==================================="
$PY Track_1/submit_fno/validate_strict.py --band Track_2/submit/band8.npz \
    --stage Track_1/submit_fno/_package8 --out Track_1/submit_fno/submission8.zip --ckpt $A

echo "=== 4/5  Track 2: build + strict gate ==================================="
$PY Track_2/build_t2.py --band Track_2/submit/band8.npz --ckpt $A
$PY Track_2/validate_t2.py --zip Track_2/submission_t2.zip

echo "=== 5/5  stage for upload =============================================="
mkdir -p "$SUB"
cp Track_1/submit_fno/submission8.zip "$SUB/track1_r13_8memb_1inv.zip"
cp Track_2/submission_t2.zip          "$SUB/track2_r01_8memb_1inv.zip"
ls -la "$SUB"
echo
echo "NEXT: record the predicted board subscores in $SUB/README.md BEFORE uploading."
