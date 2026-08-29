#!/bin/bash
# Round 11 endgame, one command.  RUN ONLY WITH AN IDLE GPU.
#
#   bash submit_fno/finalize_round11.sh          # from Track_1/
#
# validate_strict times predict() end to end, and that number goes straight into the board-time
# model at 0.431 board ms per local ms. Timing it against a training run inflates it by ~1.6x
# and the error lands in the recorded prediction, which is the one thing that must be honest.
# The script refuses to run if anything is training.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python

if pgrep -f "train2\.py" > /dev/null; then
  echo "REFUSING: train2.py is running. The timing would be inflated by ~1.6x and the recorded"
  echo "prediction would be wrong. Wait for the queue to drain."
  exit 1
fi

bash submit_fno/build_round11.sh 2>&1 | tee /tmp/r11_build.log

ms=$(grep -oP '\(\K[0-9.]+(?= ms/window\))' /tmp/r11_build.log | tail -1)
inv=$($PY - <<'EOF'
import sys, glob
sys.path.insert(0, "submit_fno")
import _fno
print(len(_fno.fuse([_fno.load(c, "cuda") for c in sorted(glob.glob("submit_fno/_package/fno*.pth"))])))
EOF
)
echo
echo "=== 3/3  recorded board prediction  (local ${ms} ms, ${inv} invocations) ======"
A="unet/fno_all_vl_s1.pth unet/fno_all_vl_s2.pth unet/fno_all_vl_s3.pth \
   unet/fno_all_vl_s4.pth unet/fno_all_vl_s5.pth \
   unet/fno_all_vl_m1220.pth unet/fno_all_vl_w48.pth unet/fno_all_vl_w64.pth"
M="unet/fno_vmlp_lift.pth unet/fno_vmlp_lift_s2.pth unet/fno_vl_s3.pth unet/fno_vl_s4.pth \
   unet/fno_vl_m1220.pth unet/fno_vl_w48.pth unet/fno_vl_w64.pth"
$PY predict_board.py --ckpt $A --err-ckpt $M --alpha 0.90 \
    --invocations "$inv" --local-ms "$ms"
echo
echo "Copy the five predicted subscores into plan/plan_20260809.md BEFORE submitting."
