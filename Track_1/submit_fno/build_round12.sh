#!/bin/bash
# Round 12 submission build, one command, in the only order that is correct.
#
#   bash submit_fno/build_round12.sh          # run from Track_1/, with an IDLE GPU
#
# What changed from round 11 (build_round11.sh):
#
#   * The package is MIXED: 8 all-data FNO members plus one all-data U-Net. Gate M measured the
#     U-Net member at +0.091 proxy / +0.123 full wfinal on the 7-FNO ensemble, against +0.019
#     for an extra FNO seed -- round 8's "architecture beats seeds" replicating across families.
#     It costs one invocation (nothing here shares a shape with an FNO) and ~0.4 local ms.
#   * Members load through `_fno.load_any`, which dispatches on each blob's own `arch` field.
#     build_zip.py still names every member fno*.pth; the name is positional, not a type.
#   * ALPHA 0.90 and band p 0.45 / smooth 3 are unchanged and re-confirmed on the final member
#     set at gate M (alpha grid 0.75/0.825/0.90/0.95/1.0 -> 82.549/82.598/82.623/82.626/82.614).
#
# It refuses to run unless every checkpoint exists, so a half-finished training queue cannot
# quietly produce a package built from the wrong members, and refuses while training is running,
# because validate_strict's ms/window feeds the board-time model at 0.431 board ms per local ms.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
# MUST match submission.py's ALPHA: sigma_ref and the error field the band is solved against
# both have to be measured at the alpha the package actually deploys.
ALPHA=0.90

# SHIPPED members, in the order build_zip.py names them fno.pth, fno_2.pth, ... fno_9.pth.
A="unet/fno_all_vl_s1.pth unet/fno_all_vl_s2.pth unet/fno_all_vl_s3.pth \
   unet/fno_all_vl_s4.pth unet/fno_all_vl_s5.pth \
   unet/fno_all_vl_m1220.pth unet/fno_all_vl_w48.pth unet/fno_all_vl_w64.pth \
   unet/unet_all_g1.pth"
# HELD-OUT counterparts of the same recipes, for the ERROR FIELD only. fit_band's sigma_ref
# comes from $A but |error| comes from $M: a band fitted to the all-data models' own memorized
# errors is what produced sps 36.94 in round 6. One-to-one with $A except the fifth w24 seed,
# which has no held-out twin -- the error STRUCTURE is what has to match, not the member count.
M="unet/fno_vl_s1.pth unet/fno_vl_s2.pth unet/fno_vl_s3.pth unet/fno_vl_s4.pth \
   unet/fno_vl_m1220.pth unet/fno_vl_w48.pth unet/fno_vl_w64.pth unet/unet_g1.pth"

if pgrep -f "train2\.py" > /dev/null; then
  echo "REFUSING: train2.py is running. validate_strict's ms/window would be inflated and the"
  echo "recorded board prediction would be wrong. Wait for the queue to drain."
  exit 1
fi

missing=0
for c in $A $M; do
  [ -f "$c" ] || { echo "MISSING $c"; missing=1; }
done
[ "$missing" -eq 0 ] || { echo "refusing to build from an incomplete member set"; exit 1; }

echo "=== 1/2  fit_band  (band.npz: base, sigma_ref, p) ==========================="
$PY submit_fno/fit_band.py --ckpt $A --err-ckpt $M --alpha $ALPHA --band-p 0.45

echo
echo "=== 2/2  validate_strict  (must end in SUBMITTABLE) ========================="
$PY submit_fno/validate_strict.py --ckpt $A 2>&1 | tee /tmp/r12_build.log

ms=$(grep -oP '\(\K[0-9.]+(?= ms/window\))' /tmp/r12_build.log | tail -1)
inv=$($PY - <<'EOF'
import sys, glob
sys.path.insert(0, "submit_fno")
import _fno
print(len(_fno.fuse([_fno.load_any(c, "cuda")
                     for c in sorted(glob.glob("submit_fno/_package/fno*.pth"))])))
EOF
)
cat <<MSG

=== next: record the board prediction BEFORE submitting =====================
This package: ${inv} invocations, ${ms} ms/window measured on THIS machine (A100).

board ms = 4.447 + 0.324*invocations + 0.431*local_ms was fitted with local ms measured on
the RTX 4060, so \$ms must be converted first: measure the round-11 package the same way on
the idle A100 and scale by (recorded 4060 ms) / (A100 ms) before passing --local-ms.

  \$PY predict_board.py --ckpt \$A --err-ckpt \$M --alpha ${ALPHA} \\
      --invocations ${inv} --local-ms <4060-equivalent ms>

Copy the five predicted subscores into plan/ BEFORE submitting.
MSG
