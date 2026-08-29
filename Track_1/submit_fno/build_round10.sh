#!/bin/bash
# Round 10 submission build, one command, in the only order that is correct.
#
#   bash submit_fno/build_round10.sh          # run from Track_1/
#
# The steps are separated for a reason and the order matters:
#   1. fit_band.py --ckpt $A --err-ckpt $M
#        $A supplies sigma_ref (the per-window band's normalizer must be the scale the SHIPPED
#        models emit); $M supplies the error field (the all-data models trained on val_seen AND
#        val_unseen, so their local errors are memorization). Omitting --err-ckpt is the single
#        most expensive mistake available here -- it is what produced sps 36.94 in round 6.
#   2. validate_strict.py --ckpt $A
#        rebuilds the zip and re-checks it under the harshest load path. Must print SUBMITTABLE.
# It refuses to run unless every checkpoint exists, so a half-finished training queue cannot
# quietly produce a package built from the wrong members.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python

# All-data members: 5 x w24 + m1220 + w64. w48 is deliberately absent -- under fusion it earns
# back its own invocation group and nothing more (78.667 with it, 78.668 without).
A="unet/fno_all_vl_s1.pth unet/fno_all_vl_s2.pth unet/fno_all_vl_s3.pth \
   unet/fno_all_vl_s4.pth unet/fno_all_vl_s5.pth \
   unet/fno_all_vl_m1220.pth unet/fno_all_vl_w64.pth"
# Held-out counterparts of the SAME recipe, for the error field.
M="unet/fno_vmlp_lift.pth unet/fno_vmlp_lift_s2.pth unet/fno_vl_s3.pth unet/fno_vl_s4.pth \
   unet/fno_vl_m1220.pth unet/fno_vl_w64.pth"

missing=0
for c in $A $M; do
  [ -f "$c" ] || { echo "MISSING $c"; missing=1; }
done
[ "$missing" -eq 0 ] || { echo "refusing to build from an incomplete member set"; exit 1; }

echo "=== 1/2  fit_band  (band.npz: base, sigma_ref, p) ==========================="
$PY submit_fno/fit_band.py --ckpt $A --err-ckpt $M

echo
echo "=== 2/2  validate_strict  (must end in SUBMITTABLE) ========================="
$PY submit_fno/validate_strict.py --ckpt $A
