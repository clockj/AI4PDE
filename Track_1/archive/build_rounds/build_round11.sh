#!/bin/bash
# Round 11 submission build, one command, in the only order that is correct.
#
#   bash submit_fno/build_round11.sh          # run from Track_1/
#
# Same two steps and the same reasons as round 10 (see build_round10.sh); what changed is the
# member set and why:
#
#   * w48 is BACK. Round 10 dropped it on an AoA 15-20 proxy margin of 0.001 and the board
#     charged rel_l2 -0.058 and mvpe -0.122 for the cut. With mode-padded fusion it no longer
#     costs an invocation either -- the package is 3 invocations with it, same as round 10 was
#     without it, because the modes-(12,20) member now shares the width-24 group.
#   * ALPHA is 0.90, not 0.75. gain_v9.py's fine grid on the 7-member held-out ensemble puts a
#     broad plateau at 0.85-0.95, +0.098 wfinal over 0.75. fit_band is passed the same value.
#   * The band is UNCHANGED in method: band_cv_v9.py's leave-one-trajectory-out CV says pooling
#     the fitting split buys 0.000 sps and smoothing k=3 is best at every p and every fit set.
#     Its ONE change is p 0.50 -> 0.45, worth +0.026 sps at alpha 0.75 and +0.032 at alpha 0.90
#     -- inside the fold spread on either run alone, but the same sign on two independent error
#     fields, and free. Do not re-tune anything else.
#
# It refuses to run unless every checkpoint exists, so a half-finished training queue cannot
# quietly produce a package built from the wrong members.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
# MUST match submission.py's ALPHA: sigma_ref and the error field the band is solved against
# both have to be measured at the alpha the package actually deploys.
ALPHA=0.90

# All-data members, in the order build_zip.py names them fno.pth, fno_2.pth, ...
A="unet/fno_all_vl_s1.pth unet/fno_all_vl_s2.pth unet/fno_all_vl_s3.pth \
   unet/fno_all_vl_s4.pth unet/fno_all_vl_s5.pth \
   unet/fno_all_vl_m1220.pth unet/fno_all_vl_w48.pth unet/fno_all_vl_w64.pth"
# Held-out counterparts of the SAME recipe, for the error field. These must match the shipped
# architecture: fit_band.py's sigma_ref comes from $A but the |error| field comes from $M, and
# a band fitted to the all-data models' own (memorized) errors is what produced sps 36.94 in
# round 6.
M="unet/fno_vmlp_lift.pth unet/fno_vmlp_lift_s2.pth unet/fno_vl_s3.pth unet/fno_vl_s4.pth \
   unet/fno_vl_m1220.pth unet/fno_vl_w48.pth unet/fno_vl_w64.pth"

missing=0
for c in $A $M; do
  [ -f "$c" ] || { echo "MISSING $c"; missing=1; }
done
[ "$missing" -eq 0 ] || { echo "refusing to build from an incomplete member set"; exit 1; }

echo "=== 1/2  fit_band  (band.npz: base, sigma_ref, p) ==========================="
$PY submit_fno/fit_band.py --ckpt $A --err-ckpt $M --alpha $ALPHA --band-p 0.45

echo
echo "=== 2/2  validate_strict  (must end in SUBMITTABLE) ========================="
$PY submit_fno/validate_strict.py --ckpt $A
