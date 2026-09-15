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
BETA="${RPDE_BETA:-0.0}"
PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
U=Track_1/unet
SUB=../sub_candidates

# The member set. `A` is what SHIPS (all-data); `M` is the HELD-OUT twins of the same recipes,
# used for the ERROR FIELD only -- a band fitted to the all-data models' own memorized errors
# is what produced sps 36.94 in round 6.
#
# Both are overridable from the environment so a new package does not need this file edited:
#   RPDE_A="..." RPDE_M="..." bash Track_1/submit_fno/build_round13.sh 0.90
# The defaults are the round-13 shipped set: 7 w24 seeds + m1220, one fused invocation.
A="${RPDE_A:-$U/fno_all_vl_s1.pth $U/fno_all_vl_s2.pth $U/fno_all_vl_s3.pth \
   $U/fno_all_vl_s4.pth $U/fno_all_vl_s5.pth $U/fno_all_vl_s6.pth $U/fno_all_vl_s7.pth \
   $U/fno_all_vl_m1220.pth}"
M="${RPDE_M:-$U/fno_vl_s1.pth $U/fno_vl_s2.pth $U/fno_vl_s3.pth $U/fno_vl_s4.pth \
   $U/fno_vl_m1220.pth}"
echo "members  (A, ships):  $A"
echo "twins    (M, band):   $M"

# `validate_strict.py` TIMES the package, and that number sets the recorded `time_score`
# prediction. Anything else on the same cpuset inflates it -- CLAUDE.md: "the 12-core cpuset,
# not the GPU, binds".
#
# The old guard was `pgrep -f train2.py`, which was right on the single-machine setup and is
# VESTIGIAL on a cluster: training runs on a compute node, so a login-node pgrep sees nothing.
# Ask Slurm instead, and only refuse for jobs that could share this node.
if command -v squeue > /dev/null 2>&1; then
  OTHER=$(squeue -M ascend -u "$USER" -h -o "%i %j %T" 2>/dev/null           | grep -c "rp_.* RUNNING" || true)
  MINE="${SLURM_JOB_ID:-none}"
  if [ "${OTHER:-0}" -gt "$([ "$MINE" = none ] && echo 0 || echo 1)" ]; then
    echo "REFUSING: other realpde Slurm jobs are RUNNING. validate_strict's ms/window would be"
    echo "inflated by cpuset contention and the recorded board prediction would be wrong."
    squeue -M ascend -u "$USER" -h -o "  %i %j %T %M" 2>/dev/null | grep "rp_"
    echo "Wait for the queue to drain, or pass RPDE_FORCE=1 to build anyway (the timing, and"
    echo "therefore the time_score prediction, will NOT be trustworthy)."
    [ "${RPDE_FORCE:-0}" = "1" ] || exit 1
  fi
elif pgrep -f "train2\.py" > /dev/null; then
  echo "REFUSING: train2.py is running on this machine."; exit 1
fi
for c in $A $M; do [ -f "$c" ] || { echo "MISSING $c"; exit 1; }; done

echo "=== 1/5  band for ALPHA=$ALPHA ============================================"
# RPDE_BAND=<path> REUSES an existing band instead of refitting.  A refit is not reproducible
# to the bit (it re-runs the models over the fitting split), and a PURE-LATENCY A/B needs the
# band the board already scored, or the four error subscores move and the experiment stops
# being about `time`.  Unset -> fit, exactly as before.
if [ -n "${RPDE_BAND:-}" ]; then
  [ -f "$RPDE_BAND" ] || { echo "MISSING band $RPDE_BAND"; exit 1; }
  cp "$RPDE_BAND" Track_2/submit/band8.npz
  echo "  REUSING band $RPDE_BAND (no refit) -- md5 $(md5sum < Track_2/submit/band8.npz)"
else
  $PY Track_1/submit_fno/fit_band.py --ckpt $A --err-ckpt $M --alpha "$ALPHA" --band-p 0.45 \
      --out Track_2/submit/band8.npz || exit 1
fi

echo "=== 2/5  ALPHA (both tracks) and BLEND_BETA (Track 1 only) ================"
sed -i "s/^ALPHA = .*/ALPHA = $ALPHA/" Track_1/submit_fno/submission.py Track_2/submit/submission.py
# FUSE is BAKED, not read from the environment: submission.py is imported by the evaluator
# in a process that never saw our shell, and two round-17 env A/Bs were silent no-ops for
# exactly that reason. Default True -- only the invocation-count experiment sets it False.
sed -i "s/^FUSE = .*/FUSE = ${RPDE_FUSE:-True}/" Track_1/submit_fno/submission.py Track_2/submit/submission.py
# AMP is BAKED the same way (round 25): None ships fp32; "fp16" ships autocast behind a
# load-time self-check against fp32 (`AMP_ACTIVE` says what runs). The value is a Python
# literal, so pass the quotes: RPDE_SHIP_AMP='"fp16"'. A job that sets it restores `AMP = None`.
sed -i "s/^AMP = .*/AMP = ${RPDE_SHIP_AMP:-None}/" Track_1/submit_fno/submission.py Track_2/submit/submission.py
grep -H "^AMP = " Track_1/submit_fno/submission.py Track_2/submit/submission.py
# GRAPH is BAKED too (round 26), and on Track 1 only: True captures predict()'s device-side chunk
# in a CUDA graph at import, verified against eager, and scales the band by (1 + GRAPH_FP) on the
# graph path so the board result fingerprints which path ran (`verify/graph_path.py`). Unset keeps
# GRAPH = False, the eager path every earlier round shipped. A job that sets it restores it.
sed -i "s/^GRAPH = .*/GRAPH = ${RPDE_SHIP_GRAPH:-False}/" Track_1/submit_fno/submission.py
if [ -n "${RPDE_SHIP_GRAPH_FP:-}" ]; then
  sed -i "s/^GRAPH_FP = .*/GRAPH_FP = $RPDE_SHIP_GRAPH_FP/" Track_1/submit_fno/submission.py
fi
grep -H "^GRAPH = \|^GRAPH_FP = " Track_1/submit_fno/submission.py
# The gated mean-blend lives only in Track 1's predict(): Track 2's step runs inside a captured
# CUDA graph, where a data-dependent early-out is not expressible. It fires only for windows
# outside the training speed range, so NOTHING LOCAL EXERCISES IT -- `verify/blend_path.py` is
# the gate for it and the strict gate below is not.
sed -i "s/^BLEND_BETA = .*/BLEND_BETA = $BETA/" Track_1/submit_fno/submission.py
grep -H "^ALPHA = " Track_1/submit_fno/submission.py Track_2/submit/submission.py
grep -H "^BLEND_BETA = " Track_1/submit_fno/submission.py

# Round 17 Q1g fast paths.  The evaluation container sets no environment variables, so these
# have to be CONSTANTS in the shipped submission.py -- written here the same way ALPHA is.
# Unset -> "" -> the pre-round-17 behaviour, so a build with no new knobs is unchanged.
# SPEC_PATH is per-track (it wins on Track 2 and loses on Track 1); HOST_WRITE is Track 1 only,
# because Track 2's ttt_step copies through pinned torch buffers and never does a numpy
# strided write.
SPECP="${RPDE_SPEC_PATH:-}"
HOSTW="${RPDE_HOST_WRITE:-}"
sed -i "s/^SPEC_PATH = .*/SPEC_PATH = \"$SPECP\"/" \
    Track_1/submit_fno/submission.py Track_2/submit/submission.py
sed -i "s/^HOST_WRITE = .*/HOST_WRITE = \"$HOSTW\"/" Track_1/submit_fno/submission.py
grep -H "^SPEC_PATH = \|^HOST_WRITE = " \
    Track_1/submit_fno/submission.py Track_2/submit/submission.py

# RPDE_FP16=1 stores the members' weights as float16. A width-24 package does not need it
# (8 members = 32 MB); a width-96 one does -- 8 x 57.0 MB is 456 MB against a 256 MB cap, and
# 8 x 28.5 MB is 228 MB. Measured cost: rel-L2 of the difference 5.5e-4 against a model rel-L2
# of 0.078, which adds in quadrature to 2e-6 of the error. The strict gate below re-scores the
# built archive through the official scoring.py, so a real degradation cannot slip past.
FP16=""; [ "${RPDE_FP16:-0}" = "1" ] && FP16="--fp16"
# RPDE_INT8=1 halves the archive AGAIN, via _fno.pack_state_dict (scales per (out, in) channel
# pair) and _fno.dequant8. A width-96 8-LAYER member is 54.3 MB fp16, so four fill 217 MB of
# the cap and eight need 434; at 29.0 MB int8 eight are 232. R18/T2R6 measured the board's
# latency slope at 0.25 / -0.04 board ms per local ms against the round-13 model's 0.803, which
# is what makes the eighth member worth having: the bill for doubling is ~0.037 `final` on
# Track 1 and ~0 on Track 2, against +0.086 of accuracy. Cost of the packing itself: rel-L2
# 3.9e-4 of an eight-member field, verified archive-against-source by verify/quant_identity.py.
[ "${RPDE_INT8:-0}" = "1" ] && FP16="--int8"
# RPDE_LINEAR = tensor | row | fp16 decides what --int8 does to a 2-D weight (round 24). The
# per-(out, in) scale reduces over range(2, dim), EMPTY for a Linear, and torch.amax(dim=())
# then reduces over every dim, so each Linear got ONE scale. No FNO member has a 2-D tensor
# (packed bytes identical in all three modes), but a ViT is 97.5 % Linear and R21 / T2R9
# shipped theirs per-tensor -- and r24e scored all three modes FREE (R21's own twins: d sps
# -0.001 / -0.005 / -0.004 for tensor / row / fp16). Unset keeps "tensor", the smallest and the
# one on the board, so an uploaded archive still rebuilds byte for byte.
[ -n "${RPDE_LINEAR:-}" ] && FP16="$FP16 --linear $RPDE_LINEAR"
# RPDE_LINEAR_ARCH="swin_unet=fp16" overrides RPDE_LINEAR per member family, matched on each blob's `arch`
# (09-14): Swin members ship fp16 Linears (verify/fuse_swin.py section 5) while the ViTs keep "tensor",
# tensor-identical to R25's. Unset, every member packs as before.
[ -n "${RPDE_LINEAR_ARCH:-}" ] && FP16="$FP16 --linear-arch $RPDE_LINEAR_ARCH"
echo "packing flags: ${FP16:-(fp32)}"
echo "=== 3/5  Track 1: build + strict gate ==================================="
$PY Track_1/submit_fno/validate_strict.py --band Track_2/submit/band8.npz $FP16 \
    --stage Track_1/submit_fno/_package8 --out Track_1/submit_fno/submission8.zip --ckpt $A
if [ "$BETA" != "0.0" ]; then
  echo "--- blend gate (validate_strict cannot see this feature) ---"
  $PY Track_1/verify/blend_path.py --dir Track_1/submit_fno/_package8 --beta "$BETA" \
    || { echo "REFUSING: the blend verifier failed"; exit 1; }
fi

echo "=== 4/5  Track 2: build + strict gate ==================================="
$PY Track_2/build_t2.py --band Track_2/submit/band8.npz $FP16 --ckpt $A
$PY Track_2/validate_t2.py --zip Track_2/submission_t2.zip

echo "=== 5/5  stage for upload =============================================="
mkdir -p "$SUB"
TAG="${RPDE_TAG:-r13_8memb_1inv}"
cp Track_1/submit_fno/submission8.zip "$SUB/track1_$TAG.zip"
cp Track_2/submission_t2.zip          "$SUB/track2_$TAG.zip"
ls -la "$SUB"
echo
echo "NEXT: record the predicted board subscores in $SUB/README.md BEFORE uploading."
echo "      then, AFTER it scores, promote the band into version control:"
echo "        cp Track_2/submit/band8.npz Track_1/bands/<round>_<tag>.npz"
echo "      and add a row to Track_1/bands/MANIFEST.md. The band is the ONLY part of a"
echo "      package that cannot be rebuilt from the checkpoints -- see that file."
