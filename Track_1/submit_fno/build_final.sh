#!/bin/bash
# FINAL RECIPE (Main Development, 2026-09-27): build the two packages of record from ONE call.
#
#   bash Track_1/submit_fno/build_final.sh OUTDIR          # repo root, GPU node
#
#   Track 1  track1_<TAG>.zip == uploaded `track1_k27fin_a080.zip`   (board 81.671149)
#   Track 2  track2_<TAG>.zip == uploaded `T2_MRGP2_track2_MRGP2.zip` (board 82.055553)
#
# Members, twins and the band are inputs; nothing is fitted here. Every shipped constant is passed explicitly, because an
# unset knob of build_round13.sh falls back to older behaviour.
#
# RPDE_REF_T1 / RPDE_REF_T2 (the uploaded archives) turn on the identity checks:
#   * the DATA inside the archives -- band.npz and the six members -- must be byte-identical to the upload;
#   * the OUTPUTS must be bit-identical (Track_1/verify/final_identity.py: Track 1 at N = 14, 56, 99, 336 in one process,
#     Track 2 over a full stream with revealed targets, under both normalisations of the revealed target).
# The CODE files differ from the upload since 2026-09-28 (guards against non-finite values, dead knobs removed); commit
# fdac00b is the tree that rebuilds them byte for byte (jobs 7569053, 7569179).
#
# build_round13.sh `sed`s constants into submission.py (both tracks) and _fno.py and leaves them there; this script
# snapshots the three files and puts them back, so the tree is unchanged when it returns.
#
# Exit 0 only if Track 1 is SUBMITTABLE, Track 2 is SUBMITTABLE as an adaptive package, no Track-2 step reads a window
# it has not been shown, and the identity checks (when asked for) pass.
set -uo pipefail
cd "$(dirname "$0")/../.."
_PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
[ -x "$_PY" ] || _PY=$(command -v python3 || echo "$_PY")
export PY=${PY:-$_PY}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1          # torch >= 2.6 refuses our checkpoints (numpy in the blob) without it
OUT=${1:?"usage: build_final.sh OUTDIR   (the uploads live in ../sub_candidates/final; build somewhere else)"}
mkdir -p "$OUT"
for v in Track_1/val_seen.npz Track_1/val_unseen.npz; do
  [ -f "$v" ] || { echo "MISSING $v -- the gates score on it. Build it first: ( cd Track_1 && \$PY make_val.py --data <train_real> )"; exit 1; }
done
TAG=${RPDE_TAG:-final_a080}
U=${RPDE_U:-Track_1/unet}
A="${RPDE_A:-$U/ffno_all_w40l8_sim_s1.pth $U/ffno_all_w40l8_sim_s2.pth $U/ffno_all_w40l8_sim_s3.pth $U/ffno_all_w40l8_sim_s4.pth $U/arch_vit_all_s1.pth $U/arch_vit_all_cmb3sim_s3.pth}"
M="${RPDE_M:-$U/p2_ffno_l8_sim_s1.pth $U/p2_ffno_l8_sim_s2.pth $U/p2_ffno_l8_sim_s3.pth $U/arch_vit_d384.pth $U/arch_vit_cmb3sim.pth}"
BAND=${RPDE_BAND:-Track_1/bands/k27_sigblend_lam0.5.npz}
SB=${RPDE_SIGBLEND:-Track_1/sigblend/band_k0_sigblend_lam0.5.json}
# The SIGBLEND npz is ~3x too wide unless the package applies the sigma-power factor, so the band and the two
# literals travel together: T1 reads EBAND (GPU-band path), T2 reads SIGPOW.
EBAND=$($PY -c "import json;print(json.load(open('$SB'))['EBAND'])") || exit 1
SIGPOW=$($PY -c "import json;q=json.load(open('$SB'))['T2_sigpow'];print('(%r, %r)' % (float(q[0]), float(q[1])))") || exit 1
echo "band $BAND  md5 $(md5sum < "$BAND" | cut -c1-8)  EBAND $EBAND  SIGPOW $SIGPOW"

KEEP=$(mktemp -d); LOG="$OUT/build_$TAG.log"
FILES="Track_1/submit_fno/submission.py Track_1/submit_fno/_fno.py Track_2/submit/submission.py"
for f in $FILES; do mkdir -p "$KEEP/$(dirname $f)"; cp -p $f "$KEEP/$f"; done
restore() { for f in $FILES; do cp -p "$KEEP/$f" $f; done; rm -rf "$KEEP"; }
trap restore EXIT

env RPDE_FORCE=${RPDE_FORCE:-0} RPDE_SUB="$OUT" RPDE_TAG="$TAG" RPDE_A="$A" RPDE_M="$M" RPDE_BAND="$BAND" \
    RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor RPDE_SHIP_AMP='"fp16"' \
    RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense \
    RPDE_SHIP_JOINT_GAIN=True "RPDE_SHIP_EBAND=$EBAND" "RPDE_SHIP_SIGPOW=$SIGPOW" \
    RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=False RPDE_SHIP_SMALL_GRAPH_LAZY=False \
    RPDE_SHIP_WARM_TAILS=True RPDE_SHIP_PIN_OUT=128 RPDE_SHIP_BATCH_CAP=56 RPDE_SHIP_ARENA_STEPS=1024 \
    RPDE_SHIP_R1=True RPDE_SHIP_R1_ASYM=True RPDE_SHIP_NUMPY_OUT=True \
    RPDE_SHIP_R1_TSM=3 RPDE_SHIP_R1_BLK_N0=20.0 RPDE_SHIP_R1_BLK=4 \
    RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_D1C_EB=True RPDE_SHIP_D1C_EB_N0=1.0 RPDE_SHIP_D1C_LEAD_N0=5.0 \
    RPDE_SHIP_VC_N0=30.0 RPDE_T2_ADAPTIVE=1 \
    bash Track_1/submit_fno/build_round13.sh 0.80 2>&1 | tee "$LOG"

RC=0
N=$(grep -ac '^SUBMITTABLE$' "$LOG")
echo "gates: $N/2 SUBMITTABLE   (log $LOG)"
[ "$N" = "2" ] || { echo "FAILED: a gate did not pass"; RC=1; }

X=$(mktemp -d)
$PY -m zipfile -e "$OUT/track2_$TAG.zip" "$X" \
  && $PY Track_2/stream_eval.py --submission "$X" --causal-check 2>&1 | tr -d '\000' | grep -a "CAUSAL\|scores" | tee -a "$LOG"
grep -aq "CAUSALITY OK" "$LOG" || { echo "FAILED: the causality check did not pass"; RC=1; }
rm -rf "$X"

# Optional: identity against the uploaded archives -- data byte for byte, outputs bit for bit.
same() { $PY - "$1" "$2" <<'PE'
import hashlib, sys, zipfile
a, b = (zipfile.ZipFile(p) for p in sys.argv[1:3])
h = lambda z: {n: hashlib.md5(z.read(n)).hexdigest() for n in sorted(z.namelist())}
ha, hb = h(a), h(b)
bad = [n for n in sorted(set(ha) | set(hb)) if ha.get(n) != hb.get(n)]
code = [n for n in bad if n.endswith(".py")]
data = [n for n in bad if not n.endswith(".py")]
print("IDENTITY", sys.argv[1], "vs", sys.argv[2], ":",
      "ALL %d FILES BYTE-IDENTICAL" % len(ha) if not bad else
      "data %s; code differs: %s" % ("BYTE-IDENTICAL" if not data else "DIFFERS %s" % data, code))
sys.exit(1 if data else 0)
PE
}
[ -n "${RPDE_REF_T1:-}" ] && { same "$OUT/track1_$TAG.zip" "$RPDE_REF_T1" || RC=1; }
[ -n "${RPDE_REF_T2:-}" ] && { same "$OUT/track2_$TAG.zip" "$RPDE_REF_T2" || RC=1; }
if [ -n "${RPDE_REF_T1:-}" ] && [ -n "${RPDE_REF_T2:-}" ]; then
  I=$(mktemp -d)
  for n in a1 b1 a2 b2; do mkdir -p "$I/$n"; done
  $PY -m zipfile -e "$RPDE_REF_T1" "$I/a1" && $PY -m zipfile -e "$OUT/track1_$TAG.zip" "$I/b1" \
    && $PY -m zipfile -e "$RPDE_REF_T2" "$I/a2" && $PY -m zipfile -e "$OUT/track2_$TAG.zip" "$I/b2" \
    && $PY Track_1/verify/final_identity.py --t1-a "$I/a1" --t1-b "$I/b1" --t2-a "$I/a2" --t2-b "$I/b2" 2>&1 \
       | tr -d '\000' | tee -a "$LOG" | grep -a "IDENTICAL\|DIFFER\|FAIL\|OK"
  [ "${PIPESTATUS[0]}" = "0" ] || { echo "FAILED: outputs differ from the uploaded packages"; RC=1; }
  rm -rf "$I"
fi
sha256sum "$OUT/track1_$TAG.zip" "$OUT/track2_$TAG.zip"
[ "$RC" = "0" ] && echo "BUILD_FINAL OK" || echo "BUILD_FINAL FAILED"
exit $RC
